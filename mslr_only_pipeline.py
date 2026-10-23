"""
================================================================================
β-VAE Pipeline -- MSLR-WEB30K ONLY, robustness-focused
================================================================================
Leaner version of the unified pipeline, built specifically because the full
pipeline (with disentanglement metrics, t-SNE, sweeps) kept crashing on
MSLR-WEB30K "due to many reasons." Differences from the unified pipeline:

  1. MSLR-only -- no Yahoo branch, no dataset-switching logic to get wrong.
  2. Skips disentanglement metrics (MIG/SAP/whole-block probe/position-
     classifier probe) and t-SNE plotting entirely. You already have
     validated MSLR disentanglement results from an earlier run; those
     don't depend on Supervised-Only/InfoRank and don't need re-running.
     This alone removes several slow, memory-heavy steps (RandomForest on
     60k+ documents, t-SNE, matplotlib) that are likely contributors to
     instability.
  3. Explicit GPU memory cleanup (del + torch.cuda.empty_cache()) after
     EVERY method, not just in the sweep functions.
  4. try/except around every method's training: if one method crashes
     (OOM, numerical issue, anything), the script logs it, frees memory,
     and moves on to the next method rather than losing everything.
  5. Results are saved to JSON incrementally, after every method finishes
     -- if the whole process is killed partway through, whatever succeeded
     so far is already on disk.
  6. Includes the direct beta-VAE vs. Supervised-Only paired comparison
     (isolates the disentanglement mechanism's contribution beyond label
     supervision alone -- addresses reviewer comment R5-1).

Trains and evaluates NINE methods:
    1. Naive Baseline        6. Affine (estimated)
    2. Supervised-Only       7. IPS (estimated)
    3. IPS (ground-truth)    8. InfoRank
    4. Regression-EM         9. beta-VAE (proposed)
    5. Bayes-IPS + Affine (ground-truth)

HOW TO USE
  1. Set MSLR_TRAIN_FILE / MSLR_VALI_FILE / MSLR_TEST_FILE below.
  2. pip install torch numpy scipy scikit-learn
  3. Run: python mslr_only_pipeline.py
  4. Results print to console AND save incrementally to results_mslr.json
     after each method -- check that file if the run is interrupted.
================================================================================
"""

import os
import json
import random
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import ttest_rel

# ============================================================================
# CONFIG -- EDIT THESE PATHS
# ============================================================================
MSLR_TRAIN_FILE = r"C:\Users\amalajkamal\Downloads\MSLR-WEB30K\Fold1\train.txt"
MSLR_VALI_FILE  = r"C:\Users\amalajkamal\Downloads\MSLR-WEB30K\Fold1\vali.txt"
MSLR_TEST_FILE  = r"C:\Users\amalajkamal\Downloads\MSLR-WEB30K\Fold1\test.txt"

RESULTS_FILE = "results_mslr.json"

SEED = 42
K_MAX_POSITION = 10
EPOCHS_VAE = 80
EPOCHS_BASELINE = 50
BASELINE_PATIENCE = 10
EARLY_STOP_PATIENCE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")


def free_gpu():
    """Call after every method to release GPU memory before the next one."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def save_results_incremental(results, ttests=None, extra=None):
    """Save whatever we have so far -- survives a crash partway through."""
    payload = {"dataset": "mslr", "results": results}
    if ttests is not None:
        payload["ttests"] = ttests
    if extra is not None:
        payload.update(extra)
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================================
# PART 1 -- DATA LOADING
# ============================================================================
class LTRDataset:
    def __init__(self, filepath, scaler=None, feature_dim=None):
        self.rows = []
        max_idx = 0
        with open(filepath, "r") as f:
            for line in f:
                parsed = self._parse_line(line)
                if parsed:
                    self.rows.append(parsed)
                    _, _, feats = parsed
                    if feats:
                        max_idx = max(max_idx, max(feats.keys()))

        self.feature_dim = feature_dim if feature_dim is not None else max_idx + 1
        self._to_matrix()

        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(self.features)
        else:
            self.scaler = scaler
        self.features = self.scaler.transform(self.features).astype(np.float32)
        self._group_by_query()

    @staticmethod
    def _parse_line(line):
        parts = line.strip().split("#")[0].strip().split()
        if len(parts) < 2:
            return None
        rel = int(parts[0])
        qid = parts[1].split(":")[1]
        feats = {}
        for tok in parts[2:]:
            if ":" in tok:
                idx, val = tok.split(":")
                feats[int(idx) - 1] = float(val)
        return qid, rel, feats

    def _to_matrix(self):
        feat_matrix, qids, rels = [], [], []
        for qid, rel, fd in self.rows:
            arr = np.zeros(self.feature_dim, dtype=np.float32)
            for i, v in fd.items():
                if i < self.feature_dim:
                    arr[i] = v
            feat_matrix.append(arr)
            qids.append(qid)
            rels.append(rel)
        self.features = np.stack(feat_matrix, axis=0)
        self.qids = np.array(qids)
        self.rels = np.array(rels)

    def _group_by_query(self):
        self.queries = {}
        for i, qid in enumerate(self.qids):
            self.queries.setdefault(qid, []).append(i)


# ============================================================================
# PART 2 -- MODEL ARCHITECTURE (exact per Section 4.2-4.3)
# ============================================================================
class RelevanceEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 96), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(96, 64), nn.ReLU(),
        )
        self.mu = nn.Linear(64, 24)
        self.logvar = nn.Linear(64, 24)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -10, 10)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar


class BiasEncoder(nn.Module):
    def __init__(self, in_dim, pos_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + pos_dim, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 48), nn.ReLU(),
        )
        self.mu = nn.Linear(48, 16)
        self.logvar = nn.Linear(48, 16)

    def forward(self, x, pos_emb):
        h = self.net(torch.cat([x, pos_emb], dim=1))
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -10, 10)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, zr, zb, pos_emb):
        x = torch.cat([zr, zb, pos_emb], dim=1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class RankingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, zr):
        return self.net(zr).squeeze(-1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, zr):
        return self.net(zr)


class PositionPredictor(nn.Module):
    def __init__(self, zb_dim=16, n_positions=None):
        super().__init__()
        n_positions = n_positions or K_MAX_POSITION
        self.net = nn.Sequential(
            nn.Linear(zb_dim, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_positions),
        )

    def forward(self, zb):
        return self.net(zb)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SimpleRanker(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class InfoRankNet(nn.Module):
    """InfoRank (Jin et al., WWW 2024). See unified pipeline for full docstring."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.obs_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.rel_head = nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h = self.trunk(x)
        p_o1 = torch.sigmoid(self.obs_head(h)).squeeze(-1)
        n = x.size(0)
        o1 = torch.ones(n, 1, device=x.device)
        o0 = torch.zeros(n, 1, device=x.device)
        p_r_o1 = torch.sigmoid(self.rel_head(torch.cat([h, o1], dim=1))).squeeze(-1)
        p_r_o0 = torch.sigmoid(self.rel_head(torch.cat([h, o0], dim=1))).squeeze(-1)
        return p_o1, p_r_o1, p_r_o0


def inforank_losses(p_o1, p_r_o1, p_r_o0, clicks, eps=1e-7):
    p_c = torch.clamp(p_r_o0 * p_o1, eps, 1 - eps)
    L = F.binary_cross_entropy(p_c, clicks)
    p_o0 = 1 - p_o1
    p_r0_o1 = 1 - p_r_o1
    p_r0_o0 = 1 - p_r_o0
    p_r1 = p_r_o1 * p_o1 + p_r_o0 * p_o0
    p_r0 = 1 - p_r1
    p_r_o1_c = torch.clamp(p_r_o1, eps, 1 - eps)
    p_r0_o1_c = torch.clamp(p_r0_o1, eps, 1 - eps)
    p_r_o0_c = torch.clamp(p_r_o0, eps, 1 - eps)
    p_r0_o0_c = torch.clamp(p_r0_o0, eps, 1 - eps)
    p_r1_c = torch.clamp(p_r1, eps, 1 - eps)
    p_r0_c = torch.clamp(p_r0, eps, 1 - eps)
    I = (p_r_o1_c * p_o1 * torch.log(p_r_o1_c / p_r1_c) +
         p_r0_o1_c * p_o1 * torch.log(p_r0_o1_c / p_r0_c) +
         p_r_o0_c * p_o0 * torch.log(p_r_o0_c / p_r1_c) +
         p_r0_o0_c * p_o0 * torch.log(p_r0_o0_c / p_r0_c))
    return L, I.mean(), p_r1


# ============================================================================
# PART 3 -- LOSSES
# ============================================================================
def free_bits_kl(mu, logvar, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    return kl_clamped.sum(dim=1).mean()


def cov_penalty(zr, zb):
    zr_c = zr - zr.mean(dim=0)
    zb_c = zb - zb.mean(dim=0)
    cov = (zr_c.T @ zb_c) / max(zr.size(0), 1)
    return (cov ** 2).sum()


def pairwise_hinge_loss(scores, rels, margin=0.5):
    n = len(rels)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    order = np.argsort(-rels)
    losses = []
    for i in range(n):
        for j in range(n):
            if rels[order[i]] > rels[order[j]]:
                losses.append(F.relu(margin + scores[order[j]] - scores[order[i]]))
    if not losses:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(losses).mean()


# ============================================================================
# PART 4 -- CLICK SIMULATION (affine model, Section 4.6.5)
# ============================================================================
def theta_k(k):
    return k ** (-0.9)


def eps_plus_k(k):
    return 1 - (k + 1) / 100.0


def eps_minus_k(k):
    return 0.65 / min(k, 10)


def gamma(y):
    return 1.0 if y > 2 else 0.0


def simulate_clicks(rels, positions):
    clicks = []
    for y, k in zip(rels, positions):
        k = min(k, K_MAX_POSITION)
        p = theta_k(k) * (eps_plus_k(k) * gamma(y) + eps_minus_k(k) * (1 - gamma(y)))
        clicks.append(1 if random.random() < p else 0)
    return clicks


# ============================================================================
# PART 5 -- EVALUATION METRICS
# ============================================================================
def ndcg_at_k(true_rels, pred_scores, k=10):
    k = min(k, len(true_rels))
    order = np.argsort(-pred_scores)[:k]
    dcg = np.sum((2 ** true_rels[order] - 1) / np.log2(np.arange(2, k + 2)))
    ideal = np.argsort(-true_rels)[:k]
    idcg = np.sum((2 ** true_rels[ideal] - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def ips_dcg_for_query(true_rels, pred_scores, k=10):
    k = min(k, len(true_rels))
    order = np.argsort(-pred_scores)[:k]
    ranked_rels = true_rels[order]
    positions = np.arange(1, k + 1)
    clicks = simulate_clicks(ranked_rels, positions)
    thetas = np.array([theta_k(min(p, K_MAX_POSITION)) for p in positions])
    gains = (2 ** ranked_rels - 1) / np.log2(positions + 1)
    return np.sum(np.array(clicks) / thetas * gains)


def paired_ttest_report(name, scores_method, scores_naive):
    scores_method = np.array(scores_method)
    scores_naive = np.array(scores_naive)
    t_stat, p_val = ttest_rel(scores_method, scores_naive)
    mean_diff = scores_method.mean() - scores_naive.mean()
    return {"method": name, "mean_diff": float(mean_diff),
            "t_stat": float(t_stat), "p_value": float(p_val)}


def evaluate_ranker_fn(score_fn, test_set, k=10, n_runs_ips=20):
    ndcgs, ips_dcgs = [], []
    for qid, idxs in test_set.queries.items():
        feats = test_set.features[idxs]
        rels = test_set.rels[idxs]
        if len(rels) < 2:
            continue
        scores = score_fn(feats)
        ndcgs.append(ndcg_at_k(rels, scores, k))
        run_vals = [ips_dcg_for_query(rels, scores, k) for _ in range(n_runs_ips)]
        ips_dcgs.append(np.mean(run_vals))
    return np.array(ndcgs), np.array(ips_dcgs)


def _quick_val_ndcg(model, vali_set):
    model.eval()
    vals = []
    with torch.no_grad():
        for qid, idxs in vali_set.queries.items():
            feats = vali_set.features[idxs]
            rels = vali_set.rels[idxs]
            if len(rels) < 2:
                continue
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            scores = model(x).cpu().numpy()
            vals.append(ndcg_at_k(rels, scores, 10))
    return np.mean(vals) if vals else 0.0


def _precompute_training_clicks(dataset):
    cache = {}
    for qid, idxs in dataset.queries.items():
        feats = dataset.features[idxs]
        rels = dataset.rels[idxs]
        if len(rels) < 2:
            continue
        order = np.argsort(-rels)[:K_MAX_POSITION]
        ranked_feats = feats[order]
        ranked_rels = rels[order]
        positions = np.arange(1, len(ranked_rels) + 1)
        clicks = simulate_clicks(ranked_rels, positions)
        cache[qid] = (ranked_feats, np.array(clicks, dtype=np.float32))
    return cache


# ============================================================================
# PART 6 -- BASELINE TRAINERS
# ============================================================================
def train_naive(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE):
    print("\n[Naive Baseline] Training...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            loss = F.binary_cross_entropy(score, c)
            opt.zero_grad(); loss.backward(); opt.step()
        val_ndcg = _quick_val_ndcg(model, vali_set)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_supervised_only(train_set, vali_set, epochs=EPOCHS_BASELINE):
    print("\n[Supervised-Only] Training (labels only, no clicks)...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, idxs in train_set.queries.items():
            feats = train_set.features[idxs]
            rels = train_set.rels[idxs]
            if len(rels) < 2:
                continue
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            y = torch.tensor(rels, dtype=torch.float32, device=DEVICE)
            score = model(x)
            point = F.mse_loss(score, y)
            pair = pairwise_hinge_loss(score, rels)
            loss = point + 0.5 * pair
            opt.zero_grad(); loss.backward(); opt.step()
        val_ndcg = _quick_val_ndcg(model, vali_set)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_ips(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE):
    print("\n[IPS] Training (ground-truth propensities)...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            k_idx = np.arange(1, len(clicks) + 1)
            w = torch.tensor(1.0 / theta_k(np.minimum(k_idx, K_MAX_POSITION)), dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            per_doc_loss = F.binary_cross_entropy(score, torch.ones_like(score), reduction="none")
            loss = (c * w * per_doc_loss).sum() / max(c.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        val_ndcg = _quick_val_ndcg(model, vali_set)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_ips_estimated(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE, random_frac=0.10):
    print("\n[IPS-Estimated] Estimating propensities via intervention harvesting...")
    qids = list(train_set.queries.keys())
    random.shuffle(qids)
    randomized_qids = set(qids[:int(len(qids) * random_frac)])
    click_counts, exam_counts = defaultdict(int), defaultdict(int)
    for qid in randomized_qids:
        idxs = train_set.queries[qid]
        feats = train_set.features[idxs]
        rels = train_set.rels[idxs]
        if len(rels) < 2:
            continue
        perm = np.random.permutation(len(rels))[:K_MAX_POSITION]
        ranked_rels = rels[perm]
        positions = np.arange(1, len(ranked_rels) + 1)
        clicks = simulate_clicks(ranked_rels, positions)
        for k, c in zip(positions, clicks):
            exam_counts[k] += 1
            click_counts[k] += c
    theta_hat = {}
    base_rate = max(click_counts.get(1, 1) / max(exam_counts.get(1, 1), 1), 1e-3)
    for k in range(1, K_MAX_POSITION + 1):
        theta_hat[k] = max(click_counts[k] / exam_counts[k] / base_rate, 1e-3) if exam_counts.get(k, 0) > 0 else theta_k(k)
    theta_hat[K_MAX_POSITION] = min(theta_hat[K_MAX_POSITION], 1.0)

    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            k_idx = np.arange(1, len(clicks) + 1)
            w_vals = np.array([1.0 / theta_hat[min(k, K_MAX_POSITION)] for k in k_idx])
            w = torch.tensor(w_vals, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            per_doc_loss = F.binary_cross_entropy(score, torch.ones_like(score), reduction="none")
            loss = (c * w * per_doc_loss).sum() / max(c.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        val_ndcg = _quick_val_ndcg(model, vali_set)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_regression_em(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE, em_iterations=5, ema_alpha=0.5):
    print("\n[Regression-EM] Training...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    theta_est = {k: theta_k(k) for k in range(1, K_MAX_POSITION + 1)}
    epochs_per_em = max(1, epochs // em_iterations)
    best_ndcg, best_state = -1, None
    for em_iter in range(em_iterations):
        for ep in range(epochs_per_em):
            model.train()
            for qid, (feats, clicks) in click_cache.items():
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
                k_idx = np.arange(1, len(clicks) + 1)
                w_vals = np.array([1.0 / theta_est[min(k, K_MAX_POSITION)] for k in k_idx])
                w = torch.tensor(w_vals, dtype=torch.float32, device=DEVICE)
                score = torch.sigmoid(model(x))
                per_doc_loss = F.binary_cross_entropy(score, torch.ones_like(score), reduction="none")
                loss = (c * w * per_doc_loss).sum() / max(c.sum().item(), 1.0)
                opt.zero_grad(); loss.backward(); opt.step()
            val_ndcg = _quick_val_ndcg(model, vali_set)
            if val_ndcg > best_ndcg:
                best_ndcg, best_state = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}
        model.eval()
        click_sum, exam_sum, rel_sum = defaultdict(float), defaultdict(float), defaultdict(float)
        with torch.no_grad():
            for qid, (feats, clicks) in click_cache.items():
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                pred_rel = torch.sigmoid(model(x)).cpu().numpy()
                for i, (c, r) in enumerate(zip(clicks, pred_rel)):
                    k = min(i + 1, K_MAX_POSITION)
                    click_sum[k] += c; exam_sum[k] += 1; rel_sum[k] += r
        for k in range(1, K_MAX_POSITION + 1):
            if exam_sum[k] > 0 and rel_sum[k] > 1e-6:
                new_est = float(np.clip(click_sum[k] / rel_sum[k], 1e-3, 1.0))
                theta_est[k] = ema_alpha * new_est + (1 - ema_alpha) * theta_est[k]
        print(f"  EM iter {em_iter+1}/{em_iterations}  val NDCG@10={val_ndcg:.4f}  (best={best_ndcg:.4f})")
    model.load_state_dict(best_state)
    return model


def train_bayes_ips(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE):
    print("\n[Bayes-IPS] Training...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            k_idx = np.minimum(np.arange(1, len(clicks) + 1), K_MAX_POSITION)
            ep_pos = np.array([eps_plus_k(k) for k in k_idx])
            ep_neg = np.array([eps_minus_k(k) for k in k_idx])
            th = np.array([theta_k(k) for k in k_idx])
            w_vals = (1.0 / th) * (ep_pos / (ep_pos + ep_neg))
            w = torch.tensor(w_vals, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            per_doc_loss = F.binary_cross_entropy(score, torch.ones_like(score), reduction="none")
            loss = (c * w * per_doc_loss).sum() / max(c.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        val_ndcg = _quick_val_ndcg(model, vali_set)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_affine(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE, estimated=False, random_frac=0.10):
    tag = "Estimated" if estimated else "Ground-truth"
    print(f"\n[Affine-{tag}] Training...")
    if estimated:
        qids = list(train_set.queries.keys())
        random.shuffle(qids)
        randomized_qids = set(qids[:int(len(qids) * random_frac)])
        click_c, exam_c, rel_c = defaultdict(float), defaultdict(float), defaultdict(float)
        for qid in randomized_qids:
            idxs = train_set.queries[qid]
            feats = train_set.features[idxs]
            rels = train_set.rels[idxs]
            if len(rels) < 2:
                continue
            perm = np.random.permutation(len(rels))[:K_MAX_POSITION]
            ranked_rels = rels[perm]
            positions = np.arange(1, len(ranked_rels) + 1)
            clicks = simulate_clicks(ranked_rels, positions)
            for k, c, y in zip(positions, clicks, ranked_rels):
                exam_c[k] += 1; click_c[k] += c; rel_c[k] += gamma(y)
        alpha_hat, beta_hat = {}, {}
        for k in range(1, K_MAX_POSITION + 1):
            if exam_c.get(k, 0) > 5:
                mean_click = click_c[k] / exam_c[k]
                mean_rel = max(rel_c[k] / exam_c[k], 1e-3)
                beta_hat[k] = max(min(mean_click * (1 - mean_rel), mean_click), 0.0)
                alpha_hat[k] = max((mean_click - beta_hat[k]) / mean_rel, 1e-3)
            else:
                alpha_hat[k] = theta_k(k) * (eps_plus_k(k) - eps_minus_k(k))
                beta_hat[k] = theta_k(k) * eps_minus_k(k)
        alpha_fn = lambda k: alpha_hat[min(k, K_MAX_POSITION)]
        beta_fn = lambda k: beta_hat[min(k, K_MAX_POSITION)]
    else:
        alpha_fn = lambda k: theta_k(min(k, K_MAX_POSITION)) * (eps_plus_k(min(k, K_MAX_POSITION)) - eps_minus_k(min(k, K_MAX_POSITION)))
        beta_fn = lambda k: theta_k(min(k, K_MAX_POSITION)) * eps_minus_k(min(k, K_MAX_POSITION))

    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = np.array(clicks)
            k_idx = np.arange(1, len(clicks) + 1)
            alpha_vals = np.array([alpha_fn(k) for k in k_idx])
            beta_vals = np.array([beta_fn(k) for k in k_idx])
            target = np.clip((c - beta_vals) / np.maximum(alpha_vals, 1e-6), 0.0, 1.0)
            target_t = torch.tensor(target, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            loss = F.mse_loss(score, target_t)
            opt.zero_grad(); loss.backward(); opt.step()
        val_ndcg = _quick_val_ndcg(model, vali_set)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_inforank(train_set, vali_set, click_cache, epochs=EPOCHS_BASELINE, eta=0.5):
    print(f"\n[InfoRank] Training (eta={eta})...")
    model = InfoRankNet(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            p_o1, p_r_o1, p_r_o0 = model(x)
            L, I, _ = inforank_losses(p_o1, p_r_o1, p_r_o0, c)
            loss = L + eta * I
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for vqid, idxs in vali_set.queries.items():
                feats_v = vali_set.features[idxs]
                rels_v = vali_set.rels[idxs]
                if len(rels_v) < 2:
                    continue
                xv = torch.tensor(feats_v, dtype=torch.float32, device=DEVICE)
                _, p_r_o1_v, p_r_o0_v = model(xv)
                p_o1_v = torch.sigmoid(model.obs_head(model.trunk(xv))).squeeze(-1)
                p_r1_v = (p_r_o1_v * p_o1_v + p_r_o0_v * (1 - p_o1_v)).cpu().numpy()
                vals.append(ndcg_at_k(rels_v, p_r1_v, 10))
        val_ndcg = np.mean(vals) if vals else 0.0
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def make_score_fn_inforank(model):
    def fn(feats_np):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            p_o1, p_r_o1, p_r_o0 = model(x)
            p_r1 = p_r_o1 * p_o1 + p_r_o0 * (1 - p_o1)
            return p_r1.cpu().numpy()
    return fn


# ============================================================================
# PART 7 -- BETA-VAE TRAINER
# ============================================================================
def train_beta_vae(train_set, vali_set, click_cache, dataset_rels_lookup, epochs=EPOCHS_VAE):
    print("\n[beta-VAE (proposed)] Training...")
    enc_r = RelevanceEncoder(train_set.feature_dim).to(DEVICE)
    enc_b = BiasEncoder(train_set.feature_dim).to(DEVICE)
    dec = Decoder().to(DEVICE)
    rank_head = RankingHead().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    pos_predictor = PositionPredictor(zb_dim=16, n_positions=K_MAX_POSITION).to(DEVICE)
    pos_emb = nn.Embedding(K_MAX_POSITION, 8).to(DEVICE)

    gen_params = (list(enc_r.parameters()) + list(enc_b.parameters()) +
                  list(dec.parameters()) + list(rank_head.parameters()) +
                  list(pos_emb.parameters()) + list(pos_predictor.parameters()))
    opt_g = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    lambda_rank, lambda_cov = 2.0, 0.001
    lambda_r_max, lambda_b_max = 0.6, 0.3
    lambda_bias_pred = 0.5
    free_bits = 0.5

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        t = ep
        if t < 20:
            lambda_r, lambda_b = 0.0, 0.0
        elif t < 60:
            frac = (t - 20) / 40
            lambda_r, lambda_b = lambda_r_max * frac, lambda_b_max * frac
        else:
            lambda_r, lambda_b = lambda_r_max, lambda_b_max
        lambda_adv = 0.1 + 0.9 * (t / max(epochs - 1, 1))

        enc_r.train(); enc_b.train(); dec.train(); rank_head.train(); disc.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            n = len(clicks)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, K_MAX_POSITION - 1)
            pos_vec = pos_emb(k_idx)
            y_pos = (k_idx >= (K_MAX_POSITION // 2)).long()

            with torch.no_grad():
                zr_detached, _, _ = enc_r(x)
            for _ in range(3):
                d_logits = disc(zr_detached)
                d_loss = F.cross_entropy(d_logits, y_pos)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            zr, mu_r, logvar_r = enc_r(x)
            zb, mu_b, logvar_b = enc_b(x, pos_vec)
            c_hat = dec(zr, zb, pos_vec)
            score = rank_head(zr)

            y_true = torch.tensor(
                [dataset_rels_lookup[qid][i] for i in range(n)],
                dtype=torch.float32, device=DEVICE
            )

            l_rec = F.binary_cross_entropy(c_hat, c)
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, y_true.detach().cpu().numpy())
            l_rank = l_point + 0.5 * l_pair
            l_kl_r = free_bits_kl(mu_r, logvar_r, free_bits)
            l_kl_b = free_bits_kl(mu_b, logvar_b, free_bits)
            l_cov = cov_penalty(zr, zb)

            zr_grl = grad_reverse(zr, lambda_adv)
            d_logits_g = disc(zr_grl)
            l_adv = F.cross_entropy(d_logits_g, y_pos)

            pos_logits_b = pos_predictor(zb)
            l_bias_pred = F.cross_entropy(pos_logits_b, k_idx.long())

            loss = (l_rec + lambda_rank * l_rank + lambda_r * l_kl_r +
                    lambda_b * l_kl_b + lambda_cov * l_cov + lambda_adv * l_adv +
                    lambda_bias_pred * l_bias_pred)

            opt_g.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt_g.step()

        enc_r.eval(); rank_head.eval()
        val_scores = []
        with torch.no_grad():
            for vqid, idxs in vali_set.queries.items():
                feats = vali_set.features[idxs]
                rels = vali_set.rels[idxs]
                if len(rels) < 2:
                    continue
                xv = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                _, mu_r_v, _ = enc_r(xv)
                sv = rank_head(mu_r_v).cpu().numpy()
                val_scores.append(ndcg_at_k(rels, sv, 10))
        val_ndcg = np.mean(val_scores) if val_scores else 0.0
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  lambda_r={lambda_r:.3f} lambda_b={lambda_b:.3f} "
                  f"lambda_adv={lambda_adv:.3f}  val NDCG@10={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {
                "enc_r": {k: v.clone() for k, v in enc_r.state_dict().items()},
                "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    enc_r.load_state_dict(best_state["enc_r"])
    rank_head.load_state_dict(best_state["rank_head"])
    return enc_r, rank_head


def make_score_fn_simple(model):
    def fn(feats_np):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            return model(x).cpu().numpy()
    return fn


def make_score_fn_vae(enc_r, rank_head):
    def fn(feats_np):
        enc_r.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            _, mu_r, _ = enc_r(x)
            return rank_head(mu_r).cpu().numpy()
    return fn


# ============================================================================
# MAIN -- crash-resilient: each method wrapped in try/except, results saved
# incrementally, GPU memory freed after every method.
# ============================================================================
def main():
    print("Loading MSLR-WEB30K...")
    train_set = LTRDataset(MSLR_TRAIN_FILE)
    vali_set = LTRDataset(MSLR_VALI_FILE, scaler=train_set.scaler, feature_dim=train_set.feature_dim)
    test_set = LTRDataset(MSLR_TEST_FILE, scaler=train_set.scaler, feature_dim=train_set.feature_dim)
    print(f"  train queries={len(train_set.queries)}  vali queries={len(vali_set.queries)}  "
          f"test queries={len(test_set.queries)}  feature_dim={train_set.feature_dim}")

    click_cache = _precompute_training_clicks(train_set)

    dataset_rels_lookup = {}
    for qid, (feats, clicks) in click_cache.items():
        idxs = train_set.queries[qid]
        rels_full = train_set.rels[idxs]
        order = np.argsort(-rels_full)[:K_MAX_POSITION]
        dataset_rels_lookup[qid] = rels_full[order]

    results = {}
    per_query_scores = {}
    scorers = {}

    # Each entry: (name, callable that returns a trained model or (enc_r, rank_head))
    def run_method(name, train_fn, score_fn_builder):
        print("\n" + "=" * 80)
        print(f"METHOD: {name}")
        print("=" * 80)
        try:
            trained = train_fn()
            score_fn = score_fn_builder(trained)
            scorers[name] = score_fn
            ndcgs, ips_dcgs = evaluate_ranker_fn(score_fn, test_set, k=10, n_runs_ips=20)
            per_query_scores[name] = {"ndcg": ndcgs, "ips_dcg": ips_dcgs}
            results[name] = {"NDCG@10": float(np.mean(ndcgs)), "IPS-DCG": float(np.mean(ips_dcgs))}
            print(f"{name:28s}  NDCG@10={np.mean(ndcgs):.4f}  IPS-DCG={np.mean(ips_dcgs):.4f}")
            save_results_incremental(results)
        except Exception as e:
            print(f"!!! {name} FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}
            save_results_incremental(results)
        finally:
            free_gpu()

    run_method("Naive Baseline", lambda: train_naive(train_set, vali_set, click_cache), make_score_fn_simple)
    run_method("Supervised-Only", lambda: train_supervised_only(train_set, vali_set), make_score_fn_simple)
    run_method("IPS (ground-truth)", lambda: train_ips(train_set, vali_set, click_cache), make_score_fn_simple)
    run_method("Regression-EM", lambda: train_regression_em(train_set, vali_set, click_cache), make_score_fn_simple)
    run_method("Bayes-IPS", lambda: train_bayes_ips(train_set, vali_set, click_cache), make_score_fn_simple)
    run_method("Affine (ground-truth)", lambda: train_affine(train_set, vali_set, click_cache, estimated=False), make_score_fn_simple)
    run_method("Affine (estimated)", lambda: train_affine(train_set, vali_set, click_cache, estimated=True), make_score_fn_simple)
    run_method("IPS (estimated)", lambda: train_ips_estimated(train_set, vali_set, click_cache), make_score_fn_simple)
    run_method("InfoRank", lambda: train_inforank(train_set, vali_set, click_cache), make_score_fn_inforank)

    print("\n" + "=" * 80)
    print("METHOD: beta-VAE (Ours)")
    print("=" * 80)
    try:
        enc_r_vae, rank_head_vae = train_beta_vae(train_set, vali_set, click_cache, dataset_rels_lookup)
        score_fn = make_score_fn_vae(enc_r_vae, rank_head_vae)
        scorers["beta-VAE (Ours)"] = score_fn
        ndcgs, ips_dcgs = evaluate_ranker_fn(score_fn, test_set, k=10, n_runs_ips=20)
        per_query_scores["beta-VAE (Ours)"] = {"ndcg": ndcgs, "ips_dcg": ips_dcgs}
        results["beta-VAE (Ours)"] = {"NDCG@10": float(np.mean(ndcgs)), "IPS-DCG": float(np.mean(ips_dcgs))}
        print(f"beta-VAE (Ours)               NDCG@10={np.mean(ndcgs):.4f}  IPS-DCG={np.mean(ips_dcgs):.4f}")
        save_results_incremental(results)
    except Exception as e:
        print(f"!!! beta-VAE FAILED: {e}")
        traceback.print_exc()
        results["beta-VAE (Ours)"] = {"error": str(e)}
        save_results_incremental(results)
    finally:
        free_gpu()

    # ---- paired t-tests vs Naive ----
    print("\n" + "=" * 80)
    print("TEST EVALUATION SUMMARY")
    print("=" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:28s}  FAILED: {r['error']}")
        else:
            print(f"{name:28s}  NDCG@10={r['NDCG@10']:.4f}  IPS-DCG={r['IPS-DCG']:.4f}")

    ttest_results = {}
    if "Naive Baseline" in per_query_scores:
        naive_ndcg = per_query_scores["Naive Baseline"]["ndcg"]
        naive_ipsdcg = per_query_scores["Naive Baseline"]["ips_dcg"]
        print("\nPaired t-tests vs. Naive Baseline:")
        for name in per_query_scores:
            if name == "Naive Baseline":
                continue
            n = min(len(per_query_scores[name]["ndcg"]), len(naive_ndcg))
            r_ndcg = paired_ttest_report(name, per_query_scores[name]["ndcg"][:n], naive_ndcg[:n])
            r_ips = paired_ttest_report(name, per_query_scores[name]["ips_dcg"][:n], naive_ipsdcg[:n])
            ttest_results[name] = {"NDCG@10": r_ndcg, "IPS-DCG": r_ips}
            print(f"  {name:28s}  NDCG@10: p={r_ndcg['p_value']:.4g}   IPS-DCG: p={r_ips['p_value']:.4g}")

    # ---- direct beta-VAE vs Supervised-Only comparison (R5-1) ----
    direct_comparison = None
    if "beta-VAE (Ours)" in per_query_scores and "Supervised-Only" in per_query_scores:
        print("\nDirect comparison: beta-VAE vs. Supervised-Only "
              "(R5-1 -- isolates the disentanglement mechanism's contribution "
              "beyond label supervision alone):")
        n = min(len(per_query_scores["beta-VAE (Ours)"]["ndcg"]), len(per_query_scores["Supervised-Only"]["ndcg"]))
        ndcg_diff = paired_ttest_report("beta-VAE_vs_SupervisedOnly",
                                         per_query_scores["beta-VAE (Ours)"]["ndcg"][:n],
                                         per_query_scores["Supervised-Only"]["ndcg"][:n])
        n2 = min(len(per_query_scores["beta-VAE (Ours)"]["ips_dcg"]), len(per_query_scores["Supervised-Only"]["ips_dcg"]))
        ips_diff = paired_ttest_report("beta-VAE_vs_SupervisedOnly",
                                        per_query_scores["beta-VAE (Ours)"]["ips_dcg"][:n2],
                                        per_query_scores["Supervised-Only"]["ips_dcg"][:n2])
        print(f"  NDCG@10:  mean_diff={ndcg_diff['mean_diff']:+.4f}  t={ndcg_diff['t_stat']:.2f}  p={ndcg_diff['p_value']:.4g}")
        print(f"  IPS-DCG:  mean_diff={ips_diff['mean_diff']:+.4f}  t={ips_diff['t_stat']:.2f}  p={ips_diff['p_value']:.4g}")
        direct_comparison = {"NDCG@10": ndcg_diff, "IPS-DCG": ips_diff}

    save_results_incremental(results, ttests=ttest_results,
                              extra={"direct_comparison_betaVAE_vs_SupervisedOnly": direct_comparison})
    print(f"\nFinal results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
