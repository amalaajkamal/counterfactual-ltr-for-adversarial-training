"""
================================================================================
Extended Ranking Depth: Full beta-VAE vs. Simple Position-Adversarial Ranker
================================================================================
Standalone script, separate from the main unified pipeline. Addresses a
question raised during discussion of the ablation study (Section 6.3): does
the margin between the full architecture and the simplest possible
adversarial-invariant ranker change when documents are ranked to a greater
depth than the paper's K=10, rather than only using more data per se?

SCOPE, DELIBERATELY NARROW:
  This script compares ONLY the two methods most relevant to that specific
  question: the full beta-VAE and the simple position-adversarial ranker
  (Section 6.3). It does NOT re-run the full baseline suite (IPS,
  Regression-EM, Bayes-IPS, Affine, InfoRank) at extended depth -- that
  would be a substantially larger undertaking and a different question
  (how do propensity-weighted methods behave at low propensity, deep in the
  ranking) from the one this script is built to answer.

WHY K MATTERS AND WHAT WAS CHANGED TO MAKE IT SAFE:
  The paper's trust-bias parameter eps_plus_k(k) = max(1 - (k+1)/100, 0)
  reaches EXACTLY ZERO at k >= 99, making even a truly relevant document
  structurally unclickable past that depth -- a formula implicitly tuned
  around K=10, not designed to extend past it. This script replaces the
  hard zero floor with a small positive floor (EPS_PLUS_FLOOR below), so
  relevant documents remain clickable, at reduced probability, at any
  depth. This is the ONLY change to the click model; everything else
  (theta_k, eps_minus_k, gamma) is identical to the main pipeline and
  already well-defined at any depth.

COMPUTE COST WARNING:
  The pairwise ranking loss is O(n^2) in documents per query. Yahoo!
  averages ~24 docs/query even at full depth (much cheaper than MSLR's
  ~125), which is why this script defaults to Yahoo! and to a moderate
  EXTENDED_K (50) rather than unbounded depth. Raising EXTENDED_K further,
  or switching to MSLR, will increase runtime substantially -- adjust
  deliberately, not by default.

HOW TO USE:
  1. Set YAHOO_TRAIN_FILE / YAHOO_VALI_FILE / YAHOO_TEST_FILE below.
  2. Adjust EXTENDED_K if desired (default 50; K=10 results from the main
     paper are hardcoded below as K10_REFERENCE for direct comparison).
  3. Run: python extended_depth_ablation_comparison.py
  4. Results print to console and save to extended_depth_results.json
================================================================================
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import ttest_rel

# ============================================================================
# CONFIG
# ============================================================================
YAHOO_TRAIN_FILE = r"E:\Learning to Rank Challenge\ltrc_yahoo\set2.train.txt"
YAHOO_VALI_FILE  = r"E:\Learning to Rank Challenge\ltrc_yahoo\set2.valid.txt"
YAHOO_TEST_FILE  = r"E:\Learning to Rank Challenge\ltrc_yahoo\set2.test.txt"

EXTENDED_K = 50          # ranking depth for this experiment (paper: K=10)
                          # raise deliberately -- pairwise loss is O(n^2)/query
SEEDS = [42, 123, 7]      # matches the main paper's 3-seed protocol
EPOCHS_VAE = 30           # reduced-budget, consistent with every multi-seed
EPOCHS_BASELINE_RANKER = 20   # sweep in the main pipeline
PATIENCE = 8

PRODUCTION_RANKER_TRAIN_FRAC = 0.01
PRODUCTION_RANKER_EPOCHS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Extended ranking depth: K={EXTENDED_K} (paper default: K=10)")

# K=10 reference point, from the main paper's 5-seed ablation (multiseed_ablation_yahoo.json)
K10_REFERENCE = {
    "full_model":     {"NDCG10_mean": 0.7321242197432023, "IPSDCG_mean": 8.640827642731574},
    "simple_ranker":  {"NDCG10_mean": 0.7316263027019152, "IPSDCG_mean": 8.611021807292527},
}
K10_REFERENCE["margin_IPSDCG"] = (K10_REFERENCE["full_model"]["IPSDCG_mean"]
                                   - K10_REFERENCE["simple_ranker"]["IPSDCG_mean"])
K10_REFERENCE["margin_NDCG10"] = (K10_REFERENCE["full_model"]["NDCG10_mean"]
                                    - K10_REFERENCE["simple_ranker"]["NDCG10_mean"])


# ============================================================================
# DATA LOADING (identical to the main pipeline's LTRDataset)
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
# MODEL ARCHITECTURE (identical to the main pipeline: Section 4.2-4.3)
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
    def __init__(self, in_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
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


class SimpleAdversarialEncoder(nn.Module):
    """Deterministic single encoder, no VAE, matching z_r's 24 dims -- see
    Section 6.3 (simple position-adversarial ranker) for full rationale."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 96), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(96, 64), nn.ReLU(),
            nn.Linear(64, 24),
        )

    def forward(self, x):
        return self.net(x)


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


def sinusoidal_position_encoding(k_idx, dim=8):
    """
    Deterministic sinusoidal positional encoding (Vaswani et al., 2017,
    "Attention Is All You Need"), used here INSTEAD OF a learned
    nn.Embedding lookup table for position.

    WHY THIS MATTERS AT EXTENDED_K SPECIFICALLY: a learned embedding
    table needs each position index to appear often enough during
    training for its row to be learned meaningfully. At the paper's
    K=10, most queries have at least 10 documents, so every position is
    reasonably well observed. At EXTENDED_K=50, most Yahoo! queries have
    far fewer than 50 documents (Yahoo! averages ~24 docs/query even
    using all available documents), so positions in roughly the 25-49
    range would be seen by only a small, biased subset of queries (only
    the longer-than-average ones), leaving that part of a learned
    embedding table poorly trained, close to its random initialization,
    exactly the depth range this experiment cares most about.

    Sinusoidal encoding has no such weakness: it is a fixed, closed-form
    function of the position index, identically well-defined whether that
    position was seen once, a thousand times, or never, during training,
    and nearby positions receive smoothly similar encodings by
    construction (unlike an undertrained embedding row, which is
    essentially arbitrary). This trades away the (at K=10) modest benefit
    of letting the model learn its own position representation, in
    exchange for a representation that remains meaningful at every depth
    tested here.

    k_idx: LongTensor of 0-indexed position indices, shape (n,).
    Returns: FloatTensor, shape (n, dim), dim must be even.
    """
    position = k_idx.float().unsqueeze(1)  # (n, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=k_idx.device).float()
                          * (-np.log(10000.0) / dim))
    pe = torch.zeros(k_idx.size(0), dim, device=k_idx.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SimpleRanker(nn.Module):
    """Used only for the small production ranker (exposure protocol)."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# LOSSES (identical to the main pipeline: Section 4.4)
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
    """Vectorized; O(n^2) in documents per query -- the main cost driver
    when EXTENDED_K is raised (see compute cost warning above)."""
    n = len(rels)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    rels_t = torch.as_tensor(np.asarray(rels), dtype=torch.float32, device=scores.device)
    mask = rels_t.unsqueeze(1) > rels_t.unsqueeze(0)
    if not mask.any():
        return torch.tensor(0.0, device=scores.device)
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)
    losses = F.relu(margin + score_diff)
    return losses[mask].mean()


# ============================================================================
# CLICK SIMULATION -- identical to the main pipeline EXCEPT eps_plus_k,
# which is given a small positive floor instead of hitting exactly zero,
# so it is well-defined at depths beyond the paper's K=10 (see header).
# ============================================================================
THETA_KAPPA = 0.9

def theta_k(k):
    """Ground-truth examination probability; well-defined at any k > 0,
    unchanged from the main pipeline."""
    return k ** (-THETA_KAPPA)


EPS_PLUS_FLOOR = 0.02   # small positive floor -- see header for rationale


def eps_plus_k(k):
    """
    MODIFIED from the main pipeline's max(1 - (k+1)/100, 0): the hard zero
    floor is replaced with EPS_PLUS_FLOOR, so a truly relevant document
    remains clickable (at reduced probability) at any depth, rather than
    becoming structurally unclickable past k=99. For k <= 99, where the
    original formula was already positive, this produces IDENTICAL values
    to the main pipeline; the floor only changes behavior in the region
    the original formula could not represent.
    """
    return max(1 - (k + 1) / 100.0, EPS_PLUS_FLOOR)


def eps_minus_k(k):
    """Unchanged from the main pipeline; already well-defined at any depth
    (caps at k=10, no zero/degenerate region)."""
    return 0.65 / min(k, 10)


def gamma(y):
    return 1.0 if y > 2 else 0.0


def simulate_clicks(rels, positions, k_max):
    clicks = []
    for y, k in zip(rels, positions):
        k = min(k, k_max)
        p = theta_k(k) * (eps_plus_k(k) * gamma(y) + eps_minus_k(k) * (1 - gamma(y)))
        clicks.append(1 if random.random() < p else 0)
    return clicks


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def ndcg_at_k(true_rels, pred_scores, k):
    k = min(k, len(true_rels))
    order = np.argsort(-pred_scores)[:k]
    dcg = np.sum((2 ** true_rels[order] - 1) / np.log2(np.arange(2, k + 2)))
    ideal = np.argsort(-true_rels)[:k]
    idcg = np.sum((2 ** true_rels[ideal] - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def ips_dcg_for_query(true_rels, pred_scores, k_max, n_runs=5):
    k = min(k_max, len(true_rels))
    order = np.argsort(-pred_scores)[:k]
    ranked_rels = true_rels[order]
    positions = np.arange(1, k + 1)
    vals = []
    for _ in range(n_runs):
        clicks = simulate_clicks(ranked_rels, positions, k_max)
        thetas = np.array([theta_k(min(p, k_max)) for p in positions])
        gains = (2 ** ranked_rels - 1) / np.log2(positions + 1)
        vals.append(np.sum(np.array(clicks) / thetas * gains))
    return np.mean(vals)


def evaluate_ranker_fn(score_fn, test_set, k_max, ndcg_k=10, n_runs_ips=10):
    """NDCG@10 (paper's standard ranking-quality metric, unaffected by
    k_max) alongside IPS-DCG computed at the EXTENDED depth k_max."""
    ndcgs, ips_dcgs = [], []
    for qid, idxs in test_set.queries.items():
        feats = test_set.features[idxs]
        rels = test_set.rels[idxs]
        if len(rels) < 2:
            continue
        scores = score_fn(feats)
        ndcgs.append(ndcg_at_k(rels, scores, ndcg_k))
        ips_dcgs.append(ips_dcg_for_query(rels, scores, k_max, n_runs=n_runs_ips))
    return np.array(ndcgs), np.array(ips_dcgs)


def paired_ttest_report(name, scores_a, scores_b):
    a, b = np.array(scores_a), np.array(scores_b)
    t_stat, p_val = ttest_rel(a, b)
    return {"method": name, "mean_diff": float(a.mean() - b.mean()),
            "t_stat": float(t_stat), "p_value": float(p_val)}


def compute_position_classifier_probe(extract_fn, test_set, k_max, random_state):
    """Frozen-representation position-classifier probe (Section 6.4 /
    Section 6.3), extended to k_max-way classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    Z_list, pos_list = [], []
    for qid, idxs in test_set.queries.items():
        feats = test_set.features[idxs]
        rels = test_set.rels[idxs]
        if len(rels) < 2:
            continue
        order = np.argsort(-rels)[:k_max]
        ranked_feats = feats[order]
        n = len(order)
        z = extract_fn(ranked_feats)
        Z_list.append(z)
        pos_list.append(np.arange(n))
    Z = np.concatenate(Z_list, axis=0)
    pos = np.concatenate(pos_list, axis=0).astype(int)
    n_positions = int(pos.max()) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        Z, pos, test_size=0.3, random_state=random_state, stratify=pos)
    lr = LogisticRegression(max_iter=1000, random_state=random_state).fit(X_train, y_train)
    acc_linear = float(accuracy_score(y_test, lr.predict(X_test)))
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                                 random_state=random_state).fit(X_train, y_train)
    acc_nonlinear = float(accuracy_score(y_test, rf.predict(X_test)))
    chance = 1.0 / n_positions
    return {"chance_level": chance, "accuracy_linear": acc_linear, "accuracy_nonlinear": acc_nonlinear,
            "linear_vs_chance": acc_linear - chance, "nonlinear_vs_chance": acc_nonlinear - chance}


# ============================================================================
# PRODUCTION RANKER (exposure protocol; Joachims et al. 2017)
# ============================================================================
def train_production_ranker(train_set, frac, epochs):
    print(f"\n[Production Ranker] Training on {frac:.0%} of training queries...")
    qids = list(train_set.queries.keys())
    random.shuffle(qids)
    n_use = max(1, int(len(qids) * frac))
    subset_qids = set(qids[:n_use])

    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    for ep in range(epochs):
        model.train()
        for qid in subset_qids:
            idxs = train_set.queries[qid]
            feats = train_set.features[idxs]
            rels = train_set.rels[idxs]
            if len(rels) < 2:
                continue
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            y = torch.tensor(rels, dtype=torch.float32, device=DEVICE)
            score = model(x)
            loss = F.mse_loss(score, y) + 0.5 * pairwise_hinge_loss(score, rels)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def precompute_training_clicks(train_set, production_ranker, k_max):
    cache = {}
    for qid, idxs in train_set.queries.items():
        feats = train_set.features[idxs]
        rels = train_set.rels[idxs]
        if len(rels) < 2:
            continue
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            exposure_scores = production_ranker(x).cpu().numpy()
        order = np.argsort(-exposure_scores)[:k_max]
        ranked_feats = feats[order]
        ranked_rels = rels[order]
        positions = np.arange(1, len(ranked_rels) + 1)
        clicks = simulate_clicks(ranked_rels, positions, k_max)
        cache[qid] = (ranked_feats, np.array(clicks, dtype=np.float32), ranked_rels)
    return cache


def quick_val_ndcg(score_fn, vali_set, ndcg_k=10):
    vals = []
    for qid, idxs in vali_set.queries.items():
        feats = vali_set.features[idxs]
        rels = vali_set.rels[idxs]
        if len(rels) < 2:
            continue
        scores = score_fn(feats)
        vals.append(ndcg_at_k(rels, scores, ndcg_k))
    return np.mean(vals) if vals else 0.0


# ============================================================================
# TRAINING: FULL BETA-VAE (Section 4.4-4.6, standard schedule)
# ============================================================================
def train_beta_vae(train_set, vali_set, click_cache, epochs, k_max):
    print("\n[Full beta-VAE] Training at extended depth...")
    in_dim = train_set.feature_dim
    enc_r = RelevanceEncoder(in_dim).to(DEVICE)
    enc_b = BiasEncoder(in_dim).to(DEVICE)
    dec = Decoder().to(DEVICE)
    rank_head = RankingHead().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    # Position representation is sinusoidal_position_encoding() (deterministic,
    # no learned parameters) rather than a learned nn.Embedding -- see that
    # function's docstring for why this matters specifically at EXTENDED_K.

    gen_params = (list(enc_r.parameters()) + list(enc_b.parameters()) +
                  list(dec.parameters()) + list(rank_head.parameters()))
    opt_g = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    lambda_rank, lambda_cov = 2.0, 0.001
    lambda_r_max, lambda_b_max = 0.6, 0.3
    free_bits = 0.5

    def score_fn(feats_np):
        enc_r.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            _, mu_r, _ = enc_r(x)
            return rank_head(mu_r).cpu().numpy()

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        t = ep
        if t < epochs * 0.25:
            lambda_r, lambda_b = 0.0, 0.0
        elif t < epochs * 0.75:
            frac = (t - epochs * 0.25) / (epochs * 0.5)
            lambda_r, lambda_b = lambda_r_max * frac, lambda_b_max * frac
        else:
            lambda_r, lambda_b = lambda_r_max, lambda_b_max
        lambda_adv = 0.1 + 0.9 * (t / max(epochs - 1, 1))

        enc_r.train(); enc_b.train(); dec.train(); rank_head.train(); disc.train()
        for qid, (feats, clicks, ranked_rels) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            n = len(clicks)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, k_max - 1)
            pos_vec = sinusoidal_position_encoding(k_idx, dim=8)
            y_pos = (k_idx >= (k_max // 2)).long()

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
            y_true = torch.tensor(ranked_rels, dtype=torch.float32, device=DEVICE)

            l_rec = F.binary_cross_entropy(c_hat, c)
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, ranked_rels)
            l_rank = l_point + 0.5 * l_pair
            l_kl_r = free_bits_kl(mu_r, logvar_r, free_bits)
            l_kl_b = free_bits_kl(mu_b, logvar_b, free_bits)
            l_cov = cov_penalty(zr, zb)
            zr_grl = grad_reverse(zr, lambda_adv)
            l_adv = F.cross_entropy(disc(zr_grl), y_pos)

            loss = (l_rec + lambda_rank * l_rank + lambda_r * l_kl_r +
                    lambda_b * l_kl_b + lambda_cov * l_cov + lambda_adv * l_adv)
            opt_g.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt_g.step()

        val_ndcg = quick_val_ndcg(score_fn, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {"enc_r": {k: v.clone() for k, v in enc_r.state_dict().items()},
                           "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    enc_r.load_state_dict(best_state["enc_r"])
    rank_head.load_state_dict(best_state["rank_head"])
    return enc_r, rank_head


# ============================================================================
# TRAINING: SIMPLE POSITION-ADVERSARIAL RANKER (Section 6.3)
# ============================================================================
def train_simple_adversarial_ranker(train_set, vali_set, click_cache, epochs, k_max):
    print("\n[Simple position-adversarial ranker] Training at extended depth...")
    enc = SimpleAdversarialEncoder(train_set.feature_dim).to(DEVICE)
    rank_head = RankingHead(in_dim=24).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    gen_params = list(enc.parameters()) + list(rank_head.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    lambda_rank = 2.0

    def score_fn(feats_np):
        enc.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            return rank_head(enc(x)).cpu().numpy()

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        lambda_adv = 0.1 + 0.9 * (ep / max(epochs - 1, 1))
        enc.train(); rank_head.train(); disc.train()
        for qid, (feats, clicks, ranked_rels) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            n = len(clicks)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, k_max - 1)
            y_pos = (k_idx >= (k_max // 2)).long()

            with torch.no_grad():
                z_detached = enc(x)
            for _ in range(3):
                d_loss = F.cross_entropy(disc(z_detached), y_pos)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            z = enc(x)
            score = rank_head(z)
            y_true = torch.tensor(ranked_rels, dtype=torch.float32, device=DEVICE)
            l_rank = F.mse_loss(score, y_true) + 0.5 * pairwise_hinge_loss(score, ranked_rels)
            z_grl = grad_reverse(z, lambda_adv)
            l_adv = F.cross_entropy(disc(z_grl), y_pos)
            loss = lambda_rank * l_rank + lambda_adv * l_adv
            opt_g.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt_g.step()

        val_ndcg = quick_val_ndcg(score_fn, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {"enc": {k: v.clone() for k, v in enc.state_dict().items()},
                           "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    enc.load_state_dict(best_state["enc"])
    rank_head.load_state_dict(best_state["rank_head"])
    return enc, rank_head


# ============================================================================
# MAIN
# ============================================================================
def make_score_fn_vae(enc_r, rank_head):
    def fn(feats_np):
        enc_r.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            _, mu_r, _ = enc_r(x)
            return rank_head(mu_r).cpu().numpy()
    return fn


def make_score_fn_simple(enc, rank_head):
    def fn(feats_np):
        enc.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            return rank_head(enc(x)).cpu().numpy()
    return fn


def main():
    print("Loading Yahoo! LETOR dataset...")
    train_set = LTRDataset(YAHOO_TRAIN_FILE)
    vali_set = LTRDataset(YAHOO_VALI_FILE, scaler=train_set.scaler, feature_dim=train_set.feature_dim)
    test_set = LTRDataset(YAHOO_TEST_FILE, scaler=train_set.scaler, feature_dim=train_set.feature_dim)
    print(f"  train queries={len(train_set.queries)}  vali queries={len(vali_set.queries)}  "
          f"test queries={len(test_set.queries)}  feature_dim={train_set.feature_dim}")

    results = {"full_model": {"NDCG10": [], "IPSDCG": [], "probe_linear": [], "probe_nonlinear": []},
               "simple_ranker": {"NDCG10": [], "IPSDCG": [], "probe_linear": [], "probe_nonlinear": []}}

    for seed in SEEDS:
        print(f"\n{'='*80}\nSEED {seed}  (K={EXTENDED_K})\n{'='*80}")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        production_ranker = train_production_ranker(train_set, PRODUCTION_RANKER_TRAIN_FRAC,
                                                      PRODUCTION_RANKER_EPOCHS)
        click_cache = precompute_training_clicks(train_set, production_ranker, EXTENDED_K)

        enc_r, rank_head_vae = train_beta_vae(train_set, vali_set, click_cache, EPOCHS_VAE, EXTENDED_K)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_vae(enc_r, rank_head_vae), test_set, EXTENDED_K, n_runs_ips=10)
        results["full_model"]["NDCG10"].append(float(np.mean(ndcg)))
        results["full_model"]["IPSDCG"].append(float(np.mean(ips)))

        def extract_full(feats_np, _enc=enc_r):
            _enc.eval()
            with torch.no_grad():
                x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
                _, mu_r, _ = _enc(x)
                return mu_r.cpu().numpy()
        probe = compute_position_classifier_probe(extract_full, test_set, EXTENDED_K, random_state=seed)
        results["full_model"]["probe_linear"].append(probe["accuracy_linear"])
        results["full_model"]["probe_nonlinear"].append(probe["accuracy_nonlinear"])
        print(f"  [Full beta-VAE] NDCG@10={np.mean(ndcg):.4f}  IPS-DCG(K={EXTENDED_K})={np.mean(ips):.4f}  "
              f"probe: lin={probe['accuracy_linear']:.4f} nl={probe['accuracy_nonlinear']:.4f} "
              f"(chance={probe['chance_level']:.4f})")
        del enc_r, rank_head_vae
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        enc_s, rank_head_s = train_simple_adversarial_ranker(train_set, vali_set, click_cache, EPOCHS_VAE, EXTENDED_K)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_simple(enc_s, rank_head_s), test_set, EXTENDED_K, n_runs_ips=10)
        results["simple_ranker"]["NDCG10"].append(float(np.mean(ndcg)))
        results["simple_ranker"]["IPSDCG"].append(float(np.mean(ips)))

        def extract_simple(feats_np, _enc=enc_s):
            _enc.eval()
            with torch.no_grad():
                x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
                return _enc(x).cpu().numpy()
        probe = compute_position_classifier_probe(extract_simple, test_set, EXTENDED_K, random_state=seed)
        results["simple_ranker"]["probe_linear"].append(probe["accuracy_linear"])
        results["simple_ranker"]["probe_nonlinear"].append(probe["accuracy_nonlinear"])
        print(f"  [Simple ranker]  NDCG@10={np.mean(ndcg):.4f}  IPS-DCG(K={EXTENDED_K})={np.mean(ips):.4f}  "
              f"probe: lin={probe['accuracy_linear']:.4f} nl={probe['accuracy_nonlinear']:.4f} "
              f"(chance={probe['chance_level']:.4f})")
        del enc_s, rank_head_s
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*80}\nSUMMARY (mean +/- std across {len(SEEDS)} seeds, K={EXTENDED_K})\n{'='*80}")
    summary = {"extended_K": EXTENDED_K, "seeds": SEEDS, "k10_reference": K10_REFERENCE}
    for arch in ["full_model", "simple_ranker"]:
        ndcg_m, ndcg_s = float(np.mean(results[arch]["NDCG10"])), float(np.std(results[arch]["NDCG10"], ddof=1))
        ips_m, ips_s = float(np.mean(results[arch]["IPSDCG"])), float(np.std(results[arch]["IPSDCG"], ddof=1))
        lin_m = float(np.mean(results[arch]["probe_linear"]))
        nl_m = float(np.mean(results[arch]["probe_nonlinear"]))
        summary[arch] = {"NDCG10_mean": ndcg_m, "NDCG10_std": ndcg_s, "IPSDCG_mean": ips_m, "IPSDCG_std": ips_s,
                          "probe_linear_mean": lin_m, "probe_nonlinear_mean": nl_m}
        print(f"  {arch:16s}  NDCG@10={ndcg_m:.4f}+/-{ndcg_s:.4f}  IPS-DCG(K={EXTENDED_K})={ips_m:.4f}+/-{ips_s:.4f}  "
              f"probe lin={lin_m:.4f} nl={nl_m:.4f}")

    margin_ips_extended = summary["full_model"]["IPSDCG_mean"] - summary["simple_ranker"]["IPSDCG_mean"]
    margin_ndcg_extended = summary["full_model"]["NDCG10_mean"] - summary["simple_ranker"]["NDCG10_mean"]
    summary["margin_IPSDCG_extended"] = margin_ips_extended
    summary["margin_NDCG10_extended"] = margin_ndcg_extended

    print(f"\n  Full-model margin over simple ranker at K={EXTENDED_K}:  "
          f"IPS-DCG {margin_ips_extended:+.4f}   NDCG@10 {margin_ndcg_extended:+.4f}")
    print(f"  Full-model margin over simple ranker at K=10 (reference): "
          f"IPS-DCG {K10_REFERENCE['margin_IPSDCG']:+.4f}   NDCG@10 {K10_REFERENCE['margin_NDCG10']:+.4f}")
    print(f"\n  If the extended-K margin is clearly larger than the K=10 reference margin,")
    print(f"  that is evidence the full architecture's advantage grows with ranking depth.")
    print(f"  If comparable or smaller, that does not support depth as the explanation.")

    with open("extended_depth_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to extended_depth_results.json")


if __name__ == "__main__":
    main()
