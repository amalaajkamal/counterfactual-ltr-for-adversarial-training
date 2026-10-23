"""
================================================================================
beta-VAE for Yandex real clicks -- ID-embedding variant
================================================================================
Addresses Reviewer #5's R5-3 and a second reviewer's related comment: rather
than the impossible "zero-shot transfer from Yahoo!" claim (Yahoo!'s encoder
expects a 700-dim content feature vector; Yandex has none), this trains the
SAME architecture family FRESH on Yandex, using LEARNED EMBEDDINGS of
query_id / url_id / domain_id in place of content features.

--------------------------------------------------------------------------
WHY NO NDCG / IPS-DCG HERE
--------------------------------------------------------------------------
Yandex's relevance_grade (0/1/2) is DERIVED FROM the same clicks we are
trying to debias -- unlike Yahoo!/MSLR's independent editorial judgments.
Using relevance_grade to compute NDCG/IPS-DCG on a RE-RANKED list would be
circular: it would score a new ranking against labels generated under the
ORIGINAL (position-confounded) ranking. Evaluation here is instead:
  1. Held-out click prediction (AUC, log-loss) on entirely unseen SESSIONS
     -- does the reconstructed click model generalize?
  2. PB-rho: Spearman correlation between z_r's ranking score and the
     ORIGINAL logged position, on held-out data -- does z_r still carry
     the position signal it shouldn't, on genuinely real clicks?
Both avoid using relevance_grade as ground truth for a ranking-quality
metric; they test what can honestly be tested on click-only real data.

--------------------------------------------------------------------------
INPUT REPRESENTATION
--------------------------------------------------------------------------
query_id, url_id, domain_id are each passed through a HASHED embedding
table (id % n_buckets -> embedding row). This is the standard "hashing
trick" for high-cardinality categorical features and requires no fixed
vocabulary -- it handles IDs never seen during training gracefully (via
hash collisions), which matters here: this subsample already has 515K+
unique url_ids from only 50K sessions, and the full ~16.5GB file has vastly
more. Position (0-9, after truncating any SERP's 11th slot for consistency
with Yahoo!/MSLR's K_MAX_POSITION=10) uses a small ordinary embedding, same
as elsewhere in this project.

--------------------------------------------------------------------------
SESSION-LEVEL TRAIN/TEST SPLIT
--------------------------------------------------------------------------
Split by SESSION_ID, not by row: all documents/clicks from a given session
go entirely into train or entirely into test. Splitting by row would leak
information (the same query/URL pair can appear in both splits otherwise),
undermining the held-out evaluation's validity.
================================================================================
"""

import csv
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# ============================================================================
# CONFIG
# ============================================================================
INPUT_CSV = "yandex_subsample.csv"

SEED = 42
K_MAX_POSITION = 10          # truncate any 11th slot, matching Yahoo!/MSLR convention
EMBED_DIM = 32                # per-field embedding dimension (query/url/domain)
QUERY_HASH_BUCKETS = 50_000
URL_HASH_BUCKETS = 200_000
DOMAIN_HASH_BUCKETS = 20_000
POS_EMBED_DIM = 8

EPOCHS = 40  # extended from the earlier reduced-budget run (15 epochs), now that
             # a GPU machine is available -- matches the paper-spec full epoch count.
MAX_SESSIONS_TO_USE = 10000   # extended from the earlier reduced-budget run (4,000
                               # sessions), applied BEFORE the train/test split (so both
                               # keep the same proportions). Still a subsample of the
                               # full session pool, not the complete dataset.
                               # Set to None to use all sessions in the CSV.
BATCH_SIZE_QUERIES = 256      # number of (session,serp) groups per optimizer step
TEST_SESSION_FRAC = 0.2
PROGRESS_EVERY_N_SERPS = 2000  # print within-epoch progress this often --
                                 # critical at this dataset's scale (~74K
                                 # SERPs/epoch): without this, NOTHING prints
                                 # until a full epoch finishes, which can look
                                 # identical to a hung process for a long time.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")
print(f"*** ACTIVE SEED FOR THIS RUN: {SEED} ***")


# ============================================================================
# PART 1 -- DATA LOADING
# ============================================================================
def load_yandex_csv(filepath):
    """
    Groups rows by (session_id, serp_id) -- one training/eval unit per SERP
    shown, exactly analogous to one "query" in the Yahoo!/MSLR pipelines.
    Truncates any SERP with more than K_MAX_POSITION documents.
    """
    groups = defaultdict(list)
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["session_id"], row["serp_id"])
            groups[key].append(row)

    serps = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: int(r["position"]))
        rows = [r for r in rows if int(r["position"]) < K_MAX_POSITION]
        if len(rows) < 2:
            continue
        session_id = rows[0]["session_id"]
        query_id = int(rows[0]["query_id"])
        positions = np.array([int(r["position"]) for r in rows])
        url_ids = np.array([int(r["url_id"]) for r in rows])
        domain_ids = np.array([int(r["domain_id"]) for r in rows])
        clicks = np.array([int(r["clicked"]) for r in rows], dtype=np.float32)
        grades = np.array([int(r["relevance_grade"]) for r in rows], dtype=np.float32)
        serps.append({
            "session_id": session_id, "query_id": query_id,
            "positions": positions, "url_ids": url_ids, "domain_ids": domain_ids,
            "clicks": clicks, "grades": grades,
        })
    return serps


def split_by_session(serps, test_frac=TEST_SESSION_FRAC, seed=SEED):
    sessions = sorted(set(s["session_id"] for s in serps))
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_test = int(len(sessions) * test_frac)
    test_sessions = set(sessions[:n_test])
    train = [s for s in serps if s["session_id"] not in test_sessions]
    test = [s for s in serps if s["session_id"] in test_sessions]
    return train, test


# ============================================================================
# PART 2 -- MODEL ARCHITECTURE
# ============================================================================
def hash_ids(ids, n_buckets):
    """ids: numpy int array -> torch LongTensor of hashed bucket indices."""
    return torch.tensor((ids % n_buckets), dtype=torch.long, device=DEVICE)


class YandexInputEncoder(nn.Module):
    """
    Replaces the dense content-feature input x used for Yahoo!/MSLR with
    learned, hashed embeddings of query_id, url_id, domain_id, concatenated
    into a single vector. This vector plays the same role x played
    elsewhere -- fed into RelevanceEncoder / BiasEncoder below.
    """

    def __init__(self):
        super().__init__()
        self.query_emb = nn.Embedding(QUERY_HASH_BUCKETS, EMBED_DIM)
        self.url_emb = nn.Embedding(URL_HASH_BUCKETS, EMBED_DIM)
        self.domain_emb = nn.Embedding(DOMAIN_HASH_BUCKETS, EMBED_DIM)
        self.out_dim = EMBED_DIM * 3

    def forward(self, query_ids, url_ids, domain_ids):
        q = self.query_emb(hash_ids(query_ids, QUERY_HASH_BUCKETS))
        u = self.url_emb(hash_ids(url_ids, URL_HASH_BUCKETS))
        d = self.domain_emb(hash_ids(domain_ids, DOMAIN_HASH_BUCKETS))
        return torch.cat([q, u, d], dim=1)


class RelevanceEncoder(nn.Module):
    """Same architecture family as the Yahoo!/MSLR pipeline (Section 4.2.1),
    with in_dim now the embedding-concat dimension instead of a content
    feature dimension."""

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
    def __init__(self, in_dim, pos_dim=POS_EMBED_DIM):
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
    def __init__(self, pos_dim=POS_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24 + 16 + pos_dim, 128), nn.ReLU(),
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


# ============================================================================
# PART 3 -- LOSSES (same formulation as the Yahoo!/MSLR pipeline)
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
# PART 4 -- TRAINING
# ============================================================================
def train_yandex_beta_vae(train_serps, epochs=EPOCHS,
                           lambda_rank=2.0, lambda_cov=0.001, lambda_adv_max=1.0):
    print("\n[Yandex beta-VAE (ID-embedding variant)] Training...")
    input_enc = YandexInputEncoder().to(DEVICE)
    enc_r = RelevanceEncoder(input_enc.out_dim).to(DEVICE)
    enc_b = BiasEncoder(input_enc.out_dim).to(DEVICE)
    dec = Decoder().to(DEVICE)
    rank_head = RankingHead().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    pos_emb_layer = nn.Embedding(K_MAX_POSITION, POS_EMBED_DIM).to(DEVICE)

    gen_params = (list(input_enc.parameters()) + list(enc_r.parameters()) +
                  list(enc_b.parameters()) + list(dec.parameters()) +
                  list(rank_head.parameters()) + list(pos_emb_layer.parameters()))
    opt_g = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    lambda_r_max, lambda_b_max = 0.6, 0.3
    free_bits = 0.5

    for ep in range(epochs):
        t = ep
        if t < 10:
            lambda_r, lambda_b = 0.0, 0.0
        elif t < 30:
            frac = (t - 10) / 20
            lambda_r, lambda_b = lambda_r_max * frac, lambda_b_max * frac
        else:
            lambda_r, lambda_b = lambda_r_max, lambda_b_max
        lambda_adv = 0.1 + (lambda_adv_max - 0.1) * (t / max(epochs - 1, 1))

        random.shuffle(train_serps)
        input_enc.train(); enc_r.train(); enc_b.train(); dec.train(); rank_head.train(); disc.train()
        epoch_rec_loss, epoch_rank_loss, n_batches = 0.0, 0.0, 0
        epoch_start = time.time()
        n_total = len(train_serps)

        for serp_idx, serp in enumerate(train_serps):
            n = len(serp["positions"])
            query_ids = np.full(n, serp["query_id"])
            x = input_enc(query_ids, serp["url_ids"], serp["domain_ids"])
            c = torch.tensor(serp["clicks"], dtype=torch.float32, device=DEVICE)
            grades = serp["grades"]

            k_idx = torch.tensor(serp["positions"], dtype=torch.long, device=DEVICE)
            pos_vec = pos_emb_layer(k_idx)
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

            y_true = torch.tensor(grades, dtype=torch.float32, device=DEVICE)

            l_rec = F.binary_cross_entropy(c_hat, c)
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, grades)
            l_rank = l_point + 0.5 * l_pair
            l_kl_r = free_bits_kl(mu_r, logvar_r, free_bits)
            l_kl_b = free_bits_kl(mu_b, logvar_b, free_bits)
            l_cov = cov_penalty(zr, zb)

            zr_grl = grad_reverse(zr, lambda_adv)
            d_logits_g = disc(zr_grl)
            l_adv = F.cross_entropy(d_logits_g, y_pos)

            loss = (l_rec + lambda_rank * l_rank + lambda_r * l_kl_r +
                    lambda_b * l_kl_b + lambda_cov * l_cov + lambda_adv * l_adv)

            opt_g.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt_g.step()

            if (serp_idx + 1) % PROGRESS_EVERY_N_SERPS == 0:
                elapsed = time.time() - epoch_start
                rate = (serp_idx + 1) / elapsed
                remaining_this_epoch = (n_total - serp_idx - 1) / rate
                print(f"    epoch {ep+1}/{epochs}: {serp_idx+1}/{n_total} SERPs "
                      f"({rate:.1f} SERPs/sec, ~{remaining_this_epoch/60:.1f} min left "
                      f"this epoch, ~{(remaining_this_epoch + (epochs-1-ep)*n_total/rate)/60:.1f} "
                      f"min left total)", flush=True)

            epoch_rec_loss += l_rec.item()
            epoch_rank_loss += l_rank.item()
            n_batches += 1

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1}/{epochs}  lambda_r={lambda_r:.3f} lambda_b={lambda_b:.3f} "
                  f"lambda_adv={lambda_adv:.3f}  rec_loss={epoch_rec_loss/max(n_batches,1):.4f}  "
                  f"rank_loss={epoch_rank_loss/max(n_batches,1):.4f}")

    return input_enc, enc_r, enc_b, dec, rank_head, pos_emb_layer


# ============================================================================
# PART 5 -- EVALUATION: held-out click prediction + PB-rho
# ============================================================================
def evaluate_yandex(test_serps, input_enc, enc_r, enc_b, dec, rank_head, pos_emb_layer):
    input_enc.eval(); enc_r.eval(); enc_b.eval(); dec.eval(); rank_head.eval()

    all_clicks, all_pred_click_probs = [], []
    all_scores, all_positions = [], []

    # per-SERP re-ranking metrics (CTR@1, CTR@5, MRR): computed only over
    # SERPs with at least one real click, since a SERP with zero clicks
    # contributes nothing to whether re-ranking moved a click upward.
    # Naive-baseline versions of the same three metrics are computed
    # alongside, using each SERP's ORIGINAL logged order (no re-ranking at
    # all) -- this makes Naive directly comparable to the proposed method
    # on Yandex without requiring any propensity/bias parameters (theta_k,
    # epsilon+/-), which are not identifiable from this dataset (Section
    # 6.5) and are therefore not used for IPS/Bayes-IPS/Affine here.
    ctr_at_1_hits, ctr_at_5_hits, reciprocal_ranks = [], [], []
    naive_ctr_at_1_hits, naive_ctr_at_5_hits, naive_reciprocal_ranks = [], [], []

    with torch.no_grad():
        for serp in test_serps:
            n = len(serp["positions"])
            query_ids = np.full(n, serp["query_id"])
            x = input_enc(query_ids, serp["url_ids"], serp["domain_ids"])
            k_idx = torch.tensor(serp["positions"], dtype=torch.long, device=DEVICE)
            pos_vec = pos_emb_layer(k_idx)

            _, mu_r, _ = enc_r(x)
            _, mu_b, _ = enc_b(x, pos_vec)
            c_hat = dec(mu_r, mu_b, pos_vec)
            score = rank_head(mu_r)

            all_clicks.extend(serp["clicks"].tolist())
            all_pred_click_probs.extend(c_hat.cpu().numpy().tolist())
            all_scores.extend(score.cpu().numpy().tolist())
            all_positions.extend(serp["positions"].tolist())

            clicks_np = serp["clicks"]
            if clicks_np.sum() > 0:
                score_np = score.cpu().numpy()
                # re-rank this SERP's documents by the model's own score,
                # descending; rerank_order[i] = original index of the
                # document now placed at new rank i
                rerank_order = np.argsort(-score_np)
                clicked_new_ranks = np.where(clicks_np[rerank_order] == 1)[0]  # 0-indexed
                first_clicked_new_rank = clicked_new_ranks.min()  # earliest clicked doc in new order

                ctr_at_1_hits.append(1.0 if first_clicked_new_rank < 1 else 0.0)
                ctr_at_5_hits.append(1.0 if first_clicked_new_rank < 5 else 0.0)
                reciprocal_ranks.append(1.0 / (first_clicked_new_rank + 1))

                # Naive: same three metrics, but using the ORIGINAL logged
                # position order instead of the model's re-ranked order --
                # sorted by position ascending (0 = topmost originally shown).
                naive_order = np.argsort(serp["positions"])
                naive_clicked_ranks = np.where(clicks_np[naive_order] == 1)[0]
                first_clicked_naive_rank = naive_clicked_ranks.min()

                naive_ctr_at_1_hits.append(1.0 if first_clicked_naive_rank < 1 else 0.0)
                naive_ctr_at_5_hits.append(1.0 if first_clicked_naive_rank < 5 else 0.0)
                naive_reciprocal_ranks.append(1.0 / (first_clicked_naive_rank + 1))

    all_clicks = np.array(all_clicks)
    all_pred_click_probs = np.clip(np.array(all_pred_click_probs), 1e-7, 1 - 1e-7)
    all_scores = np.array(all_scores)
    all_positions = np.array(all_positions)

    # held-out click prediction quality
    auc = roc_auc_score(all_clicks, all_pred_click_probs) if len(set(all_clicks)) > 1 else float("nan")
    ll = log_loss(all_clicks, all_pred_click_probs)

    # PB-rho: correlation between z_r-based ranking score and ORIGINAL
    # logged position, on held-out data -- should be close to 0 if z_r has
    # successfully excluded position information, even on real clicks.
    rho, rho_p = spearmanr(all_scores, all_positions)

    ctr_at_1 = float(np.mean(ctr_at_1_hits)) if ctr_at_1_hits else float("nan")
    ctr_at_5 = float(np.mean(ctr_at_5_hits)) if ctr_at_5_hits else float("nan")
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    n_serps_with_clicks = len(reciprocal_ranks)

    naive_ctr_at_1 = float(np.mean(naive_ctr_at_1_hits)) if naive_ctr_at_1_hits else float("nan")
    naive_ctr_at_5 = float(np.mean(naive_ctr_at_5_hits)) if naive_ctr_at_5_hits else float("nan")
    naive_mrr = float(np.mean(naive_reciprocal_ranks)) if naive_reciprocal_ranks else float("nan")

    print(f"\nHeld-out click prediction: AUC={auc:.4f}  log-loss={ll:.4f}  (n={len(all_clicks)})")
    print(f"PB-rho (z_r score vs. original position): rho={rho:.4f}  p={rho_p:.4g}")
    print(f"Re-ranking metrics (n={n_serps_with_clicks} SERPs with >=1 click):")
    print(f"  Naive (original logged order):  CTR@1={naive_ctr_at_1:.4f}  "
          f"CTR@5={naive_ctr_at_5:.4f}  MRR={naive_mrr:.4f}")
    print(f"  Proposed (re-ranked by z_r):     CTR@1={ctr_at_1:.4f}  "
          f"CTR@5={ctr_at_5:.4f}  MRR={mrr:.4f}")
    return {"auc": float(auc), "log_loss": float(ll), "pb_rho": float(rho), "pb_rho_p": float(rho_p),
            "ctr_at_1": ctr_at_1, "ctr_at_5": ctr_at_5, "mrr": mrr,
            "naive_ctr_at_1": naive_ctr_at_1, "naive_ctr_at_5": naive_ctr_at_5, "naive_mrr": naive_mrr,
            "n_serps_with_clicks": n_serps_with_clicks, "n_eval_rows": len(all_clicks)}


def compute_position_classifier_probe_yandex(test_serps, input_enc, enc_r,
                                                random_state=SEED):
    """
    Bias/disentanglement check, mirroring the frozen-z_r position-classifier
    probe used for Yahoo!/MSLR (Section 6.4): extracts frozen mu_r for every
    document in the held-out test set, together with its TRUE logged
    position (0..K_MAX_POSITION-1), then trains a FRESH classifier
    (Logistic Regression and Random Forest) on a train/test split of this
    extracted data to predict position from mu_r alone.

    The desired result is LOW position-prediction accuracy from z_r (close
    to chance = 1/K_MAX_POSITION), indicating z_r has NOT retained the
    position information it shouldn't, while the ranking-quality metrics
    from evaluate_yandex (MRR, CTR@k) remain good, i.e. disentanglement is
    achieved without sacrificing ranking usefulness. This is a genuinely
    independent check: the classifier here is newly trained from scratch,
    separate from the adversarial discriminator used during training, so it
    does not simply re-report what the discriminator was already optimized
    to achieve.
    """
    input_enc.eval(); enc_r.eval()

    all_zr, all_pos = [], []
    with torch.no_grad():
        for serp in test_serps:
            n = len(serp["positions"])
            query_ids = np.full(n, serp["query_id"])
            x = input_enc(query_ids, serp["url_ids"], serp["domain_ids"])
            _, mu_r, _ = enc_r(x)
            all_zr.append(mu_r.cpu().numpy())
            all_pos.extend(serp["positions"].tolist())

    Zr = np.concatenate(all_zr, axis=0)
    pos = np.array(all_pos)
    n_positions = int(pos.max()) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        Zr, pos, test_size=0.3, random_state=random_state, stratify=pos)

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)
    acc_linear = float(accuracy_score(y_test, lr.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                                 random_state=random_state)
    rf.fit(X_train, y_train)
    acc_nonlinear = float(accuracy_score(y_test, rf.predict(X_test)))

    chance = 1.0 / n_positions
    print(f"\n[Position-Classifier Probe on frozen z_r, Yandex]")
    print(f"  Chance level ({n_positions}-way classification): {chance:.4f} ({chance:.2%})")
    print(f"  Logistic Regression accuracy: {acc_linear:.4f} ({acc_linear:.2%})  "
          f"[{acc_linear - chance:+.2%} vs. chance]")
    print(f"  Random Forest accuracy:       {acc_nonlinear:.4f} ({acc_nonlinear:.2%})  "
          f"[{acc_nonlinear - chance:+.2%} vs. chance]")
    print("  Lower accuracy (closer to chance) indicates better disentanglement: "
          "z_r retains little position information usable by an independently-trained classifier.")
    return {"chance_level": chance, "accuracy_linear": acc_linear,
            "accuracy_nonlinear": acc_nonlinear,
            "linear_vs_chance": acc_linear - chance,
            "nonlinear_vs_chance": acc_nonlinear - chance,
            "n_positions": n_positions}


def extract_qualitative_examples(test_serps, input_enc, enc_r, enc_b, dec, rank_head,
                                   pos_emb_layer, n_helped=3, n_hurt=3, random_state=SEED):
    """
    Addresses a reviewer request for qualitative examples and failure-case
    analysis, not just aggregate metrics: for every test SERP with at least
    one real click, computes where that click lands under (a) the original
    logged order and (b) the model's re-ranked order, and selects a small,
    representative set of "helped" cases (click moved to an earlier rank)
    and "hurt" cases (click moved to a later rank) for manual inspection.

    This does not identify specific documents by content (Yandex's
    query_id/url_id/domain_id are anonymized integer IDs with no available
    text), but does show the STRUCTURE of each case: how many positions the
    click moved, and in which direction, which is what a failure-case
    analysis of a re-ranking method needs to show even without interpretable
    document text.
    """
    input_enc.eval(); enc_r.eval(); enc_b.eval(); dec.eval(); rank_head.eval()

    cases = []
    with torch.no_grad():
        for serp in test_serps:
            clicks_np = serp["clicks"]
            if clicks_np.sum() == 0:
                continue
            n = len(serp["positions"])
            query_ids = np.full(n, serp["query_id"])
            x = input_enc(query_ids, serp["url_ids"], serp["domain_ids"])
            k_idx = torch.tensor(serp["positions"], dtype=torch.long, device=DEVICE)
            pos_vec = pos_emb_layer(k_idx)
            _, mu_r, _ = enc_r(x)
            score = rank_head(mu_r).cpu().numpy()

            naive_order = np.argsort(serp["positions"])
            naive_rank = np.where(clicks_np[naive_order] == 1)[0].min()

            rerank_order = np.argsort(-score)
            new_rank = np.where(clicks_np[rerank_order] == 1)[0].min()

            cases.append({
                "session_id": serp["session_id"], "query_id": int(serp["query_id"]),
                "n_documents": n, "original_click_rank": int(naive_rank),
                "reranked_click_rank": int(new_rank),
                "rank_change": int(naive_rank) - int(new_rank),  # positive = helped (moved earlier)
            })

    cases.sort(key=lambda c: c["rank_change"])
    hurt_cases = [c for c in cases if c["rank_change"] < 0][:n_hurt]
    helped_cases = sorted([c for c in cases if c["rank_change"] > 0],
                           key=lambda c: -c["rank_change"])[:n_helped]

    n_helped_total = sum(1 for c in cases if c["rank_change"] > 0)
    n_hurt_total = sum(1 for c in cases if c["rank_change"] < 0)
    n_unchanged_total = sum(1 for c in cases if c["rank_change"] == 0)

    print(f"\n[Qualitative Examples, Yandex]")
    print(f"  Across {len(cases)} SERPs with >=1 click: "
          f"{n_helped_total} helped (click moved earlier), "
          f"{n_hurt_total} hurt (click moved later), "
          f"{n_unchanged_total} unchanged.")
    print(f"\n  Representative HELPED cases (click moved to an earlier rank):")
    for c in helped_cases:
        print(f"    session={c['session_id']} query_id={c['query_id']} "
              f"n_docs={c['n_documents']}: original click rank={c['original_click_rank']} "
              f"-> re-ranked click rank={c['reranked_click_rank']} "
              f"(moved {c['rank_change']} positions earlier)")
    print(f"\n  Representative HURT cases (click moved to a later rank):")
    for c in hurt_cases:
        print(f"    session={c['session_id']} query_id={c['query_id']} "
              f"n_docs={c['n_documents']}: original click rank={c['original_click_rank']} "
              f"-> re-ranked click rank={c['reranked_click_rank']} "
              f"(moved {-c['rank_change']} positions later)")

    return {"n_helped": n_helped_total, "n_hurt": n_hurt_total, "n_unchanged": n_unchanged_total,
            "helped_examples": helped_cases, "hurt_examples": hurt_cases}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"Loading {INPUT_CSV}...")
    serps = load_yandex_csv(INPUT_CSV)
    print(f"  {len(serps)} SERPs loaded (after truncating to top-{K_MAX_POSITION} positions, "
          f"dropping SERPs with <2 documents)")

    if MAX_SESSIONS_TO_USE is not None:
        all_sessions = sorted(set(s["session_id"] for s in serps))
        if len(all_sessions) > MAX_SESSIONS_TO_USE:
            rng = random.Random(SEED)
            rng.shuffle(all_sessions)
            keep_sessions = set(all_sessions[:MAX_SESSIONS_TO_USE])
            serps = [s for s in serps if s["session_id"] in keep_sessions]
            print(f"  Subsampled to {MAX_SESSIONS_TO_USE} sessions ({len(serps)} SERPs) "
                  f"(MAX_SESSIONS_TO_USE)")

    train_serps, test_serps = split_by_session(serps)
    print(f"  train SERPs={len(train_serps)}  test SERPs={len(test_serps)} (split by session)")

    input_enc, enc_r, enc_b, dec, rank_head, pos_emb_layer = train_yandex_beta_vae(train_serps)

    results = evaluate_yandex(test_serps, input_enc, enc_r, enc_b, dec, rank_head, pos_emb_layer)
    probe_results = compute_position_classifier_probe_yandex(test_serps, input_enc, enc_r)
    results["position_classifier_probe"] = probe_results
    qualitative_results = extract_qualitative_examples(test_serps, input_enc, enc_r, enc_b,
                                                          dec, rank_head, pos_emb_layer)
    results["qualitative_examples"] = qualitative_results

    out_filename = f"results_yandex_seed{SEED}.json"
    with open(out_filename, "w") as f:
        json.dump({"dataset": "yandex", "seed": SEED, "n_sessions_used": MAX_SESSIONS_TO_USE,
                   "results": results}, f, indent=2)
    print(f"\nSaved results to {out_filename}")


if __name__ == "__main__":
    main()
