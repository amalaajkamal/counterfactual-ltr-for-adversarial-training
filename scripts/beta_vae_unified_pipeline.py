"""
================================================================================
β-VAE Adversarial Disentanglement for Counterfactual LTR
Unified pipeline: Yahoo! LETOR Set 2  AND  MSLR-WEB30K
================================================================================
Full reproduction pipeline matching the architecture and loss formulation
specified in Sections 4.2–4.6 of the manuscript "Adversarial Disentanglement
for Counterfactual Learning to Rank" (ESWA-D-26-05900).

--------------------------------------------------------------------------
WHY THIS FILE EXISTS (context for future reference)
--------------------------------------------------------------------------
Earlier candidate implementations recovered from prior work used a simplified
Position-Based Model with a sigmoid attraction function for click simulation
(exam(k) * sigmoid(rel-1)), which does NOT match the affine click model with
explicit trust-bias parameters (theta_k, eps+_k, eps-_k, gamma(y)) described
in Section 4.6.5 of the manuscript. Since trust bias is central to the
paper's motivation and its differentiation from plain IPS, this script
implements the click simulation EXACTLY as specified in Section 4.6.5 (see
`simulate_clicks` below), rather than the simplified PBM variant found in
recovered notebooks. Run this on Yahoo! first (DATASET = "yahoo") to check
the resulting NDCG@10 / IPS-DCG against the published Table 1 values
(0.6968 / 13.7929) as a sanity check before trusting the MSLR run.

Trains and evaluates SIX methods for Table 1:
    1. Naive Baseline (no debiasing)
    2. IPS (Inverse Propensity Scoring)
    3. Regression-EM
    4. Bayes-IPS
    5. Affine Corrections
    6. β-VAE (proposed method)

Reports NDCG@10 and IPS-DCG on the test set, plus paired t-tests vs. Naive.

--------------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------------
1. Set TRAIN_FILE / VALI_FILE / TEST_FILE below to your MSLR-WEB30K Fold1
   paths (standard libsvm-style format: "rel qid:ID 1:val 2:val ...").
2. `pip install torch numpy scipy scikit-learn` (if not already installed).
3. Run: python mslr_beta_vae_full_pipeline.py
4. Results print to console and are saved to results_mslr.json

NOTE ON POSITION TRUNCATION (relevant to reviewer comment on MSLR's ~125
docs/query vs. K=10 max position in the paper): all position-dependent
components (position embeddings, discriminator labels, click simulation,
IPS/Affine/Bayes-IPS reweighting) apply only to the top-K=10 documents per
query, ranked by the CURRENT model's predicted score during training and by
TRUE RELEVANCE for the one-time click-simulation pass used for baseline
propensity tables. Documents beyond rank 10 are excluded from click
simulation and bias-related loss terms, consistent with K=10 in Section 4.2.2.
This choice is made explicit here since the manuscript does not state it.
================================================================================
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import ttest_rel

# ============================================================================
# CONFIG — EDIT THESE PATHS AND SET DATASET
# ============================================================================
DATASET = "yahoo"   # "yahoo" to sanity-check against published Table 1 numbers,
                     # then switch to "mslr" once verified.

YAHOO_TRAIN_FILE = r"C:\Users\amala\Desktop\ltrc_yahoo\set2.train.txt"
YAHOO_VALI_FILE  = r"C:\Users\amala\Desktop\ltrc_yahoo\set2.valid.txt"
YAHOO_TEST_FILE  = r"C:\Users\amala\Desktop\ltrc_yahoo\set2.test.txt"

MSLR_TRAIN_FILE = r"C:\Users\amalajkamal\Downloads\MSLR-WEB30K\Fold1\train.txt"
MSLR_VALI_FILE  = r"C:\Users\amalajkamal\Downloads\MSLR-WEB30K\Fold1\vali.txt"
MSLR_TEST_FILE  = r"C:\Users\amalajkamal\Downloads\MSLR-WEB30K\Fold1\test.txt"

if DATASET == "yahoo":
    TRAIN_FILE, VALI_FILE, TEST_FILE = YAHOO_TRAIN_FILE, YAHOO_VALI_FILE, YAHOO_TEST_FILE
elif DATASET == "mslr":
    TRAIN_FILE, VALI_FILE, TEST_FILE = MSLR_TRAIN_FILE, MSLR_VALI_FILE, MSLR_TEST_FILE
else:
    raise ValueError('DATASET must be "yahoo" or "mslr"')

RESULTS_FILE = f"results_{DATASET}.json"

RUN_SENSITIVITY_SWEEP = False   # already run and results collected (sensitivity_sweep_yahoo.json)

RUN_HYPERPARAMETER_SWEEP = False  # set True to run the hyperparameter sensitivity
                                    # sweep (lambda_rank, lambda_cov, lambda_adv_max)
                                    # after the main pipeline finishes -- adds
                                    # significant runtime (11 settings x reduced-
                                    # budget beta-VAE training); leave False for
                                    # normal runs. Independent of RUN_SENSITIVITY_SWEEP
                                    # -- run one at a time to avoid GPU contention.

RUN_FACTORIAL_ABLATION = False   # set True to run the adv x KL x bias-strength
                                    # factorial ablation after the main pipeline
                                    # finishes -- 12 reduced-budget beta-VAE runs.
                                    # Independent of the other two sweeps -- run
                                    # one at a time to avoid GPU contention.

RUN_MULTISEED_ABLATION = False  # set True to run the 5-seed, 6-variant ablation
                                    # (Reviewer #9, Point 6) -- 30 reduced-budget
                                    # training runs total. Intended for Yahoo!
                                    # (Table 4 is Yahoo!-only), but works for
                                    # either DATASET. Large and separate from the
                                    # other three sweeps -- leave False and enable
                                    # deliberately on its own run to avoid GPU
                                    # contention with everything else.

RUN_MISMATCHED_BIAS_EVAL = False  # set True to run the mismatched train/eval
                                    # bias function check (Reviewer #9, Point 4)
                                    # after the main pipeline finishes. Trains
                                    # NOTHING new -- reuses the already-trained
                                    # Affine and beta-VAE models and re-evaluates
                                    # IPS-DCG under several deliberately
                                    # mismatched eval-time bias assumptions.
                                    # Cheap (evaluation only, no training), safe
                                    # to leave on alongside any other run.

RUN_MULTI_BIAS_FACTOR = True  # set True to run the multi-bias-factor
                                    # experiment: trains BOTH the full beta-VAE
                                    # and the simple position-adversarial ranker
                                    # (Section 6.3) on click data with a SECOND,
                                    # untargeted bias factor added, across 3
                                    # seeds -- 6 reduced-budget training runs
                                    # total. Tests whether the full architecture's
                                    # separate bias encoder provides a genuine
                                    # advantage when a bias factor is present that
                                    # neither architecture's discriminator was
                                    # trained to target. Independent of the other
                                    # sweeps -- leave False and enable deliberately
                                    # on its own run to avoid GPU contention.

SEED = 7   # change this (e.g. 123, 7, 2026) between runs to check result stability
K_MAX_POSITION = 10          # max position considered for bias modeling (paper: K=10)
EPOCHS_VAE = 80               # paper: max epochs 80, early stop patience 30
EPOCHS_BASELINE = 50           # increased from 30 -- with patience==epochs, early
                                # stopping could never actually trigger before
BASELINE_PATIENCE = 10          # separate, tighter patience than the VAE's 30,
                                # so baselines get a real chance to converge
                                # AND a real chance to stop early once they do
EARLY_STOP_PATIENCE = 30       # used only by train_beta_vae (paper spec: patience 30)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")


# ============================================================================
# PART 1 — DATA LOADING (libsvm-style: MSLR-WEB30K / Yahoo LETOR format)
# ============================================================================
class LTRDataset:
    """Loader for MSLR-WEB30K / Yahoo LETOR libsvm-style files."""

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
# PART 2 — MODEL ARCHITECTURE (exact per Section 4.2–4.3 of the manuscript)
# ============================================================================
class RelevanceEncoder(nn.Module):
    """Enc_r: R^d -> R^{2x24}. Matches Eq. (1) in Section 4.2.1."""

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
    """Enc_b: R^{d+8} -> R^{2x16}. Matches Section 4.2.1."""

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
    """Dec: R^{24+16+8} -> [0,1]. Matches Section 4.2.3."""

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
    """Rank: R^24 -> R. Matches Section 4.2.4. in_dim parametrized (default
    24, unchanged for all existing calls) to support the single-encoder
    ablation variant, whose shared 40-dim latent needs a differently-sized
    input layer here."""

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
    """D: R^24 -> R^2 (binary top-5 vs bottom-5). Matches Section 4.3.1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, zr):
        return self.net(zr)  # raw logits; softmax applied via CE loss


class PositionPredictor(nn.Module):
    """
    NOT part of the original manuscript architecture -- added to fix an
    observed disentanglement asymmetry: z_r has an explicit adversarial
    mechanism pushing it AWAY from position (Discriminator + GRL), but z_b
    had no symmetric mechanism pushing it TOWARD position -- the decoder's
    direct access to the raw position embedding e_p let it reconstruct
    clicks without needing z_b to encode position at all, so z_b drifted
    toward encoding other click-correlated (relevance-adjacent) signal
    instead. This head is trained via plain supervised cross-entropy
    (predict position bin from z_b), providing the missing forcing
    function. K_MAX_POSITION-way classification (multi-class, unlike the
    binary top-5/bottom-5 split used for the z_r discriminator, since here
    we want it to actually recover fine-grained position, not just coarse
    top/bottom split).
    """

    def __init__(self, zb_dim=16, n_positions=None):
        super().__init__()
        n_positions = n_positions or K_MAX_POSITION
        self.net = nn.Sequential(
            nn.Linear(zb_dim, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_positions),
        )

    def forward(self, zb):
        return self.net(zb)  # raw logits; softmax applied via CE loss


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


# Simple feedforward ranker used for the 5 non-VAE baselines
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
    """
    InfoRank (Jin et al., WWW 2024) -- "InfoRank: Unbiased Learning-to-Rank
    via Conditional Mutual Information Minimization". Added as a baseline
    to address the reviewer comment on missing comparison to recent
    (2023-2025) neural/causal ULTR methods.

    Faithful to the paper's core formulation (Eqs. 8-17), with one
    deliberate architectural simplification: the paper's multi-head
    attention feature-interaction module (Eqs. 8-11) is designed for
    per-field CATEGORICAL user-item features (one-hot per field), which
    our dense, already-numeric LETOR-style feature vectors are not. We
    replace it with a standard MLP trunk over the dense feature vector,
    keeping the paper's core mechanism -- the dual observation/relevance
    heads and the conditional-mutual-information regularizer (Eqs. 12-14)
    -- exactly as specified. This is analogous to how the paper itself
    reports an "InfoRank (Debiasing) + DNN" variant using a plain DNN
    trunk in place of the attention module.

    Unlike our beta-VAE, InfoRank consolidates document features and
    position into a single input x (per the paper: "we combine user-item
    features and positions into a unified x"). Since our other baselines
    (IPS, Affine, etc.) receive position only as a REWEIGHTING factor
    (not as a ranker input), and our beta-VAE's z_r never receives
    position at all, we follow the paper's stated design of estimating
    observation as a fully LATENT variable inferred from document
    features alone (Section 3.1: "our approach does not require the
    availability of raw observation data"; Section 3.2.1's Remark),
    so InfoRank's input here is document features only, matching the
    other baselines' ranking-input convention for a fair comparison.
    """

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.obs_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # relevance head takes [trunk_output; observation_scalar] -- see
        # forward(): evaluated twice per input, once with o=1, once o=0,
        # to obtain both P(R=1|O=1,X=x) and P(R=1|O=0,X=x) (Eq. 11's Remark)
        self.rel_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        h = self.trunk(x)
        p_o1 = torch.sigmoid(self.obs_head(h)).squeeze(-1)  # P(O=1|X=x)

        n = x.size(0)
        o1 = torch.ones(n, 1, device=x.device)
        o0 = torch.zeros(n, 1, device=x.device)
        p_r_o1 = torch.sigmoid(self.rel_head(torch.cat([h, o1], dim=1))).squeeze(-1)  # P(R=1|O=1,X=x)
        p_r_o0 = torch.sigmoid(self.rel_head(torch.cat([h, o0], dim=1))).squeeze(-1)  # P(R=1|O=0,X=x)
        return p_o1, p_r_o1, p_r_o0


def inforank_losses(p_o1, p_r_o1, p_r_o0, clicks, eps=1e-7):
    """
    Computes InfoRank's two loss terms (paper Eqs. 12-17):
    L (click BCE, Eq. 16) using P(C=1|X=x) = P(R=1|O=0,X=x)*P(O=1|X=x)
    (Eq. 12), and I (conditional mutual information regularizer, Eqs.
    13-15), which is minimized to enforce conditional independence
    between the relevance and observation estimations given features --
    this is InfoRank's debiasing mechanism, analogous in intent to our
    beta-VAE's adversarial+covariance-penalty mechanism, but achieved via
    an information-theoretic regularizer on a SINGLE shared-trunk network
    rather than architecturally separate encoders.
    """
    # --- L: click BCE (Eq. 16) ---
    p_c = torch.clamp(p_r_o0 * p_o1, eps, 1 - eps)  # Eq. 12
    L = F.binary_cross_entropy(p_c, clicks)

    # --- I: conditional mutual information (Eqs. 13-15) ---
    p_o0 = 1 - p_o1
    p_r0_o1 = 1 - p_r_o1
    p_r0_o0 = 1 - p_r_o0
    p_r1 = p_r_o1 * p_o1 + p_r_o0 * p_o0          # marginal P(R=1|x), Eq. 15
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
    I = I.mean()

    return L, I, p_r1  # p_r1 = marginal relevance, used for ranking at inference


# ============================================================================
# PART 3 — LOSSES (exact per Section 4.4)
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
    """
    Vectorized. The original implementation used a nested Python double
    for-loop (n^2 individual GPU calls), which is the dominant bottleneck
    for any caller with a large, untruncated per-query document count --
    notably train_supervised_only on MSLR-WEB30K (~125 docs/query on
    average, no top-K truncation, unlike the click-based methods' top-10
    exposure window). This version is numerically IDENTICAL to the
    original (same set of ordered pairs where rels[a] > rels[b], same
    hinge formula, same mean), computed as a single batched tensor
    operation instead of up to ~n^2 sequential GPU launches.
    """
    n = len(rels)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    rels_t = torch.as_tensor(np.asarray(rels), dtype=torch.float32, device=scores.device)
    mask = rels_t.unsqueeze(1) > rels_t.unsqueeze(0)   # mask[a, b] = True if rels[a] > rels[b]
    if not mask.any():
        return torch.tensor(0.0, device=scores.device)
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # score_diff[a, b] = scores[b] - scores[a]
    losses = F.relu(margin + score_diff)
    return losses[mask].mean()


# ============================================================================
# PART 4 — CLICK SIMULATION (affine model, Section 4.6.5 / 5.1)
# ============================================================================
# Global position-bias configuration, used by theta_k() below. Defaults match
# the paper exactly (theta_k = k^-0.9). Changed only by
# run_position_bias_sensitivity_sweep() (comment #5/#6), which restores the
# default afterward -- all other training/evaluation in this script uses the
# paper's exact setting unless this sweep function is explicitly called.
THETA_FORM = "power"   # "power" -> k^-kappa ; "log" -> 1/log2(k+1)
THETA_KAPPA = 0.9       # exponent used when THETA_FORM == "power"


def theta_k(k):
    """
    Ground-truth examination probability. k is 1-indexed position.
    Default (THETA_FORM='power', THETA_KAPPA=0.9) exactly matches the
    manuscript's theta_k = k^-0.9 (Section 4.6.5). Parameterized to support
    the position-bias sensitivity sweep addressing reviewer comment #5/#6.
    """
    if THETA_FORM == "power":
        return k ** (-THETA_KAPPA)
    elif THETA_FORM == "log":
        return 1.0 / np.log2(k + 1)
    else:
        raise ValueError(f"Unknown THETA_FORM: {THETA_FORM}")


# Separate, independently-configurable eval-time bias function (Reviewer #9,
# Point 4: "The click simulator used to generate training data and the
# counterfactual evaluation metric use the identical examination-probability
# function... Please test with a mismatched training and evaluation bias
# function to demonstrate genuine robustness, not pattern-matching."
# THETA_FORM/THETA_KAPPA (above) govern TRAINING click simulation throughout
# this script, unchanged. THETA_FORM_EVAL/THETA_KAPPA_EVAL govern only
# ips_dcg_for_query()'s evaluation-time click simulation and propensity
# denominator (see theta_k_eval() and ips_dcg_for_query() below). Defaulted
# to mirror THETA_FORM/THETA_KAPPA exactly, so every existing caller in this
# script (evaluate_ranker_fn() and everything built on it) is unaffected
# unless run_mismatched_bias_evaluation() below explicitly sets these to a
# different value than the training-time globals.
THETA_FORM_EVAL = "power"
THETA_KAPPA_EVAL = 0.9


def theta_k_eval(k):
    """
    Evaluation-time examination probability, independent of theta_k()
    (training-time). Identical formula, separate global state -- see the
    comment above THETA_FORM_EVAL for why this exists and how it is used.
    """
    if THETA_FORM_EVAL == "power":
        return k ** (-THETA_KAPPA_EVAL)
    elif THETA_FORM_EVAL == "log":
        return 1.0 / np.log2(k + 1)
    else:
        raise ValueError(f"Unknown THETA_FORM_EVAL: {THETA_FORM_EVAL}")


def eps_plus_k(k):
    # Clipped at 0: for k > 99 this formula would otherwise go negative,
    # which is invalid as a click probability. Only reachable now that
    # click simulation is no longer truncated to the top-10 exposure
    # window (see _precompute_training_clicks) -- deep documents (e.g.
    # MSLR's ~125 avg candidates/query) can have k > 99.
    return max(1 - (k + 1) / 100.0, 0.0)


def eps_minus_k(k):
    return 0.65 / min(k, 10)


def gamma(y):
    return 1.0 if y > 2 else 0.0


def simulate_clicks(rels, positions):
    """rels, positions: 1-indexed position per document (aligned lists)."""
    clicks = []
    for y, k in zip(rels, positions):
        k = min(k, K_MAX_POSITION)  # positions beyond K clipped to K (see note above)
        p = theta_k(k) * (eps_plus_k(k) * gamma(y) + eps_minus_k(k) * (1 - gamma(y)))
        clicks.append(1 if random.random() < p else 0)
    return clicks


# ============================================================================
# MULTI-BIAS-FACTOR EXPERIMENT (reviewer question: does the architecture's
# capacity for richer bias structure matter when a SECOND, untargeted bias
# factor is present, beyond the single position-bias signal tested
# elsewhere in this script?)
# ============================================================================
PRESENTATION_BOOST_STRENGTH = 0.5   # multiplicative click-probability boost
                                      # for "boosted" documents (see below)
PRESENTATION_BOOST_FRAC = 0.30       # fraction of documents flagged as boosted


def presentation_boost_flag(feat_vector):
    """
    Deterministic, position- and relevance-INDEPENDENT second bias factor:
    a document is "boosted" (e.g., a rich snippet, bold text, or other
    presentation effect that increases click probability independent of
    both true relevance and display position) based solely on a hash of
    its own feature vector. Since the hash has no relationship to the
    document's relevance label or its position in any ranking, this factor
    is uncorrelated with both by construction, a genuinely orthogonal
    second bias source, unlike position and trust bias in the main click
    model (Section 4.6.5), which are BOTH purely functions of position k
    and therefore not independent of each other.

    Crucially, NEITHER the full beta-VAE's discriminator nor the simple
    position-adversarial ranker's discriminator is given this flag as a
    target anywhere in this experiment: both continue to adversarially
    train z_r against POSITION ONLY, exactly as in every other experiment
    in this script. This factor is introduced purely as an UNTARGETED
    confound in the click-generation process, to test whether each
    architecture's z_r ends up differently contaminated by a bias source
    it was never explicitly trained to remove.
    """
    h = hash(feat_vector.tobytes()) % 1000
    return 1 if h < int(PRESENTATION_BOOST_FRAC * 1000) else 0


def simulate_clicks_multi_bias(rels, positions, feats):
    """
    Click simulation identical to simulate_clicks() (position + trust bias,
    Section 4.6.5), with an additional multiplicative boost for documents
    flagged by presentation_boost_flag(), a second bias factor independent
    of both position and relevance. Used only by the multi-bias-factor
    experiment; every other training and evaluation call in this script
    continues to use simulate_clicks() unchanged.
    """
    clicks = []
    for y, k, feat in zip(rels, positions, feats):
        k = min(k, K_MAX_POSITION)
        p = theta_k(k) * (eps_plus_k(k) * gamma(y) + eps_minus_k(k) * (1 - gamma(y)))
        if presentation_boost_flag(feat):
            p = min(p * (1 + PRESENTATION_BOOST_STRENGTH), 1.0)
        clicks.append(1 if random.random() < p else 0)
    return clicks


def _precompute_training_clicks_multi_bias(dataset):
    """
    Multi-bias-factor counterpart to _precompute_training_clicks(): same
    exposure-ranking protocol (production ranker if set, else true-relevance
    fallback), but calls simulate_clicks_multi_bias() instead of
    simulate_clicks(), so the presentation-boost factor is present in the
    training clicks used for this experiment specifically.
    """
    cache = {}
    for qid, idxs in dataset.queries.items():
        feats = dataset.features[idxs]
        rels = dataset.rels[idxs]
        if len(rels) < 2:
            continue
        if PRODUCTION_RANKER is not None:
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                exposure_scores = PRODUCTION_RANKER(x).cpu().numpy()
            order = np.argsort(-exposure_scores)[:K_MAX_POSITION]
        else:
            order = np.argsort(-rels)[:K_MAX_POSITION]
        ranked_feats = feats[order]
        ranked_rels = rels[order]
        positions = np.arange(1, len(ranked_rels) + 1)
        clicks = simulate_clicks_multi_bias(ranked_rels, positions, ranked_feats)
        cache[qid] = (ranked_feats, np.array(clicks, dtype=np.float32))
    return cache


def simulate_clicks_eval(rels, positions):
    """
    Evaluation-time counterpart to simulate_clicks(), using theta_k_eval()
    instead of theta_k(). Used only by ips_dcg_for_query() below (Reviewer
    #9, Point 4) -- every other caller of click simulation in this script
    (training) continues to use simulate_clicks()/theta_k() unchanged.
    """
    clicks = []
    for y, k in zip(rels, positions):
        k = min(k, K_MAX_POSITION)
        p = theta_k_eval(k) * (eps_plus_k(k) * gamma(y) + eps_minus_k(k) * (1 - gamma(y)))
        clicks.append(1 if random.random() < p else 0)
    return clicks


# ============================================================================
# PART 5 — EVALUATION METRICS
# ============================================================================
def ndcg_at_k(true_rels, pred_scores, k=10):
    k = min(k, len(true_rels))
    order = np.argsort(-pred_scores)[:k]
    dcg = np.sum((2 ** true_rels[order] - 1) / np.log2(np.arange(2, k + 2)))
    ideal = np.argsort(-true_rels)[:k]
    idcg = np.sum((2 ** true_rels[ideal] - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def ips_dcg_for_query(true_rels, pred_scores, k=10):
    """
    Section 5.3: rank by pred_scores, simulate clicks at resulting positions,
    then IPS-reweighted DCG using the ORIGINAL (submitted) manuscript's
    gain-based formula:
        IPS-DCG = sum_i (c_i/theta_i) * (2^y_pi(i) - 1) / log2(i+1)

    Uses theta_k_eval()/simulate_clicks_eval() throughout (both the click
    simulation and the propensity denominator), NOT theta_k() -- decoupled
    from training-time bias entirely. By default THETA_FORM_EVAL/
    THETA_KAPPA_EVAL mirror THETA_FORM/THETA_KAPPA exactly, so every
    existing caller is unaffected; run_mismatched_bias_evaluation() (Reviewer
    #9, Point 4) is the only caller that sets these to a genuinely different
    value than what the model was trained under.
    """
    k = min(k, len(true_rels))
    order = np.argsort(-pred_scores)[:k]
    ranked_rels = true_rels[order]
    positions = np.arange(1, k + 1)
    clicks = simulate_clicks_eval(ranked_rels, positions)
    thetas = np.array([theta_k_eval(min(p, K_MAX_POSITION)) for p in positions])
    gains = (2 ** ranked_rels - 1) / np.log2(positions + 1)
    ips_dcg = np.sum(np.array(clicks) / thetas * gains)
    return ips_dcg


def paired_ttest_report(name, scores_method, scores_naive):
    scores_method = np.array(scores_method)
    scores_naive = np.array(scores_naive)
    t_stat, p_val = ttest_rel(scores_method, scores_naive)
    mean_diff = scores_method.mean() - scores_naive.mean()
    return {"method": name, "mean_diff": float(mean_diff),
            "t_stat": float(t_stat), "p_value": float(p_val)}


def evaluate_ranker_fn(score_fn, test_set, k=10, n_runs_ips=1):
    """score_fn(features_np) -> scores_np for one query's documents."""
    ndcgs, ips_dcgs = [], []
    for qid, idxs in test_set.queries.items():
        feats = test_set.features[idxs]
        rels = test_set.rels[idxs]
        if len(rels) < 2:
            continue
        scores = score_fn(feats)
        ndcgs.append(ndcg_at_k(rels, scores, k))
        # average over a few simulated-click draws to reduce noise
        run_vals = [ips_dcg_for_query(rels, scores, k) for _ in range(n_runs_ips)]
        ips_dcgs.append(np.mean(run_vals))
    return np.array(ndcgs), np.array(ips_dcgs)


# ============================================================================
# PART 5.5 — TRAINING: SUPERVISED-ONLY (label-only, no clicks) -- new
# baseline addressing Reviewer #5's supervision-fairness concern
# ============================================================================
def train_supervised_only(train_set, vali_set, epochs=EPOCHS_BASELINE):
    """
    Trains purely on ground-truth relevance labels y (MSE + pairwise hinge,
    identical formula to the beta-VAE's own L_rank -- Section 4.4.2), with
    NO click signal, NO position information, and NO debiasing mechanism
    of any kind. Uses the FULL candidate document set per query (not the
    top-K click-simulated subset used by the click-only baselines), since
    this baseline has no click simulation step at all.

    Purpose: isolates how much of the beta-VAE's advantage over the
    click-only baselines (IPS, Regression-EM, Bayes-IPS, Affine) is
    attributable to label supervision alone, versus the disentanglement
    mechanism specifically. If beta-VAE still meaningfully outperforms
    this baseline, that is evidence the disentanglement mechanism
    contributes beyond what label supervision alone provides; if beta-VAE
    performs similarly to this baseline, the disentanglement mechanism's
    contribution on top of label supervision is limited. This baseline is
    architecturally identical to SimpleRanker (used by all other
    non-VAE baselines) for a fair comparison, isolating the training
    OBJECTIVE (labels-only vs. clicks-only vs. labels+disentanglement) as
    the only varying factor.
    """
    print("\n[Supervised-Only] Training (ground-truth labels only, no clicks)...")
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
            loss = point + 0.5 * pair  # matches beta-VAE's L_rank exactly (Section 4.4.2)
            opt.zero_grad(); loss.backward(); opt.step()

        val_ndcg = _quick_val_ndcg(model, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


# ============================================================================
# PART 6 — TRAINING: NAIVE BASELINE
# ============================================================================
def train_naive(train_set, vali_set, epochs=EPOCHS_BASELINE):
    print("\n[Naive Baseline] Training (no debiasing)...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # one-time simulated clicks per training query (top-K by true relevance)
    click_cache = _precompute_training_clicks(train_set)

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, idxs in train_set.queries.items():
            feats, clicks = click_cache.get(qid, (None, None))
            if feats is None:
                continue
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            loss = F.binary_cross_entropy(score, c)
            opt.zero_grad(); loss.backward(); opt.step()

        val_ndcg = _quick_val_ndcg(model, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


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


PRODUCTION_RANKER = None  # set by train_production_ranker(); used by _precompute_training_clicks
PRODUCTION_RANKER_TRAIN_FRAC = 0.01  # Joachims et al. (2017): 1% of full-info training data
PRODUCTION_RANKER_EPOCHS = 20


def train_production_ranker(train_set, frac=PRODUCTION_RANKER_TRAIN_FRAC, epochs=PRODUCTION_RANKER_EPOCHS):
    """
    Trains a small, deliberately weak baseline ranker on a SMALL FRACTION of
    the full training data, following the protocol of Joachims, Swaminathan,
    and Schnabel (2017), "Unbiased Learning-to-Rank with Biased Feedback"
    (WSDM), Section 7.1: they train a Ranking SVM on 1% of the full-info
    training data to obtain a "Production Ranker" S0, then use S0's
    predicted scores -- not true relevance -- to establish the presented
    ranking on which synthetic clicks are simulated.

    We replace their Ranking SVM with the same SimpleRanker architecture
    used elsewhere in this pipeline (functionally equivalent: a supervised
    pointwise scorer), trained via MSE + pairwise hinge against ground-truth
    relevance labels on ONLY `frac` of the training queries. This model is
    intentionally weak/imperfect (matching Joachims et al.'s framing of a
    realistic, imperfect production ranker) since it sees very little
    supervision -- it should NOT be confused with the Supervised-Only
    baseline (Section 6.1), which is trained on the FULL training set and
    evaluated as a first-class comparison method in its own right.

    The resulting model's predicted scores are used by
    _precompute_training_clicks (via the PRODUCTION_RANKER global) to
    establish the top-K exposure ranking for ALL methods' click simulation
    -- baselines and beta-VAE alike -- preserving the fairness principle
    that every method sees the identical simulated click data.
    """
    global PRODUCTION_RANKER
    print(f"\n[Production Ranker] Training on {frac:.0%} of training queries "
          f"(Joachims et al. 2017 protocol) to establish the presented ranking...")
    qids = list(train_set.queries.keys())
    random.shuffle(qids)
    n_use = max(1, int(len(qids) * frac))
    subset_qids = set(qids[:n_use])
    print(f"  Using {n_use}/{len(qids)} training queries")

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
            point = F.mse_loss(score, y)
            pair = pairwise_hinge_loss(score, rels)
            loss = point + 0.5 * pair
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    PRODUCTION_RANKER = model
    print("  Production Ranker trained. This model's scores will now determine "
          "the exposure ranking used for all methods' click simulation.")
    return model


def _precompute_training_clicks(dataset):
    """
    One pass of click simulation per training query. Top-K documents under
    the EXPOSURE RANKING are kept; positions beyond K are dropped.

    If PRODUCTION_RANKER has been set (via train_production_ranker(), called
    once in main() before any method trains), the exposure ranking is
    established by that model's predicted scores, following Joachims et al.
    (2017)'s protocol -- a small, imperfect baseline ranker's predictions,
    not ground-truth relevance, determine what was "presented" to the
    simulated user. If PRODUCTION_RANKER is None, this falls back to a
    true-relevance-based ordering (kept for backward compatibility and for
    any standalone use of this function outside main()).

    Cached and reused across all baseline trainers so all methods see the
    SAME simulated clicks (fair comparison) -- this holds regardless of
    which exposure-ranking strategy is active, since every trainer calls
    this same function against the same (global) PRODUCTION_RANKER state.
    """
    cache = {}
    for qid, idxs in dataset.queries.items():
        feats = dataset.features[idxs]
        rels = dataset.rels[idxs]
        if len(rels) < 2:
            continue

        if PRODUCTION_RANKER is not None:
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                exposure_scores = PRODUCTION_RANKER(x).cpu().numpy()
            order = np.argsort(-exposure_scores)[:K_MAX_POSITION]
        else:
            order = np.argsort(-rels)[:K_MAX_POSITION]

        ranked_feats = feats[order]
        ranked_rels = rels[order]
        positions = np.arange(1, len(ranked_rels) + 1)
        clicks = simulate_clicks(ranked_rels, positions)
        cache[qid] = (ranked_feats, np.array(clicks, dtype=np.float32))
    return cache


# ============================================================================
# PART 7 — TRAINING: IPS
# ============================================================================
def train_ips(train_set, vali_set, epochs=EPOCHS_BASELINE):
    print("\n[IPS] Training (ground-truth propensities theta_k = k^-0.9)...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    click_cache = _precompute_training_clicks(train_set)

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            k_idx = np.arange(1, len(clicks) + 1)
            w = torch.tensor(1.0 / theta_k(np.minimum(k_idx, K_MAX_POSITION)),
                              dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            # standard IPS: only clicked items contribute (reweighted by 1/theta_k)
            per_doc_loss = F.binary_cross_entropy(score, torch.ones_like(score), reduction="none")
            loss = (c * w * per_doc_loss).sum() / max(c.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()

        val_ndcg = _quick_val_ndcg(model, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_ips_estimated(train_set, vali_set, epochs=EPOCHS_BASELINE, random_frac=0.10):
    """
    IPS with ESTIMATED propensities (addresses reviewer comment on realistic
    propensity estimation). Propensities estimated via intervention
    harvesting: randomize document order for `random_frac` of training
    queries, use observed CTR-by-position on that randomized subset as the
    propensity estimate theta_hat_k, then run standard IPS training with
    theta_hat_k in place of ground truth.
    """
    print(f"\n[IPS-Estimated] Estimating propensities via {random_frac:.0%} intervention harvesting...")
    qids = list(train_set.queries.keys())
    random.shuffle(qids)
    n_random = int(len(qids) * random_frac)
    randomized_qids = set(qids[:n_random])

    click_counts = defaultdict(int)
    exam_counts = defaultdict(int)
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
        if exam_counts.get(k, 0) > 0:
            theta_hat[k] = max(click_counts[k] / exam_counts[k] / base_rate, 1e-3)
        else:
            theta_hat[k] = theta_k(k)  # fallback to functional form if no data
    theta_hat[K_MAX_POSITION] = min(theta_hat[K_MAX_POSITION], 1.0)
    print(f"  Estimated theta_hat: {theta_hat}")

    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    click_cache = _precompute_training_clicks(train_set)

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
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, theta_hat


# ============================================================================
# PART 8 — TRAINING: REGRESSION-EM
# ============================================================================
def train_regression_em(train_set, vali_set, epochs=EPOCHS_BASELINE, em_iterations=5, ema_alpha=0.5):
    print("\n[Regression-EM] Training (alternating propensity/relevance estimation)...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    click_cache = _precompute_training_clicks(train_set)

    theta_est = {k: theta_k(k) for k in range(1, K_MAX_POSITION + 1)}  # init

    epochs_per_em = max(1, epochs // em_iterations)
    best_ndcg, best_state, best_theta = -1, None, dict(theta_est)
    for em_iter in range(em_iterations):
        # --- E-step: train ranker with current theta_est (like IPS) ---
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

            # track best model by validation NDCG at every epoch, not just per EM iter
            val_ndcg = _quick_val_ndcg(model, vali_set)
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_theta = dict(theta_est)

        # --- M-step: re-estimate theta_k via regression on residual CTR ---
        # Damped with an EMA against the previous estimate to prevent the
        # oscillation/instability that a hard replacement can cause early on,
        # when the ranker's predicted relevance is still unreliable.
        model.eval()
        click_sum, exam_sum, rel_sum = defaultdict(float), defaultdict(float), defaultdict(float)
        with torch.no_grad():
            for qid, (feats, clicks) in click_cache.items():
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                pred_rel = torch.sigmoid(model(x)).cpu().numpy()
                for i, (c, r) in enumerate(zip(clicks, pred_rel)):
                    k = min(i + 1, K_MAX_POSITION)
                    click_sum[k] += c
                    exam_sum[k] += 1
                    rel_sum[k] += r
        for k in range(1, K_MAX_POSITION + 1):
            if exam_sum[k] > 0 and rel_sum[k] > 1e-6:
                new_est = float(np.clip(click_sum[k] / rel_sum[k], 1e-3, 1.0))
                theta_est[k] = ema_alpha * new_est + (1 - ema_alpha) * theta_est[k]
        val_ndcg = _quick_val_ndcg(model, vali_set)
        print(f"  EM iter {em_iter+1}/{em_iterations}  val NDCG@10={val_ndcg:.4f}  "
              f"(best so far={best_ndcg:.4f})  theta_est={theta_est}")

    model.load_state_dict(best_state)
    return model, best_theta


# ============================================================================
# PART 9 — TRAINING: BAYES-IPS
# ============================================================================
def train_bayes_ips(train_set, vali_set, epochs=EPOCHS_BASELINE):
    print("\n[Bayes-IPS] Training (trust-bias-aware Bayesian reweighting)...")
    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    click_cache = _precompute_training_clicks(train_set)

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
            # w_k = (1/theta_k) * eps+_k / (eps+_k + eps-_k)   (Section 2.3)
            w_vals = (1.0 / th) * (ep_pos / (ep_pos + ep_neg))
            w = torch.tensor(w_vals, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            per_doc_loss = F.binary_cross_entropy(score, torch.ones_like(score), reduction="none")
            loss = (c * w * per_doc_loss).sum() / max(c.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()

        val_ndcg = _quick_val_ndcg(model, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


# ============================================================================
# PART 10 — TRAINING: AFFINE CORRECTIONS
# ============================================================================
def train_affine(train_set, vali_set, epochs=EPOCHS_BASELINE, estimated=False, random_frac=0.10):
    """
    estimated=False -> ground-truth alpha_k, beta_k (matches Table 1 row).
    estimated=True  -> alpha_k, beta_k estimated via 10% intervention
                        harvesting (matches Table 3's "Affine (estimated)").
    """
    tag = "Estimated" if estimated else "Ground-truth"
    print(f"\n[Affine-{tag}] Training...")

    if estimated:
        qids = list(train_set.queries.keys())
        random.shuffle(qids)
        randomized_qids = set(qids[: int(len(qids) * random_frac)])
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
                exam_c[k] += 1
                click_c[k] += c
                rel_c[k] += gamma(y)
        alpha_hat, beta_hat = {}, {}
        for k in range(1, K_MAX_POSITION + 1):
            if exam_c.get(k, 0) > 5:
                mean_click = click_c[k] / exam_c[k]
                mean_rel = max(rel_c[k] / exam_c[k], 1e-3)
                # crude linear-regression-style estimate of slope/intercept
                beta_hat[k] = max(min(mean_click * (1 - mean_rel), mean_click), 0.0)
                alpha_hat[k] = max((mean_click - beta_hat[k]) / mean_rel, 1e-3)
            else:
                alpha_hat[k] = theta_k(k) * (eps_plus_k(k) - eps_minus_k(k))
                beta_hat[k] = theta_k(k) * eps_minus_k(k)
        print(f"  Estimated alpha: {alpha_hat}")
        print(f"  Estimated beta:  {beta_hat}")
        alpha_fn = lambda k: alpha_hat[min(k, K_MAX_POSITION)]
        beta_fn = lambda k: beta_hat[min(k, K_MAX_POSITION)]
    else:
        alpha_fn = lambda k: theta_k(min(k, K_MAX_POSITION)) * (
            eps_plus_k(min(k, K_MAX_POSITION)) - eps_minus_k(min(k, K_MAX_POSITION)))
        beta_fn = lambda k: theta_k(min(k, K_MAX_POSITION)) * eps_minus_k(min(k, K_MAX_POSITION))

    model = SimpleRanker(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    click_cache = _precompute_training_clicks(train_set)

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = np.array(clicks)
            k_idx = np.arange(1, len(clicks) + 1)
            alpha_vals = np.array([alpha_fn(k) for k in k_idx])
            beta_vals = np.array([beta_fn(k) for k in k_idx])
            # Affine estimator: [ (c_i - beta_k)/alpha_k ] * lambda(d|q,f), applied to ALL docs
            target = np.clip((c - beta_vals) / np.maximum(alpha_vals, 1e-6), 0.0, 1.0)
            target_t = torch.tensor(target, dtype=torch.float32, device=DEVICE)
            score = torch.sigmoid(model(x))
            loss = F.mse_loss(score, target_t)
            opt.zero_grad(); loss.backward(); opt.step()

        val_ndcg = _quick_val_ndcg(model, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


# ============================================================================
# PART 10.5 — TRAINING: INFORANK (Jin et al., WWW 2024) -- new baseline
# addressing the reviewer comment on missing recent (2023-2025) baselines
# ============================================================================
def train_inforank(train_set, vali_set, epochs=EPOCHS_BASELINE, eta=0.5):
    """
    eta: weight of the conditional-mutual-information regularizer (paper's
    Eq. 17); eta=0.5 matches the paper's main-experiment setting (Appendix
    B.3: "In the main experiment, we set eta as 0.5").
    """
    print(f"\n[InfoRank] Training (Jin et al. 2024; eta={eta})...")
    model = InfoRankNet(train_set.feature_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # paper: lr=0.001
    click_cache = _precompute_training_clicks(train_set)

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            p_o1, p_r_o1, p_r_o0 = model(x)
            L, I, _ = inforank_losses(p_o1, p_r_o1, p_r_o0, c)
            loss = L + eta * I  # paper Eq. 17
            opt.zero_grad(); loss.backward(); opt.step()

        val_ndcg = _quick_val_ndcg_inforank(model, vali_set)
        print(f"  epoch {ep+1}/{epochs}  val NDCG@10={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg, best_state, patience = val_ndcg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _quick_val_ndcg_inforank(model, vali_set):
    model.eval()
    vals = []
    with torch.no_grad():
        for qid, idxs in vali_set.queries.items():
            feats = vali_set.features[idxs]
            rels = vali_set.rels[idxs]
            if len(rels) < 2:
                continue
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            _, p_r_o1, p_r_o0 = model(x)
            p_o1 = torch.sigmoid(model.obs_head(model.trunk(x))).squeeze(-1)
            p_r1 = (p_r_o1 * p_o1 + p_r_o0 * (1 - p_o1)).cpu().numpy()  # marginal relevance, Eq. 15
            vals.append(ndcg_at_k(rels, p_r1, 10))
    return np.mean(vals) if vals else 0.0


def make_score_fn_inforank(model):
    """Score function using InfoRank's marginal relevance P(R=1|X=x) (Eq. 15) -- position-free."""
    def fn(feats_np):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            p_o1, p_r_o1, p_r_o0 = model(x)
            p_r1 = p_r_o1 * p_o1 + p_r_o0 * (1 - p_o1)
            return p_r1.cpu().numpy()
    return fn


# ============================================================================
# PART 11 — TRAINING: β-VAE (proposed method, exact per Section 4.4–4.6)
# ============================================================================
def train_beta_vae(train_set, vali_set, epochs=EPOCHS_VAE,
                    lambda_rank=2.0, lambda_cov=0.001, lambda_adv_max=1.0,
                    adv_enabled=True, kl_reg_enabled=True, pairwise_loss_enabled=True,
                    track_history=False, click_cache_override=None):
    """
    lambda_rank, lambda_cov, lambda_adv_max default to the paper's exact
    values. Overridable for the hyperparameter sensitivity sweep (reviewer
    comment on hyperparameter selection/sensitivity) --
    run_hyperparameter_sensitivity_sweep() below. lambda_adv_max is the
    ceiling of the adaptive schedule lambda_adv(t) = 0.1 + (lambda_adv_max
    - 0.1) * t/T (paper default lambda_adv_max=1.0 reproduces the exact
    formula lambda_adv(t) = 0.1 + 0.9*t/T stated in Section 4.6.2).

    adv_enabled, kl_reg_enabled: presence/absence toggles for the factorial
    ablation (reviewer question on separating adversarial training's and KL
    regularization's contributions, crossed with bias strength) --
    run_factorial_ablation_sweep() below. When adv_enabled=False, lambda_adv
    is forced to 0 for the entire run (no adversarial pressure at all, and
    no discriminator updates are needed since GRL scales gradient by 0).
    When kl_reg_enabled=False, lambda_r/lambda_b are forced to 0 throughout
    (annealing schedule skipped; equivalent to the "w/o beta-annealing" +
    "w/o KL regularization" combined case). Both default True, matching the
    paper's full model.

    pairwise_loss_enabled: toggle for the "w/o pairwise ranking loss"
    ablation cell (Table 4 / run_multiseed_ablation() below). When False,
    l_rank uses only the pointwise MSE term (l_point), dropping the
    0.5*l_pair term entirely, rather than the paper's full l_rank =
    l_point + 0.5*l_pair (Section 4.4.2). The covariance-penalty ablation
    cell ("w/o covariance penalty") needs no separate toggle here, since
    lambda_cov is already a free parameter -- passing lambda_cov=0.0
    achieves it directly.

    track_history: if True, records per-epoch mean loss components (total,
    reconstruction, KL(z_r), KL(z_b), ranking, adversarial) and validation
    NDCG@10 for regenerating Figure 5-style training-dynamics plots (see
    plot_training_dynamics() below). Adds negligible overhead; off by
    default since most callers (sweeps) don't need it.
    """
    print("\n[β-VAE (proposed)] Training...")
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

    click_cache = click_cache_override if click_cache_override is not None else _precompute_training_clicks(train_set)
    lambda_r_max, lambda_b_max = 0.6, 0.3
    lambda_bias_pred = 0.5   # NOT in original manuscript -- see PositionPredictor docstring
    free_bits = 0.5

    best_ndcg, best_state, patience = -1, None, 0
    history = {"epoch": [], "total_loss": [], "rec_loss": [], "kl_r": [], "kl_b": [],
               "rank_loss": [], "adv_loss": [], "val_ndcg": []} if track_history else None
    for ep in range(epochs):
        t = ep
        # beta-annealing schedule (Section 4.6.1)
        if not kl_reg_enabled:
            lambda_r, lambda_b = 0.0, 0.0
        elif t < 20:
            lambda_r, lambda_b = 0.0, 0.0
        elif t < 60:
            frac = (t - 20) / 40
            lambda_r, lambda_b = lambda_r_max * frac, lambda_b_max * frac
        else:
            lambda_r, lambda_b = lambda_r_max, lambda_b_max

        # adaptive adversarial weight (Section 4.6.2), ceiling = lambda_adv_max
        # (paper default lambda_adv_max=1.0 reproduces 0.1 + 0.9*t/T exactly)
        lambda_adv = 0.0 if not adv_enabled else 0.1 + (lambda_adv_max - 0.1) * (t / max(epochs - 1, 1))

        enc_r.train(); enc_b.train(); dec.train(); rank_head.train(); disc.train()
        epoch_losses = {"total": [], "rec": [], "kl_r": [], "kl_b": [], "rank": [], "adv": []}
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            n = len(clicks)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, K_MAX_POSITION - 1)
            pos_vec = pos_emb(k_idx)
            y_pos = (k_idx >= (K_MAX_POSITION // 2)).long()  # top-5 vs bottom-5

            # --- discriminator updates (3x per generator step, Algorithm 1) ---
            with torch.no_grad():
                zr_detached, _, _ = enc_r(x)
            for _ in range(3):
                d_logits = disc(zr_detached)
                d_loss = F.cross_entropy(d_logits, y_pos)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # --- generator step (joint objective, Section 4.4.6) ---
            zr, mu_r, logvar_r = enc_r(x)
            zb, mu_b, logvar_b = enc_b(x, pos_vec)
            c_hat = dec(zr, zb, pos_vec)
            score = rank_head(zr)

            # relevance labels for this query's top-K documents (cached lookup)
            y_true = torch.tensor(
                [dataset_rels_lookup[qid][i] for i in range(n)],
                dtype=torch.float32, device=DEVICE
            )

            l_rec = F.binary_cross_entropy(c_hat, c)
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, y_true.detach().cpu().numpy())
            l_rank = l_point + (0.5 * l_pair if pairwise_loss_enabled else 0.0)
            l_kl_r = free_bits_kl(mu_r, logvar_r, free_bits)
            l_kl_b = free_bits_kl(mu_b, logvar_b, free_bits)
            l_cov = cov_penalty(zr, zb)

            zr_grl = grad_reverse(zr, lambda_adv)
            d_logits_g = disc(zr_grl)
            l_adv = F.cross_entropy(d_logits_g, y_pos)

            # NOT in original manuscript -- supervised position-prediction loss
            # on z_b, added to give it a direct forcing function toward encoding
            # position (mirroring, in the opposite direction, the adversarial
            # pressure that already pushes z_r away from position). See
            # PositionPredictor docstring for the diagnosed issue this fixes.
            pos_logits_b = pos_predictor(zb)
            l_bias_pred = F.cross_entropy(pos_logits_b, k_idx.long())

            loss = (l_rec + lambda_rank * l_rank + lambda_r * l_kl_r +
                    lambda_b * l_kl_b + lambda_cov * l_cov + lambda_adv * l_adv +
                    lambda_bias_pred * l_bias_pred)

            if track_history:
                epoch_losses["total"].append(loss.item())
                epoch_losses["rec"].append(l_rec.item())
                epoch_losses["kl_r"].append(l_kl_r.item())
                epoch_losses["kl_b"].append(l_kl_b.item())
                epoch_losses["rank"].append(l_rank.item())
                epoch_losses["adv"].append(l_adv.item())

            opt_g.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt_g.step()

        # validation
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
        with torch.no_grad():
            bias_pred_acc = (pos_logits_b.argmax(dim=1) == k_idx).float().mean().item()
        print(f"  epoch {ep+1}/{epochs}  lambda_r={lambda_r:.3f} lambda_b={lambda_b:.3f} "
              f"lambda_adv={lambda_adv:.3f}  val NDCG@10={val_ndcg:.4f}  "
              f"z_b pos-pred acc (last batch)={bias_pred_acc:.3f}")

        if track_history:
            history["epoch"].append(ep + 1)
            history["total_loss"].append(float(np.mean(epoch_losses["total"])))
            history["rec_loss"].append(float(np.mean(epoch_losses["rec"])))
            history["kl_r"].append(float(np.mean(epoch_losses["kl_r"])))
            history["kl_b"].append(float(np.mean(epoch_losses["kl_b"])))
            history["rank_loss"].append(float(np.mean(epoch_losses["rank"])))
            history["adv_loss"].append(float(np.mean(epoch_losses["adv"])))
            history["val_ndcg"].append(float(val_ndcg))

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {
                "enc_r": {k: v.clone() for k, v in enc_r.state_dict().items()},
                "enc_b": {k: v.clone() for k, v in enc_b.state_dict().items()},
                "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()},
                "pos_emb": {k: v.clone() for k, v in pos_emb.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    enc_r.load_state_dict(best_state["enc_r"])
    enc_b.load_state_dict(best_state["enc_b"])
    rank_head.load_state_dict(best_state["rank_head"])
    pos_emb.load_state_dict(best_state["pos_emb"])
    return enc_r, rank_head, enc_b, pos_emb, history


# ============================================================================
# PART 11.5 — SINGLE-ENCODER ABLATION VARIANT ("no separation")
# ============================================================================
class SingleEncoder(nn.Module):
    """
    A single, undivided encoder replacing Enc_r + Enc_b: R^d -> R^{2x40}
    (40 = 24 + 16, matching the FULL model's combined z_r/z_b latent
    capacity exactly, so this ablation isolates the effect of SEPARATING
    relevance from bias, not a difference in total latent size). Structurally
    mirrors RelevanceEncoder's depth (in_dim -> 128 -> 96 -> 64), only the
    output layer width changes (40 instead of 24), for a fair architectural
    comparison. Used by train_single_encoder_ablation() below for the
    "single encoder (no separation)" cell of the ablation study (Table 4,
    Section 6.3): no adversarial discriminator, no covariance penalty
    (nothing to decorrelate with only one latent block), no position
    predictor (no z_b to push toward position) -- this is the "no
    disentanglement mechanism at all" baseline the ablation is meant to
    isolate.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 96), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(96, 64), nn.ReLU(),
        )
        self.mu = nn.Linear(64, 40)
        self.logvar = nn.Linear(64, 40)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -10, 10)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar


def train_single_encoder_ablation(train_set, vali_set, epochs=EPOCHS_VAE,
                                    lambda_rank=2.0, pairwise_loss_enabled=True):
    """
    Trains the "single encoder (no separation)" ablation variant: one
    SingleEncoder (40-dim latent, matching the full model's combined z_r+z_b
    capacity) feeding both the RankingHead (in_dim=40) and the Decoder
    (reused unchanged: its 48-dim input, 40 + 8-dim position embedding,
    exactly matches the full model's 24+16+8, so no architectural change
    needed there). No discriminator, no adversarial training, no covariance
    penalty, no position predictor -- this variant has no disentanglement
    mechanism of any kind, isolating what the full separation architecture
    contributes over an equally-sized but undivided latent.

    KL annealing schedule mirrors the full model's combined budget
    (lambda_r_max + lambda_b_max = 0.6 + 0.3 = 0.9), same timing (0 for
    epoch < 20, linear ramp 20-60, flat thereafter), applied to the single
    latent's free-bits KL term.
    """
    print("\n[Single-Encoder Ablation] Training (no relevance/bias separation)...")
    enc = SingleEncoder(train_set.feature_dim).to(DEVICE)
    dec = Decoder().to(DEVICE)
    rank_head = RankingHead(in_dim=40).to(DEVICE)
    pos_emb = nn.Embedding(K_MAX_POSITION, 8).to(DEVICE)

    gen_params = (list(enc.parameters()) + list(dec.parameters()) +
                  list(rank_head.parameters()) + list(pos_emb.parameters()))
    opt = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    click_cache = _precompute_training_clicks(train_set)
    lambda_kl_max = 0.6 + 0.3  # matches full model's combined lambda_r_max + lambda_b_max
    free_bits = 0.5

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        t = ep
        if t < 20:
            lambda_kl = 0.0
        elif t < 60:
            lambda_kl = lambda_kl_max * (t - 20) / 40
        else:
            lambda_kl = lambda_kl_max

        enc.train(); dec.train(); rank_head.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            c = torch.tensor(clicks, dtype=torch.float32, device=DEVICE)
            n = len(clicks)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, K_MAX_POSITION - 1)
            pos_vec = pos_emb(k_idx)

            z, mu, logvar = enc(x)
            c_hat = dec(z[:, :24], z[:, 24:], pos_vec)  # Decoder expects (zr_like, zb_like, pos); split is arbitrary since z is undivided
            score = rank_head(z)

            y_true = torch.tensor(
                [dataset_rels_lookup[qid][i] for i in range(n)],
                dtype=torch.float32, device=DEVICE
            )

            l_rec = F.binary_cross_entropy(c_hat, c)
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, y_true.detach().cpu().numpy())
            l_rank = l_point + (0.5 * l_pair if pairwise_loss_enabled else 0.0)
            l_kl = free_bits_kl(mu, logvar, free_bits)

            loss = l_rec + lambda_rank * l_rank + lambda_kl * l_kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt.step()

        enc.eval(); rank_head.eval()
        val_scores = []
        with torch.no_grad():
            for vqid, idxs in vali_set.queries.items():
                feats = vali_set.features[idxs]
                rels = vali_set.rels[idxs]
                if len(rels) < 2:
                    continue
                xv = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                _, mu_v, _ = enc(xv)
                sv = rank_head(mu_v).cpu().numpy()
                val_scores.append(ndcg_at_k(rels, sv, 10))
        val_ndcg = np.mean(val_scores) if val_scores else 0.0
        print(f"  epoch {ep+1}/{epochs}  lambda_kl={lambda_kl:.3f}  val NDCG@10={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {"enc": {k: v.clone() for k, v in enc.state_dict().items()},
                           "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    enc.load_state_dict(best_state["enc"])
    rank_head.load_state_dict(best_state["rank_head"])
    return enc, rank_head


def make_score_fn_single_encoder(enc, rank_head):
    def fn(feats_np):
        enc.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            _, mu, _ = enc(x)
            return rank_head(mu).cpu().numpy()
    return fn


# ============================================================================
# PART 11.6 — SIMPLE POSITION-ADVERSARIAL NEURAL RANKER (ablation baseline)
# ============================================================================
class SimpleAdversarialEncoder(nn.Module):
    """
    A single, DETERMINISTIC encoder (no VAE at all: no mu/logvar, no
    reparameterization, no sampling, no KL term), used for the "simple
    position-adversarial neural ranker" ablation cell requested by a
    reviewer: does the full beta-VAE machinery, dual probabilistic
    encoders, KL regularization, covariance penalty, reconstruction, add
    anything beyond just bolting adversarial position-invariance onto a
    plain deterministic ranker? Output dimension matches z_r's 24 dims for
    a fair, capacity-matched comparison. Structurally mirrors
    RelevanceEncoder's depth (in_dim -> 128 -> 96 -> 64 -> 24) so the only
    varying factor is the presence/absence of the probabilistic/VAE
    machinery and the rest of the disentanglement architecture, not encoder
    depth or width.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 96), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(96, 64), nn.ReLU(),
            nn.Linear(64, 24),
        )

    def forward(self, x):
        return self.net(x)  # deterministic, no distribution, no sampling


def train_simple_adversarial_ranker(train_set, vali_set, epochs=EPOCHS_VAE,
                                      lambda_rank=2.0, lambda_adv_max=1.0,
                                      pairwise_loss_enabled=True, click_cache_override=None):
    """
    Trains the "simple position-adversarial neural ranker" ablation cell:
    one deterministic encoder feeding both the RankingHead (reused
    unchanged, in_dim=24, identical to z_r's own ranking head) and a
    Discriminator (reused unchanged, identical to the main model's), with
    adversarial training via gradient reversal exactly as in the main
    model (Section 4.4.3), but with NO decoder, NO reconstruction loss, NO
    VAE/KL term of any kind, NO covariance penalty, and NO separate bias
    branch. This isolates whether the elaborate dual-encoder, variational,
    disentanglement-specific architecture contributes anything beyond what
    the single most essential mechanism, adversarial position-invariance
    on a plain ranker's own representation, already provides on its own.

    Uses the SAME adaptive adversarial weight schedule as the main model
    (Section 4.6.2) for a fair, matched comparison; no annealing schedule
    is needed here since there is no KL term to anneal.
    """
    print("\n[Simple Position-Adversarial Ranker] Training (no VAE, no dual encoder)...")
    enc = SimpleAdversarialEncoder(train_set.feature_dim).to(DEVICE)
    rank_head = RankingHead(in_dim=24).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    gen_params = list(enc.parameters()) + list(rank_head.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    click_cache = click_cache_override if click_cache_override is not None else _precompute_training_clicks(train_set)

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        t = ep
        lambda_adv = 0.1 + (lambda_adv_max - 0.1) * (t / max(epochs - 1, 1))

        enc.train(); rank_head.train(); disc.train()
        for qid, (feats, clicks) in click_cache.items():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            n = len(clicks)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, K_MAX_POSITION - 1)
            y_pos = (k_idx >= (K_MAX_POSITION // 2)).long()

            with torch.no_grad():
                z_detached = enc(x)
            for _ in range(3):
                d_logits = disc(z_detached)
                d_loss = F.cross_entropy(d_logits, y_pos)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            z = enc(x)
            score = rank_head(z)

            y_true = torch.tensor(
                [dataset_rels_lookup[qid][i] for i in range(n)],
                dtype=torch.float32, device=DEVICE
            )
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, y_true.detach().cpu().numpy())
            l_rank = l_point + (0.5 * l_pair if pairwise_loss_enabled else 0.0)

            z_grl = grad_reverse(z, lambda_adv)
            d_logits_g = disc(z_grl)
            l_adv = F.cross_entropy(d_logits_g, y_pos)

            loss = lambda_rank * l_rank + lambda_adv * l_adv
            opt_g.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt_g.step()

        enc.eval(); rank_head.eval()
        val_scores = []
        with torch.no_grad():
            for vqid, idxs in vali_set.queries.items():
                feats = vali_set.features[idxs]
                rels = vali_set.rels[idxs]
                if len(rels) < 2:
                    continue
                xv = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                zv = enc(xv)
                sv = rank_head(zv).cpu().numpy()
                val_scores.append(ndcg_at_k(rels, sv, 10))
        val_ndcg = np.mean(val_scores) if val_scores else 0.0
        print(f"  epoch {ep+1}/{epochs}  lambda_adv={lambda_adv:.3f}  val NDCG@10={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {"enc": {k: v.clone() for k, v in enc.state_dict().items()},
                           "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    enc.load_state_dict(best_state["enc"])
    rank_head.load_state_dict(best_state["rank_head"])
    return enc, rank_head


def make_score_fn_simple_adversarial(enc, rank_head):
    def fn(feats_np):
        enc.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            z = enc(x)
            return rank_head(z).cpu().numpy()
    return fn


class _MuExtractorWrapper:
    """
    Small adapter so compute_position_classifier_probe_simple() (built for
    encoders that return a single tensor, like SimpleAdversarialEncoder)
    can also be used safely with RelevanceEncoder, which returns a tuple
    (z, mu, logvar). Needed for the "reconstruction + adversarial, no
    explicit z_b" ablation cell, whose encoder IS a RelevanceEncoder.
    Without this wrapper, calling the probe directly on a RelevanceEncoder
    would assign the whole tuple where a single tensor is expected and
    fail at the first .cpu() call.
    """
    def __init__(self, enc_r):
        self.enc_r = enc_r

    def eval(self):
        self.enc_r.eval()

    def __call__(self, x):
        _, mu_r, _ = self.enc_r(x)
        return mu_r


def compute_position_classifier_probe_simple(enc, test_set, k_max=K_MAX_POSITION, random_state=SEED):
    """
    Runs the same frozen-representation position-classifier probe used for
    the main model's z_r (Section 6.4) against this simpler architecture's
    single latent, so the two architectures' position-invariance can be
    compared on equal footing, not just their ranking performance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    enc.eval()
    Z_list, pos_list = [], []
    with torch.no_grad():
        for qid, idxs in test_set.queries.items():
            feats = test_set.features[idxs]
            rels = test_set.rels[idxs]
            if len(rels) < 2:
                continue
            order = np.argsort(-rels)[:k_max]
            ranked_feats = feats[order]
            n = len(order)
            x = torch.tensor(ranked_feats, dtype=torch.float32, device=DEVICE)
            z = enc(x)
            Z_list.append(z.cpu().numpy())
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
    print(f"\n[Position-Classifier Probe on frozen z, Simple Adversarial Ranker]")
    print(f"  Chance level ({n_positions}-way): {chance:.4f}")
    print(f"  Logistic Regression accuracy: {acc_linear:.4f}  [{acc_linear-chance:+.2%} vs. chance]")
    print(f"  Random Forest accuracy:       {acc_nonlinear:.4f}  [{acc_nonlinear-chance:+.2%} vs. chance]")
    return {"chance_level": chance, "accuracy_linear": acc_linear, "accuracy_nonlinear": acc_nonlinear,
            "linear_vs_chance": acc_linear - chance, "nonlinear_vs_chance": acc_nonlinear - chance}


# ============================================================================
# PART 11.7 — RECONSTRUCTION + ADVERSARIAL, NO EXPLICIT z_b (ablation cell)
# ============================================================================
class DecoderNoZb(nn.Module):
    """
    Decoder variant for the "reconstruction + adversarial, no explicit
    z_b" ablation cell (reviewer-requested comparison #5): reconstructs
    clicks from z_r and the position representation ALONE, with no
    separate bias latent concatenated in. Input dimension is 24 + 8 = 32
    (z_r + position), vs. the full model's Decoder at 24 + 16 + 8 = 48
    (z_r + z_b + position). Architecturally mirrors the full Decoder's
    depth exactly, only the input width differs, for a fair comparison.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, zr, pos_emb):
        x = torch.cat([zr, pos_emb], dim=1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_reconstruction_adversarial_no_zb(train_set, vali_set, epochs=EPOCHS_VAE,
                                             lambda_rank=2.0, lambda_adv_max=1.0,
                                             pairwise_loss_enabled=True):
    """
    Trains the "reconstruction + adversarial model without explicit z_b"
    ablation cell (reviewer-requested comparison #5): a single
    probabilistic relevance encoder $z_r$ (identical to the full model's
    RelevanceEncoder, with its own KL/free-bits term, unlike the fully
    deterministic simple position-adversarial ranker), used for BOTH
    ranking (via the standard RankingHead) and reconstruction (via
    DecoderNoZb above, which reconstructs clicks from $z_r$ and position
    directly, with no separate bias latent). Adversarial training via
    gradient reversal is applied to $z_r$ exactly as in the full model.

    This isolates the SPECIFIC question of whether a separate, explicit
    $z_b$ is necessary, distinct from every other ablation cell in this
    script: unlike "single encoder (no separation)", this variant DOES
    have adversarial training and a probabilistic (VAE-style) $z_r$;
    unlike "simple position-adversarial ranker" (Section 6.3), this
    variant DOES have a reconstruction objective and retains the
    diagnostic capability that comes with it. No covariance penalty is
    used, since there is no second latent block to decorrelate $z_r$
    against.
    """
    print("\n[Reconstruction + Adversarial, no explicit z_b] Training...")
    enc_r = RelevanceEncoder(train_set.feature_dim).to(DEVICE)
    dec = DecoderNoZb().to(DEVICE)
    rank_head = RankingHead().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    pos_emb = nn.Embedding(K_MAX_POSITION, 8).to(DEVICE)

    gen_params = (list(enc_r.parameters()) + list(dec.parameters()) +
                  list(rank_head.parameters()) + list(pos_emb.parameters()))
    opt_g = torch.optim.Adam(gen_params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    click_cache = _precompute_training_clicks(train_set)
    lambda_r_max = 0.6  # matches the full model's own z_r-specific ceiling exactly
    free_bits = 0.5

    best_ndcg, best_state, patience = -1, None, 0
    for ep in range(epochs):
        t = ep
        if t < 20:
            lambda_r = 0.0
        elif t < 60:
            lambda_r = lambda_r_max * (t - 20) / 40
        else:
            lambda_r = lambda_r_max
        lambda_adv = 0.1 + (lambda_adv_max - 0.1) * (t / max(epochs - 1, 1))

        enc_r.train(); dec.train(); rank_head.train(); disc.train()
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
                d_loss = F.cross_entropy(disc(zr_detached), y_pos)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            zr, mu_r, logvar_r = enc_r(x)
            c_hat = dec(zr, pos_vec)
            score = rank_head(zr)

            y_true = torch.tensor(
                [dataset_rels_lookup[qid][i] for i in range(n)],
                dtype=torch.float32, device=DEVICE
            )
            l_rec = F.binary_cross_entropy(c_hat, c)
            l_point = F.mse_loss(score, y_true)
            l_pair = pairwise_hinge_loss(score, y_true.detach().cpu().numpy())
            l_rank = l_point + (0.5 * l_pair if pairwise_loss_enabled else 0.0)
            l_kl_r = free_bits_kl(mu_r, logvar_r, free_bits)

            zr_grl = grad_reverse(zr, lambda_adv)
            l_adv = F.cross_entropy(disc(zr_grl), y_pos)

            loss = l_rec + lambda_rank * l_rank + lambda_r * l_kl_r + lambda_adv * l_adv
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
        print(f"  epoch {ep+1}/{epochs}  lambda_r={lambda_r:.3f} lambda_adv={lambda_adv:.3f}  "
              f"val NDCG@10={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {"enc_r": {k: v.clone() for k, v in enc_r.state_dict().items()},
                           "rank_head": {k: v.clone() for k, v in rank_head.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    enc_r.load_state_dict(best_state["enc_r"])
    rank_head.load_state_dict(best_state["rank_head"])
    return enc_r, rank_head


def make_score_fn_no_zb(enc_r, rank_head):
    def fn(feats_np):
        enc_r.eval(); rank_head.eval()
        with torch.no_grad():
            x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            _, mu_r, _ = enc_r(x)
            return rank_head(mu_r).cpu().numpy()
    return fn


def compute_presentation_boost_probe(extract_fn, test_set, random_state=SEED):
    """
    Frozen-representation classifier probe for the presentation-boost
    factor (see presentation_boost_flag() above), the untargeted second
    bias factor introduced for the multi-bias-factor experiment. Mirrors
    compute_position_classifier_probe_simple()/the main model's frozen-z_r
    position probe (Section 6.4) exactly, but predicts the boost flag
    instead of position, and is architecture-agnostic: `extract_fn` maps a
    batch of raw features (numpy array) to that architecture's frozen
    relevance representation (mu_r for the full beta-VAE; the single
    latent for the simple adversarial ranker), so the same probe code
    applies to both.

    This is the direct test of the underlying hypothesis: since neither
    architecture's discriminator ever targets this factor, does one
    architecture's relevance representation end up more informative about
    it than the other's? Lower accuracy (closer to the 50% chance level
    for this binary factor) indicates the representation absorbed less of
    this untargeted bias source.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    Z_list, boost_list = [], []
    for qid, idxs in test_set.queries.items():
        feats = test_set.features[idxs]
        rels = test_set.rels[idxs]
        if len(rels) < 2:
            continue
        order = np.argsort(-rels)[:K_MAX_POSITION]
        ranked_feats = feats[order]
        z = extract_fn(ranked_feats)
        Z_list.append(z)
        boost_list.extend([presentation_boost_flag(f) for f in ranked_feats])
    Z = np.concatenate(Z_list, axis=0)
    boost = np.array(boost_list).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        Z, boost, test_size=0.3, random_state=random_state, stratify=boost)
    lr = LogisticRegression(max_iter=1000, random_state=random_state).fit(X_train, y_train)
    acc_linear = float(accuracy_score(y_test, lr.predict(X_test)))
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                                 random_state=random_state).fit(X_train, y_train)
    acc_nonlinear = float(accuracy_score(y_test, rf.predict(X_test)))
    chance = 0.5  # binary factor
    return {"chance_level": chance, "accuracy_linear": acc_linear, "accuracy_nonlinear": acc_nonlinear,
            "linear_vs_chance": acc_linear - chance, "nonlinear_vs_chance": acc_nonlinear - chance}


MULTI_BIAS_SEEDS = [42, 123, 7]   # 3 seeds; matches the main results' seed count


def run_multi_bias_factor_experiment(train_set, vali_set, test_set, epochs=30):
    # epochs default (30) matches SWEEP_EPOCHS_VAE's value, kept as a literal
    # here rather than referencing that global directly, since SWEEP_EPOCHS_VAE
    # is defined later in this file (Part 13) and Python evaluates default
    # argument values at function-definition time, not call time -- referencing
    # it here would raise NameError as soon as the module loads.
    """
    Addresses a reviewer-motivated question raised in discussion of the
    simple-position-adversarial-ranker comparison (Section 6.3): does the
    full architecture's capacity for richer bias structure, specifically
    its separate bias encoder z_b and reconstruction objective, matter
    when a SECOND bias factor is present in the click data that NEITHER
    architecture's discriminator is trained to target?

    Trains both the full beta-VAE and the simple position-adversarial
    ranker (Section 6.3) on click data generated with the added
    presentation-boost factor (simulate_clicks_multi_bias() /
    _precompute_training_clicks_multi_bias() above), across 3 seeds.
    Both architectures' discriminators continue to target position only,
    exactly as in every other experiment in this script; the boost flag is
    never given to either model as a supervised or adversarial target. For
    each seed and each architecture, evaluates ranking performance and,
    critically, runs compute_presentation_boost_probe() against the frozen
    relevance representation to test whether this untargeted bias factor
    was absorbed differently by the two architectures.

    Reduced training budget (SWEEP_EPOCHS_VAE), consistent with every
    other multi-seed sweep in this script.
    """
    print("\n" + "=" * 80)
    print("MULTI-BIAS-FACTOR EXPERIMENT: presentation-boost factor, untargeted "
          "by either architecture's discriminator")
    print(f"NOTE: reduced training budget ({epochs} epochs), {len(MULTI_BIAS_SEEDS)} seeds -- "
          f"tests whether an untargeted second bias factor is absorbed differently "
          f"by the full architecture's z_r vs. the simple ranker's single latent.")
    print("=" * 80)

    results = {"full_model": {"NDCG10": [], "IPSDCG": [], "boost_probe_linear": [], "boost_probe_nonlinear": []},
               "simple_ranker": {"NDCG10": [], "IPSDCG": [], "boost_probe_linear": [], "boost_probe_nonlinear": []}}

    for seed in MULTI_BIAS_SEEDS:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        click_cache = _precompute_training_clicks_multi_bias(train_set)

        print("  [Full beta-VAE, multi-bias click data]")
        enc_r, rank_head, _, _, _ = train_beta_vae(train_set, vali_set, epochs=epochs,
                                                     click_cache_override=click_cache)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_vae(enc_r, rank_head), test_set, k=10, n_runs_ips=5)
        results["full_model"]["NDCG10"].append(float(np.mean(ndcg)))
        results["full_model"]["IPSDCG"].append(float(np.mean(ips)))

        def extract_full(feats_np, _enc=enc_r):
            _enc.eval()
            with torch.no_grad():
                x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
                _, mu_r, _ = _enc(x)
                return mu_r.cpu().numpy()
        probe_full = compute_presentation_boost_probe(extract_full, test_set, random_state=seed)
        results["full_model"]["boost_probe_linear"].append(probe_full["accuracy_linear"])
        results["full_model"]["boost_probe_nonlinear"].append(probe_full["accuracy_nonlinear"])
        print(f"    NDCG@10={np.mean(ndcg):.4f}  IPS-DCG={np.mean(ips):.4f}  "
              f"boost-probe: linear={probe_full['accuracy_linear']:.4f} "
              f"nonlinear={probe_full['accuracy_nonlinear']:.4f}  (chance=0.50)")
        del enc_r, rank_head
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print("  [Simple position-adversarial ranker, multi-bias click data]")
        enc_s, rank_head_s = train_simple_adversarial_ranker(train_set, vali_set, epochs=epochs,
                                                                click_cache_override=click_cache)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_simple_adversarial(enc_s, rank_head_s), test_set, k=10, n_runs_ips=5)
        results["simple_ranker"]["NDCG10"].append(float(np.mean(ndcg)))
        results["simple_ranker"]["IPSDCG"].append(float(np.mean(ips)))

        def extract_simple(feats_np, _enc=enc_s):
            _enc.eval()
            with torch.no_grad():
                x = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
                return _enc(x).cpu().numpy()
        probe_simple = compute_presentation_boost_probe(extract_simple, test_set, random_state=seed)
        results["simple_ranker"]["boost_probe_linear"].append(probe_simple["accuracy_linear"])
        results["simple_ranker"]["boost_probe_nonlinear"].append(probe_simple["accuracy_nonlinear"])
        print(f"    NDCG@10={np.mean(ndcg):.4f}  IPS-DCG={np.mean(ips):.4f}  "
              f"boost-probe: linear={probe_simple['accuracy_linear']:.4f} "
              f"nonlinear={probe_simple['accuracy_nonlinear']:.4f}  (chance=0.50)")
        del enc_s, rank_head_s
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "-" * 80)
    print(f"MULTI-BIAS-FACTOR EXPERIMENT SUMMARY (mean +/- std across {len(MULTI_BIAS_SEEDS)} seeds)")
    print("-" * 80)
    summary = {}
    for arch in ["full_model", "simple_ranker"]:
        ndcg_m, ndcg_s = float(np.mean(results[arch]["NDCG10"])), float(np.std(results[arch]["NDCG10"], ddof=1))
        ips_m, ips_s = float(np.mean(results[arch]["IPSDCG"])), float(np.std(results[arch]["IPSDCG"], ddof=1))
        lin_m, lin_s = float(np.mean(results[arch]["boost_probe_linear"])), float(np.std(results[arch]["boost_probe_linear"], ddof=1))
        nl_m, nl_s = float(np.mean(results[arch]["boost_probe_nonlinear"])), float(np.std(results[arch]["boost_probe_nonlinear"], ddof=1))
        summary[arch] = {"NDCG10_mean": ndcg_m, "NDCG10_std": ndcg_s, "IPSDCG_mean": ips_m, "IPSDCG_std": ips_s,
                          "boost_probe_linear_mean": lin_m, "boost_probe_linear_std": lin_s,
                          "boost_probe_nonlinear_mean": nl_m, "boost_probe_nonlinear_std": nl_s,
                          "seeds": MULTI_BIAS_SEEDS, "n_seeds": len(MULTI_BIAS_SEEDS)}
        print(f"  {arch:16s}  NDCG@10={ndcg_m:.4f}+/-{ndcg_s:.4f}  IPS-DCG={ips_m:.4f}+/-{ips_s:.4f}  "
              f"boost-probe linear={lin_m:.4f}+/-{lin_s:.4f}  nonlinear={nl_m:.4f}+/-{nl_s:.4f}  (chance=0.50)")
    print("\n  Lower boost-probe accuracy (closer to 0.50 chance) indicates the architecture's z_r")
    print("  absorbed LESS of the untargeted presentation-boost bias factor -- this is the direct")
    print("  test of whether the full architecture's separate bias encoder provides a genuine")
    print("  advantage when a second, untargeted bias factor is present.")

    out_file = f"multi_bias_factor_{DATASET}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved multi-bias-factor experiment results to {out_file}")
    return summary


# ============================================================================
# PART 12 — DISENTANGLEMENT METRICS: MIG and SAP (reviewer comment #3)
# ============================================================================
def extract_test_embeddings(enc_r, enc_b, pos_emb, test_set, k_max=K_MAX_POSITION):
    """
    For each test query, take the top-k_max documents by TRUE relevance
    (same protocol used for training click simulation, for consistency),
    and extract z_r (mu_r), z_b (mu_b) using the position each document
    would occupy under that relevance-ordered ranking. Returns:
        Z_r: (N, 24) array
        Z_b: (N, 16) array
        rel_factor: (N,) true relevance labels (the "relevance" generative factor)
        pos_factor: (N,) position index 0..k_max-1 (the "position" generative factor)
    """
    enc_r.eval(); enc_b.eval()
    Zr_list, Zb_list, rel_list, pos_list = [], [], [], []
    with torch.no_grad():
        for qid, idxs in test_set.queries.items():
            feats = test_set.features[idxs]
            rels = test_set.rels[idxs]
            if len(rels) < 2:
                continue
            order = np.argsort(-rels)[:k_max]
            ranked_feats = feats[order]
            ranked_rels = rels[order]
            n = len(ranked_rels)
            x = torch.tensor(ranked_feats, dtype=torch.float32, device=DEVICE)
            k_idx = torch.clamp(torch.arange(n, device=DEVICE), 0, K_MAX_POSITION - 1)
            pos_vec = pos_emb(k_idx)
            _, mu_r, _ = enc_r(x)
            _, mu_b, _ = enc_b(x, pos_vec)
            Zr_list.append(mu_r.cpu().numpy())
            Zb_list.append(mu_b.cpu().numpy())
            rel_list.append(ranked_rels)
            pos_list.append(np.arange(n))
    Zr = np.concatenate(Zr_list, axis=0)
    Zb = np.concatenate(Zb_list, axis=0)
    rel_factor = np.concatenate(rel_list, axis=0).astype(int)
    pos_factor = np.concatenate(pos_list, axis=0).astype(int)
    return Zr, Zb, rel_factor, pos_factor


def _discrete_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_mig(Z, factor, n_neighbors=3, random_state=SEED):
    """
    Mutual Information Gap (Chen et al., 2018).
    Z: (N, d) latent block. factor: (N,) discrete generative factor.
    Uses a k-NN based MI estimator (mutual_info_classif) per dimension,
    since Z is continuous and the factor is discrete/ordinal.
    Returns MIG in [0, 1] (higher = more disentangled: one dimension
    clearly "owns" this factor, with a large gap to the runner-up).
    """
    from sklearn.feature_selection import mutual_info_classif
    mi_per_dim = mutual_info_classif(Z, factor, discrete_features=False,
                                      n_neighbors=n_neighbors, random_state=random_state)
    sorted_mi = np.sort(mi_per_dim)[::-1]
    top1, top2 = sorted_mi[0], sorted_mi[1] if len(sorted_mi) > 1 else 0.0
    H = _discrete_entropy(factor)
    mig = (top1 - top2) / H if H > 1e-9 else 0.0
    return float(mig), mi_per_dim


def compute_sap(Z, factor, test_frac=0.3, random_state=SEED):
    """
    Separated Attribute Predictability (Kumar et al., 2018).
    For each latent dimension, fit a simple linear model predicting the
    factor from that dimension alone (R^2 on a held-out split); SAP is
    the gap between the best and second-best dimension's R^2.
    Returns SAP in roughly [0, 1] (higher = more disentangled).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    n = Z.shape[0]
    idx_train, idx_test = train_test_split(np.arange(n), test_size=test_frac,
                                            random_state=random_state)
    scores = []
    for d in range(Z.shape[1]):
        x_train = Z[idx_train, d].reshape(-1, 1)
        x_test = Z[idx_test, d].reshape(-1, 1)
        y_train, y_test = factor[idx_train], factor[idx_test]
        reg = LinearRegression().fit(x_train, y_train)
        pred = reg.predict(x_test)
        r2 = max(r2_score(y_test, pred), 0.0)  # clip negative R^2 to 0
        scores.append(r2)
    scores = np.sort(scores)[::-1]
    top1, top2 = scores[0], scores[1] if len(scores) > 1 else 0.0
    return float(top1 - top2), scores


def compute_block_informativeness(Z, factor, test_frac=0.3, random_state=SEED):
    """
    Whole-block informativeness probe (DCI-style; Eastwood & Williams, 2018).

    Unlike MIG/SAP, which score each latent DIMENSION individually and
    therefore assume axis-aligned factors, this probe uses the FULL latent
    BLOCK (all dimensions jointly) to predict the factor. This matches what
    this architecture actually claims: block-level independence between z_r
    and z_b (enforced via the covariance penalty and adversarial training on
    the whole vector), NOT axis-alignment of any single dimension. Reports
    R^2 on a held-out split for both a linear probe (Ridge) and a nonlinear
    probe (Random Forest), since the model's own RankingHead is a deep
    nonlinear network, so relevance may be encoded in a way a linear probe
    alone would under-detect.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    idx_train, idx_test = train_test_split(np.arange(Z.shape[0]), test_size=test_frac,
                                            random_state=random_state)
    X_train, X_test = Z[idx_train], Z[idx_test]
    y_train, y_test = factor[idx_train], factor[idx_test]

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    r2_linear = max(r2_score(y_test, ridge.predict(X_test)), 0.0)

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1,
                                random_state=random_state)
    rf.fit(X_train, y_train)
    r2_nonlinear = max(r2_score(y_test, rf.predict(X_test)), 0.0)

    return {"R2_linear": float(r2_linear), "R2_nonlinear": float(r2_nonlinear)}


def compute_position_classifier_accuracy(Z, pos_factor, n_positions=K_MAX_POSITION,
                                          test_frac=0.3, random_state=SEED):
    """
    Direct probe requested by a reviewer (Reviewer #5, R5-5): "train a
    fresh classifier to predict position from frozen z_r and report its
    accuracy against chance." Unlike the adversarial discriminator (which
    is trained jointly, adversarially, as part of the model itself), this
    classifier is trained FRESH and INDEPENDENTLY after training is
    complete, on frozen (detached) Z -- gradients from this classifier
    never reach the encoder. This directly tests whether z_r retains
    position information that an independent attacker could exploit,
    which is a stronger and more literal test than either the adversarial
    discriminator's own accuracy (which is entangled with training) or
    the whole-block regression R^2 (compute_block_informativeness above,
    which reports variance explained rather than classification accuracy).

    Reports both a linear (Logistic Regression) and nonlinear (Random
    Forest) classifier's accuracy, alongside the chance-level baseline
    (1/n_positions for uniform K-way classification).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    idx_train, idx_test = train_test_split(np.arange(Z.shape[0]), test_size=test_frac,
                                            random_state=random_state, stratify=pos_factor)
    X_train, X_test = Z[idx_train], Z[idx_test]
    y_train, y_test = pos_factor[idx_train], pos_factor[idx_test]

    logreg = LogisticRegression(max_iter=1000, multi_class="multinomial").fit(X_train, y_train)
    acc_linear = float(accuracy_score(y_test, logreg.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                                 random_state=random_state)
    rf.fit(X_train, y_train)
    acc_nonlinear = float(accuracy_score(y_test, rf.predict(X_test)))

    chance = 1.0 / n_positions
    return {"accuracy_linear": acc_linear, "accuracy_nonlinear": acc_nonlinear,
            "chance_level": chance,
            "linear_vs_chance": acc_linear - chance,
            "nonlinear_vs_chance": acc_nonlinear - chance}


def compute_silhouette_by_position(Zb, pos_factor, sample_size=2000, random_state=SEED):
    """
    Quantitative alternative to visual t-SNE inspection, requested by a
    reviewer: "the z_b space colored by position does not visibly show the
    'strong clustering' described. Consider a quantitative measure (e.g.,
    silhouette score by position) rather than relying on the visual."

    Computes the silhouette score (Rousseeuw, 1987) for z_b using the
    position label as the cluster assignment: for each point, this measures
    how much closer it is to other points sharing its position than to
    points at other positions, in the ORIGINAL 16-dimensional z_b space (not
    a 2D t-SNE projection, avoiding any distortion t-SNE's dimensionality
    reduction could introduce into a visual impression). Silhouette scores
    range from -1 (points are on average closer to other clusters than
    their own -- no clustering structure) to +1 (points are far closer to
    their own cluster than any other -- clean, well-separated clustering);
    scores near 0 indicate overlapping or ambiguous cluster boundaries.

    Subsamples to `sample_size` points for tractability, since silhouette
    score computation is O(n^2) in the number of points.
    """
    from sklearn.metrics import silhouette_score

    n = Zb.shape[0]
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        Zb_s, pos_s = Zb[idx], pos_factor[idx]
    else:
        Zb_s, pos_s = Zb, pos_factor

    score = silhouette_score(Zb_s, pos_s, metric="euclidean")
    return {"silhouette_score": float(score), "n_samples": int(len(pos_s))}


def plot_tsne_disentanglement(Zr, Zb, rel_factor, pos_factor, output_path,
                               n_samples=1000, random_state=SEED):
    """
    Regenerates Figure 4-style t-SNE visualization (Section 6.4): a 2x2 panel
    -- z_r colored by relevance, z_r colored by position, z_b colored by
    position, z_b colored by relevance -- on a random sample of n_samples
    test documents, matching the manuscript's "1,000 randomly sampled test
    documents" protocol. Each latent block is projected independently.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n = Zr.shape[0]
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n, size=min(n_samples, n), replace=False)
    Zr_s = Zr[sample_idx]
    Zb_s = Zb[sample_idx]
    rel_s = rel_factor[sample_idx]
    pos_s = pos_factor[sample_idx]

    print(f"\nRunning t-SNE on z_r ({Zr_s.shape[0]} samples, {Zr_s.shape[1]}-D)...")
    tsne_r = TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(Zr_s)
    print(f"Running t-SNE on z_b ({Zb_s.shape[0]} samples, {Zb_s.shape[1]}-D)...")
    tsne_b = TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(Zb_s)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Latent Space Disentanglement: z_r (Relevance) vs z_b (Bias)",
                 fontsize=15, fontweight="bold")

    sc = axes[0, 0].scatter(tsne_r[:, 0], tsne_r[:, 1], c=rel_s, cmap="RdYlGn",
                             s=25, edgecolors="k", linewidths=0.3, alpha=0.85)
    axes[0, 0].set_title("z_r (Relevance Space) - Colored by True Relevance", fontweight="bold")
    plt.colorbar(sc, ax=axes[0, 0], label="Relevance Label")

    sc = axes[0, 1].scatter(tsne_r[:, 0], tsne_r[:, 1], c=pos_s, cmap="viridis",
                             s=25, edgecolors="k", linewidths=0.3, alpha=0.85)
    axes[0, 1].set_title("z_r (Relevance Space) - Colored by Position", fontweight="bold")
    axes[0, 1].text(0.02, 0.98, "Should NOT cluster by position", transform=axes[0, 1].transAxes,
                     va="top", bbox=dict(facecolor="lightgreen", alpha=0.6))
    plt.colorbar(sc, ax=axes[0, 1], label="Position")

    sc = axes[1, 0].scatter(tsne_b[:, 0], tsne_b[:, 1], c=pos_s, cmap="viridis",
                             s=25, edgecolors="k", linewidths=0.3, alpha=0.85)
    axes[1, 0].set_title("z_b (Bias Space) - Colored by Position", fontweight="bold")
    axes[1, 0].text(0.02, 0.98, "Should cluster by position", transform=axes[1, 0].transAxes,
                     va="top", bbox=dict(facecolor="lightgreen", alpha=0.6))
    plt.colorbar(sc, ax=axes[1, 0], label="Position")

    sc = axes[1, 1].scatter(tsne_b[:, 0], tsne_b[:, 1], c=rel_s, cmap="RdYlGn",
                             s=25, edgecolors="k", linewidths=0.3, alpha=0.85)
    axes[1, 1].set_title("z_b (Bias Space) - Colored by Relevance", fontweight="bold")
    axes[1, 1].text(0.02, 0.98, "Should NOT cluster by relevance", transform=axes[1, 1].transAxes,
                     va="top", bbox=dict(facecolor="lightgreen", alpha=0.6))
    plt.colorbar(sc, ax=axes[1, 1], label="Relevance Label")

    for ax in axes.flat:
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE figure to {output_path}")


def plot_training_dynamics(history, output_path):
    """
    Regenerates Figure 5 (Training Dynamics) directly from logged per-epoch
    loss components and validation NDCG@10 from an actual, verified
    training run -- addresses reviewer observation that the original
    Figure 5 was internally inconsistent with its own caption (total loss
    rising rather than converging, KL dropping abruptly rather than rising
    through the annealing schedule, validation NDCG peaking early then
    declining rather than stabilizing, x-axis truncated before epoch 60).
    Unlike the original figure, every value plotted here is taken directly
    from `history`, recorded during training (see train_beta_vae's
    track_history option), so the figure is guaranteed consistent with
    the run that produced it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = history["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Training Dynamics: β-VAE", fontsize=15, fontweight="bold")

    axes[0, 0].plot(ep, history["total_loss"], color="blue", label="β-VAE Total Loss")
    axes[0, 0].set_title("Training Loss", fontweight="bold")
    axes[0, 0].legend()

    axes[0, 1].plot(ep, history["rec_loss"], color="green", label="Reconstruction")
    axes[0, 1].set_title("Reconstruction Loss", fontweight="bold")
    axes[0, 1].legend()

    axes[0, 2].plot(ep, history["kl_r"], color="red", label="KL(z_r)")
    axes[0, 2].plot(ep, history["kl_b"], color="orange", label="KL(z_b)")
    axes[0, 2].set_title("KL Divergence (β Annealing)", fontweight="bold")
    axes[0, 2].legend()

    axes[1, 0].plot(ep, history["rank_loss"], color="purple", label="Ranking Loss")
    axes[1, 0].set_title("Ranking Loss", fontweight="bold")
    axes[1, 0].legend()

    axes[1, 1].plot(ep, history["adv_loss"], color="darkred", label="Adversarial Loss")
    axes[1, 1].set_title("Adversarial Loss (Debiasing)", fontweight="bold")
    axes[1, 1].legend()

    axes[1, 2].plot(ep, history["val_ndcg"], color="blue", marker="o", markersize=3, label="β-VAE Val NDCG@10")
    axes[1, 2].set_title("Validation NDCG@10", fontweight="bold")
    axes[1, 2].legend()

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
    axes[1, 2].set_ylabel("NDCG@10")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training dynamics figure to {output_path}")


def compute_cross_correlation(Zr, Zb):
    """
    Full pairwise Pearson correlation between every z_r dimension and
    every z_b dimension (a d_r x d_b matrix, 24x16 for this architecture),
    computed on frozen test-set embeddings. Reports the mean absolute
    correlation and its standard deviation across all dimension pairs,
    exactly the two figures already reported for Yahoo! LETOR in Section
    6.4 (mean |rho| = 0.0237, sigma = 0.031) -- addresses the manuscript's
    outstanding [TBD] for the equivalent MSLR-WEB-30K figures.

    This is a linear-correlation measure, distinct from the covariance
    penalty (cov_penalty()) used during TRAINING: cov_penalty is an
    unnormalized sum-of-squared-covariances loss term, while this is a
    normalized, dimension-by-dimension Pearson correlation computed
    post-hoc for analysis, matching the manuscript's own stated caveat
    that this measure captures only linear dependence (Section 6.4).
    """
    Zr_c = Zr - Zr.mean(axis=0, keepdims=True)
    Zb_c = Zb - Zb.mean(axis=0, keepdims=True)
    Zr_std = Zr.std(axis=0, keepdims=True) + 1e-8
    Zb_std = Zb.std(axis=0, keepdims=True) + 1e-8
    Zr_norm = Zr_c / Zr_std
    Zb_norm = Zb_c / Zb_std
    corr_matrix = (Zr_norm.T @ Zb_norm) / Zr.shape[0]  # (d_r, d_b) Pearson correlations
    mean_abs_corr = float(np.mean(np.abs(corr_matrix)))
    std_corr = float(np.std(corr_matrix))
    return {"mean_abs_cross_correlation": mean_abs_corr, "std_cross_correlation": std_corr,
            "matrix_shape": list(corr_matrix.shape)}


def run_disentanglement_metrics(enc_r, enc_b, pos_emb, test_set):
    print("\n" + "=" * 80)
    print("DISENTANGLEMENT METRICS (MIG, SAP) — reviewer comment #3")
    print("=" * 80)
    Zr, Zb, rel_factor, pos_factor = extract_test_embeddings(enc_r, enc_b, pos_emb, test_set)
    print(f"Extracted embeddings: Zr={Zr.shape}, Zb={Zb.shape}, "
          f"n_documents={len(rel_factor)}")

    print("\n" + "-" * 80)
    print("CROSS-CORRELATION: z_r vs z_b (linear-correlation measure; Section 6.4)")
    print("-" * 80)
    cross_corr = compute_cross_correlation(Zr, Zb)
    print(f"  Mean absolute cross-correlation: {cross_corr['mean_abs_cross_correlation']:.4f}")
    print(f"  Standard deviation of cross-correlation distribution: {cross_corr['std_cross_correlation']:.4f}")
    print(f"  Matrix shape (d_r x d_b): {cross_corr['matrix_shape']}")
    print("  This is a linear (Pearson-style) measure only; it does not on its own establish")
    print("  statistical independence, since z_r and z_b could be uncorrelated while remaining")
    print("  dependent through a nonlinear relationship (see MIG/SAP/whole-block probes below).")

    results = {"cross_correlation": cross_corr}
    for block_name, Z in [("z_r (relevance space)", Zr), ("z_b (bias space)", Zb)]:
        for factor_name, factor in [("relevance", rel_factor), ("position", pos_factor)]:
            mig, _ = compute_mig(Z, factor)
            sap, _ = compute_sap(Z, factor)
            results[f"{block_name} vs {factor_name}"] = {"MIG": mig, "SAP": sap}
            print(f"  {block_name:24s} vs {factor_name:10s}  MIG={mig:.4f}  SAP={sap:.4f}")

    print("\nExpected pattern for successful disentanglement (per-dimension, axis-aligned):")
    print("  z_r vs relevance -> HIGH (z_r should predict relevance well, cleanly)")
    print("  z_r vs position  -> LOW  (z_r should NOT predict position)")
    print("  z_b vs position  -> HIGH (z_b should predict position well, cleanly)")
    print("  z_b vs relevance -> LOW  (z_b should NOT predict relevance)")
    print("  NOTE: MIG/SAP score individual dimensions and assume axis-aligned")
    print("  factors. This architecture instead targets BLOCK-level independence")
    print("  (enforced via the covariance penalty and adversarial training on the")
    print("  full z_r/z_b vectors), with relevance/bias expected to be encoded as")
    print("  a distributed pattern across each block rather than concentrated in")
    print("  a single dimension. The whole-block probe below tests that claim.")

    print("\n" + "-" * 80)
    print("WHOLE-BLOCK INFORMATIVENESS PROBE (DCI-style; complementary to MIG/SAP)")
    print("-" * 80)
    block_results = {}
    for block_name, Z in [("z_r (relevance space)", Zr), ("z_b (bias space)", Zb)]:
        for factor_name, factor in [("relevance", rel_factor), ("position", pos_factor)]:
            probe = compute_block_informativeness(Z, factor)
            block_results[f"{block_name} vs {factor_name}"] = probe
            print(f"  {block_name:24s} vs {factor_name:10s}  "
                  f"R2_linear={probe['R2_linear']:.4f}  R2_nonlinear={probe['R2_nonlinear']:.4f}")

    print("\nExpected pattern for successful disentanglement (whole-block):")
    print("  z_r vs relevance -> HIGH R2 (the full z_r block predicts relevance well)")
    print("  z_r vs position  -> LOW  R2 (the full z_r block does NOT predict position)")
    print("  z_b vs position  -> HIGH R2 (the full z_b block predicts position well)")
    print("  z_b vs relevance -> LOW  R2 (the full z_b block does NOT predict relevance)")

    print("\n" + "-" * 80)
    print("FROZEN z_r POSITION-CLASSIFIER PROBE (reviewer request, R5-5)")
    print("-" * 80)
    probe = compute_position_classifier_accuracy(Zr, pos_factor)
    chance = probe["chance_level"]
    print(f"  Chance level (uniform {K_MAX_POSITION}-way classification): {chance:.4f} ({chance*100:.2f}%)")
    print(f"  Logistic Regression accuracy: {probe['accuracy_linear']:.4f} "
          f"({probe['accuracy_linear']*100:.2f}%)  [{probe['linear_vs_chance']*100:+.2f}pp vs. chance]")
    print(f"  Random Forest accuracy:       {probe['accuracy_nonlinear']:.4f} "
          f"({probe['accuracy_nonlinear']*100:.2f}%)  [{probe['nonlinear_vs_chance']*100:+.2f}pp vs. chance]")
    print("  A freshly-trained classifier achieving accuracy close to chance indicates")
    print("  z_r retains little exploitable position information beyond what a random")
    print("  guesser would achieve, independent of the adversarial discriminator itself.")
    results["position_classifier_probe_zr"] = probe

    print("\n" + "-" * 80)
    print("SILHOUETTE SCORE: z_b CLUSTERING BY POSITION (quantitative alternative to visual t-SNE)")
    print("-" * 80)
    sil = compute_silhouette_by_position(Zb, pos_factor)
    print(f"  Silhouette score (z_b, position labels, n={sil['n_samples']}): "
          f"{sil['silhouette_score']:.4f}")
    print("  Range: -1 (no clustering structure) to +1 (clean, well-separated clusters).")
    print("  Computed directly in the full 16-D z_b space (not a 2D t-SNE projection),")
    print("  so this is unaffected by any visual distortion t-SNE's dimensionality")
    print("  reduction might introduce.")
    results["silhouette_score_zb_position"] = sil

    tsne_path = f"tsne_{DATASET}.png"
    plot_tsne_disentanglement(Zr, Zb, rel_factor, pos_factor, tsne_path)

    results["block_informativeness"] = block_results
    return results


# ============================================================================
# PART 13 — POSITION-BIAS SENSITIVITY SWEEP (reviewer comments #5/#6)
# ============================================================================
# Reduced epoch/patience budget used ONLY inside this sweep. The sweep tests
# whether beta-VAE's advantage over Affine is STABLE in direction across
# different assumed bias severities/forms -- that is a relative comparison,
# not a claim about each setting's best-achievable absolute performance, so
# a smaller training budget per setting is an appropriate and explicit scope
# choice here. All other results in this script use the full paper-spec
# budget (EPOCHS_VAE=80, EPOCHS_BASELINE=50) and are unaffected by this.
SWEEP_EPOCHS_VAE = 30
SWEEP_EPOCHS_BASELINE = 20
SWEEP_PATIENCE = 8


def run_position_bias_sensitivity_sweep(train_set, vali_set, test_set):
    """
    Addresses reviewer comment #5/#6: "the simulated position bias is
    modeled using a fixed function (theta_k = k^-0.9)... Please justify
    this choice or include experiments under more diverse and realistic
    bias settings."

    Retrains beta-VAE and Affine (ground-truth) under each of several
    position-bias severities/forms and compares NDCG@10 / IPS-DCG. The
    claim being tested is that beta-VAE's advantage over Affine (its
    strongest baseline; see Table 1) is stable in DIRECTION across settings,
    not that either method is optimally tuned for every setting.
    """
    global THETA_FORM, THETA_KAPPA, THETA_FORM_EVAL, THETA_KAPPA_EVAL
    orig_form, orig_kappa = THETA_FORM, THETA_KAPPA  # restore at the end
    orig_form_eval, orig_kappa_eval = THETA_FORM_EVAL, THETA_KAPPA_EVAL

    settings = [
        ("power", 0.5, "theta_k = k^-0.5 (mild position bias)"),
        ("power", 0.7, "theta_k = k^-0.7"),
        ("power", 0.9, "theta_k = k^-0.9 (paper default)"),
        ("power", 1.1, "theta_k = k^-1.1"),
        ("power", 1.3, "theta_k = k^-1.3 (severe position bias)"),
        ("log", None, "theta_k = 1/log2(k+1) (alternative functional form)"),
    ]

    print("\n" + "=" * 80)
    print("POSITION-BIAS SENSITIVITY SWEEP (reviewer comment #5/#6)")
    print(f"NOTE: reduced training budget per setting (VAE={SWEEP_EPOCHS_VAE} epochs, "
          f"baseline={SWEEP_EPOCHS_BASELINE} epochs, patience={SWEEP_PATIENCE}) -- "
          f"this sweep tests RELATIVE stability across settings, not per-setting "
          f"optimal performance. See Section [X] for full-budget primary results.")
    print("=" * 80)

    sweep_results = []
    for form, kappa, label in settings:
        THETA_FORM = form
        THETA_KAPPA = kappa if kappa is not None else THETA_KAPPA
        THETA_FORM_EVAL, THETA_KAPPA_EVAL = THETA_FORM, THETA_KAPPA  # keep matched: this sweep tests train=eval across settings, unlike run_mismatched_bias_evaluation()
        print(f"\n--- Setting: {label} ---")

        affine_model = train_affine(train_set, vali_set, epochs=SWEEP_EPOCHS_BASELINE,
                                     estimated=False)
        enc_r_vae, rank_head_vae, _, _, _ = train_beta_vae(train_set, vali_set,
                                                          epochs=SWEEP_EPOCHS_VAE)

        affine_ndcg, affine_ips = evaluate_ranker_fn(
            make_score_fn_simple(affine_model), test_set, k=10, n_runs_ips=5)
        vae_ndcg, vae_ips = evaluate_ranker_fn(
            make_score_fn_vae(enc_r_vae, rank_head_vae), test_set, k=10, n_runs_ips=5)

        row = {
            "setting": label, "theta_form": form, "theta_kappa": THETA_KAPPA,
            "Affine_NDCG10": float(np.mean(affine_ndcg)), "Affine_IPSDCG": float(np.mean(affine_ips)),
            "betaVAE_NDCG10": float(np.mean(vae_ndcg)), "betaVAE_IPSDCG": float(np.mean(vae_ips)),
        }
        row["betaVAE_wins_NDCG"] = row["betaVAE_NDCG10"] > row["Affine_NDCG10"]
        row["betaVAE_wins_IPSDCG"] = row["betaVAE_IPSDCG"] > row["Affine_IPSDCG"]
        sweep_results.append(row)
        del affine_model, enc_r_vae, rank_head_vae
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  Affine:  NDCG@10={row['Affine_NDCG10']:.4f}  IPS-DCG={row['Affine_IPSDCG']:.4f}")
        print(f"  beta-VAE: NDCG@10={row['betaVAE_NDCG10']:.4f}  IPS-DCG={row['betaVAE_IPSDCG']:.4f}")
        print(f"  beta-VAE wins: NDCG={row['betaVAE_wins_NDCG']}  IPS-DCG={row['betaVAE_wins_IPSDCG']}")

    THETA_FORM, THETA_KAPPA = orig_form, orig_kappa  # restore paper-default setting
    THETA_FORM_EVAL, THETA_KAPPA_EVAL = orig_form_eval, orig_kappa_eval

    print("\n" + "-" * 80)
    print("SWEEP SUMMARY")
    print("-" * 80)
    n_wins_ndcg = sum(r["betaVAE_wins_NDCG"] for r in sweep_results)
    n_wins_ips = sum(r["betaVAE_wins_IPSDCG"] for r in sweep_results)
    print(f"beta-VAE beat Affine on NDCG@10 in {n_wins_ndcg}/{len(sweep_results)} settings")
    print(f"beta-VAE beat Affine on IPS-DCG in {n_wins_ips}/{len(sweep_results)} settings")

    sweep_file = f"sensitivity_sweep_{DATASET}_seed{SEED}.json"
    with open(sweep_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Saved sweep results to {sweep_file}")
    return sweep_results


# ============================================================================
# PART 14 — HYPERPARAMETER SENSITIVITY SWEEP (reviewer comment on
# hyperparameter selection/sensitivity)
# ============================================================================
def run_hyperparameter_sensitivity_sweep(train_set, vali_set, test_set):
    """
    Addresses the reviewer question: "How were the beta coefficient,
    adversarial loss weight, latent-space dimensions, and other critical
    hyperparameters selected, and how sensitive are the ranking results to
    these choices?"

    Varies lambda_rank, lambda_cov, and lambda_adv_max ONE AT A TIME around
    their paper-reported defaults (lambda_rank=2.0, lambda_cov=0.001,
    lambda_adv_max=1.0), holding all other hyperparameters fixed, and
    retrains beta-VAE only (Affine and other baselines are unaffected by
    these beta-VAE-specific hyperparameters, so are not re-run here).
    Reports NDCG@10 / IPS-DCG for each value, using the same reduced
    training budget as the position-bias sweep (see note there) -- this
    tests RELATIVE sensitivity to each hyperparameter, not per-value
    optimal tuning.

    NOTE: d_r, d_b (latent dimensions) are NOT included in this sweep --
    varying them changes the architecture itself (encoder/decoder/ranking
    head/discriminator input-output shapes all depend on d_r, d_b), which
    is a substantially larger undertaking than varying a loss weight. If
    needed later, this would require a separate, more involved sweep.
    """
    DEFAULT_LAMBDA_RANK = 2.0
    DEFAULT_LAMBDA_COV = 0.001
    DEFAULT_LAMBDA_ADV_MAX = 1.0

    sweep_groups = [
        ("lambda_rank", [1.0, 1.5, 2.0, 2.5, 3.0], DEFAULT_LAMBDA_RANK),
        ("lambda_cov", [0.0001, 0.001, 0.01], DEFAULT_LAMBDA_COV),
        ("lambda_adv_max", [0.5, 0.75, 1.0], DEFAULT_LAMBDA_ADV_MAX),
    ]

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SENSITIVITY SWEEP")
    print(f"NOTE: reduced training budget per setting (VAE={SWEEP_EPOCHS_VAE} epochs) -- "
          f"this sweep tests RELATIVE sensitivity to each hyperparameter, not "
          f"per-value optimal performance. d_r/d_b (architecture-changing) are "
          f"NOT included -- see docstring.")
    print("=" * 80)

    sweep_results = []
    for hp_name, values, default_val in sweep_groups:
        for val in values:
            kwargs = {"lambda_rank": DEFAULT_LAMBDA_RANK,
                      "lambda_cov": DEFAULT_LAMBDA_COV,
                      "lambda_adv_max": DEFAULT_LAMBDA_ADV_MAX}
            kwargs[hp_name] = val
            is_default = (val == default_val)
            tag = " (default)" if is_default else ""
            print(f"\n--- {hp_name} = {val}{tag} (others at default) ---")

            enc_r_vae, rank_head_vae, _, _, _ = train_beta_vae(
                train_set, vali_set, epochs=SWEEP_EPOCHS_VAE, **kwargs)
            vae_ndcg, vae_ips = evaluate_ranker_fn(
                make_score_fn_vae(enc_r_vae, rank_head_vae), test_set, k=10, n_runs_ips=5)

            row = {
                "hyperparameter": hp_name, "value": val, "is_default": is_default,
                "NDCG10": float(np.mean(vae_ndcg)), "IPSDCG": float(np.mean(vae_ips)),
            }
            sweep_results.append(row)
            print(f"  NDCG@10={row['NDCG10']:.4f}  IPS-DCG={row['IPSDCG']:.4f}")
            del enc_r_vae, rank_head_vae
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # compute deltas relative to each hyperparameter's default-value row
    for hp_name, values, default_val in sweep_groups:
        default_row = next(r for r in sweep_results
                            if r["hyperparameter"] == hp_name and r["is_default"])
        for r in sweep_results:
            if r["hyperparameter"] == hp_name:
                r["delta_NDCG10_pct"] = 100 * (r["NDCG10"] - default_row["NDCG10"]) / default_row["NDCG10"]
                r["delta_IPSDCG_pct"] = 100 * (r["IPSDCG"] - default_row["IPSDCG"]) / default_row["IPSDCG"]

    print("\n" + "-" * 80)
    print("SWEEP SUMMARY (delta vs. each hyperparameter's default value)")
    print("-" * 80)
    for r in sweep_results:
        tag = " <- default" if r["is_default"] else ""
        print(f"  {r['hyperparameter']:16s} = {r['value']:<8}  "
              f"NDCG@10={r['NDCG10']:.4f} ({r['delta_NDCG10_pct']:+.2f}%)  "
              f"IPS-DCG={r['IPSDCG']:.4f} ({r['delta_IPSDCG_pct']:+.2f}%){tag}")

    sweep_file = f"hyperparam_sweep_{DATASET}_seed{SEED}.json"
    with open(sweep_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved hyperparameter sweep results to {sweep_file}")
    return sweep_results


# ============================================================================
# PART 15 — FACTORIAL ABLATION: ADVERSARIAL TRAINING x KL REGULARIZATION
# ACROSS BIAS STRENGTHS (reviewer question on separating their contributions)
# ============================================================================
def run_factorial_ablation_sweep(train_set, vali_set, test_set):
    """
    Addresses the reviewer question: "The ablation study reports that
    adversarial training is essential for eliminating bias; can the authors
    quantitatively separate the contributions of beta-VAE regularization
    and adversarial training, particularly under different bias strengths
    and latent-space configurations?"

    The original ablation (Table 4) removes each component ONE AT A TIME
    from the full model, at a single fixed bias strength (theta_k=k^-0.9).
    This does not (a) isolate whether the two components INTERACT (does
    removing both together cause more/less damage than the sum of removing
    each alone?), or (b) test whether this separation holds as bias
    strength varies. This sweep addresses both: a full 2x2 factorial
    (adversarial training on/off x KL regularization on/off) crossed with
    3 bias-strength settings (kappa = 0.5 mild, 0.9 default, 1.3 severe),
    12 total training runs at the reduced sweep budget (see note on
    SWEEP_EPOCHS_VAE above -- this tests relative/interaction effects, not
    per-cell optimal tuning).

    Latent-space CONFIGURATIONS (d_r, d_b) are NOT included here, for the
    same reason given in run_hyperparameter_sensitivity_sweep's docstring:
    varying them changes the architecture itself, a substantially larger
    undertaking than toggling loss terms.
    """
    global THETA_FORM, THETA_KAPPA, THETA_FORM_EVAL, THETA_KAPPA_EVAL
    orig_form, orig_kappa = THETA_FORM, THETA_KAPPA
    orig_form_eval, orig_kappa_eval = THETA_FORM_EVAL, THETA_KAPPA_EVAL

    bias_settings = [0.5, 0.9, 1.3]
    factorial_cells = [
        (True, True, "full (adv=on, KL=on)"),
        (True, False, "adv=on, KL=off"),
        (False, True, "adv=off, KL=on"),
        (False, False, "both off"),
    ]

    print("\n" + "=" * 80)
    print("FACTORIAL ABLATION: ADVERSARIAL TRAINING x KL REGULARIZATION x BIAS STRENGTH")
    print(f"NOTE: reduced training budget per cell (VAE={SWEEP_EPOCHS_VAE} epochs) -- "
          f"tests relative/interaction effects, not per-cell optimal performance.")
    print("=" * 80)

    THETA_FORM = "power"
    THETA_FORM_EVAL = "power"
    results = []
    for kappa in bias_settings:
        THETA_KAPPA = kappa
        THETA_KAPPA_EVAL = kappa  # keep matched: this sweep tests train=eval, unlike run_mismatched_bias_evaluation()
        print(f"\n--- Bias strength: theta_k = k^-{kappa} ---")
        cell_results = {}
        for adv_on, kl_on, label in factorial_cells:
            print(f"  [{label}]")
            enc_r_vae, rank_head_vae, _, _, _ = train_beta_vae(
                train_set, vali_set, epochs=SWEEP_EPOCHS_VAE,
                adv_enabled=adv_on, kl_reg_enabled=kl_on)
            ndcg, ips = evaluate_ranker_fn(
                make_score_fn_vae(enc_r_vae, rank_head_vae), test_set, k=10, n_runs_ips=5)
            row = {
                "kappa": kappa, "adv_enabled": adv_on, "kl_reg_enabled": kl_on,
                "label": label, "NDCG10": float(np.mean(ndcg)), "IPSDCG": float(np.mean(ips)),
            }
            results.append(row)
            cell_results[(adv_on, kl_on)] = row
            print(f"    NDCG@10={row['NDCG10']:.4f}  IPS-DCG={row['IPSDCG']:.4f}")

            # Explicit cleanup: without this, GPU memory from each of the 12
            # sequential model trainings can accumulate (especially on MSLR,
            # with its larger per-query document counts), eventually causing
            # a CUDA out-of-memory error that crashes the kernel/process
            # rather than raising a catchable Python exception.
            del enc_r_vae, rank_head_vae
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # interaction effect for this bias strength, on IPS-DCG:
        # if effects were purely additive (no interaction), the "both off"
        # drop should equal the sum of the two single-component drops.
        full = cell_results[(True, True)]["IPSDCG"]
        adv_off = cell_results[(False, True)]["IPSDCG"]
        kl_off = cell_results[(True, False)]["IPSDCG"]
        both_off = cell_results[(False, False)]["IPSDCG"]
        additive_prediction = full - (full - adv_off) - (full - kl_off)
        interaction = both_off - additive_prediction
        print(f"  Interaction check (IPS-DCG): both-off={both_off:.4f}, "
              f"additive-prediction={additive_prediction:.4f}, "
              f"interaction-residual={interaction:+.4f}")

    THETA_FORM, THETA_KAPPA = orig_form, orig_kappa
    THETA_FORM_EVAL, THETA_KAPPA_EVAL = orig_form_eval, orig_kappa_eval

    sweep_file = f"factorial_ablation_{DATASET}_seed{SEED}.json"
    with open(sweep_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved factorial ablation results to {sweep_file}")
    return results


# ============================================================================
# PART 15.5 — MULTI-SEED ABLATION WITH VARIANCE (Reviewer #9, Point 6)
# ============================================================================
ABLATION_SEEDS = [42, 123, 7, 2024, 2026]   # 5 seeds, per reviewer request


def run_multiseed_ablation(train_set, vali_set, test_set, epochs=SWEEP_EPOCHS_VAE):
    """
    Addresses Reviewer #9, Point 6: "The ablation table reports single-run
    point estimates for an adversarially trained model, which is known to
    be sensitive to initialization. Please report results averaged over at
    least five seeds with variance."

    Retrains all 6 ablation variants (full model, w/o adversarial training,
    w/o beta-annealing, w/o covariance penalty, w/o pairwise ranking loss,
    single encoder / no separation) across 5 random seeds each, 30 total
    training runs, at the reduced sweep budget (see SWEEP_EPOCHS_VAE note
    above -- this tests relative seed-to-seed stability of each ablation
    finding, not per-cell optimal tuning, consistent with the scope of
    every other multi-run sweep in this script). Reports mean +/- standard
    deviation NDCG@10 and IPS-DCG per variant, replacing Table 4's current
    single-run point estimates.

    Re-seeds (random, numpy, torch) at the start of each seed's block, so
    both weight initialization AND click-simulation stochasticity vary
    across seeds, not just one or the other.
    """
    variants = [
        ("Full model", dict(adv_enabled=True, kl_reg_enabled=True, pairwise_loss_enabled=True, lambda_cov=0.001)),
        ("w/o adversarial training", dict(adv_enabled=False, kl_reg_enabled=True, pairwise_loss_enabled=True, lambda_cov=0.001)),
        ("w/o beta-annealing", dict(adv_enabled=True, kl_reg_enabled=False, pairwise_loss_enabled=True, lambda_cov=0.001)),
        ("w/o covariance penalty", dict(adv_enabled=True, kl_reg_enabled=True, pairwise_loss_enabled=True, lambda_cov=0.0)),
        ("w/o pairwise ranking loss", dict(adv_enabled=True, kl_reg_enabled=True, pairwise_loss_enabled=False, lambda_cov=0.001)),
        ("dual encoder w/o covariance or beta-VAE regularization", dict(adv_enabled=True, kl_reg_enabled=False, pairwise_loss_enabled=True, lambda_cov=0.0)),
    ]

    print("\n" + "=" * 80)
    print(f"MULTI-SEED ABLATION (Reviewer #9, Point 6): {len(variants)+1} variants x "
          f"{len(ABLATION_SEEDS)} seeds = {(len(variants)+1)*len(ABLATION_SEEDS)} training runs")
    print(f"NOTE: reduced training budget per run ({epochs} epochs) -- tests seed-to-seed "
          f"stability of each ablation finding, not per-cell optimal performance.")
    print("=" * 80)

    per_variant_results = defaultdict(lambda: {"NDCG10": [], "IPSDCG": []})
    simple_ranker_probe_results = {"accuracy_linear": [], "accuracy_nonlinear": [], "chance_level": []}
    no_zb_probe_results = {"accuracy_linear": [], "accuracy_nonlinear": [], "chance_level": []}

    for seed in ABLATION_SEEDS:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for name, kwargs in variants:
            print(f"  [{name}]")
            enc_r_v, rank_head_v, _, _, _ = train_beta_vae(train_set, vali_set, epochs=epochs, **kwargs)
            ndcg, ips = evaluate_ranker_fn(make_score_fn_vae(enc_r_v, rank_head_v), test_set, k=10, n_runs_ips=5)
            per_variant_results[name]["NDCG10"].append(float(np.mean(ndcg)))
            per_variant_results[name]["IPSDCG"].append(float(np.mean(ips)))
            print(f"    NDCG@10={np.mean(ndcg):.4f}  IPS-DCG={np.mean(ips):.4f}")
            del enc_r_v, rank_head_v
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        print(f"  [single encoder (no separation)]")
        enc_v, rank_head_v = train_single_encoder_ablation(train_set, vali_set, epochs=epochs)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_single_encoder(enc_v, rank_head_v), test_set, k=10, n_runs_ips=5)
        per_variant_results["single encoder (no separation)"]["NDCG10"].append(float(np.mean(ndcg)))
        per_variant_results["single encoder (no separation)"]["IPSDCG"].append(float(np.mean(ips)))
        print(f"    NDCG@10={np.mean(ndcg):.4f}  IPS-DCG={np.mean(ips):.4f}")
        del enc_v, rank_head_v
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print(f"  [simple position-adversarial ranker (no VAE, no dual encoder)]")
        enc_s, rank_head_s = train_simple_adversarial_ranker(train_set, vali_set, epochs=epochs)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_simple_adversarial(enc_s, rank_head_s), test_set, k=10, n_runs_ips=5)
        per_variant_results["simple position-adversarial ranker"]["NDCG10"].append(float(np.mean(ndcg)))
        per_variant_results["simple position-adversarial ranker"]["IPSDCG"].append(float(np.mean(ips)))
        print(f"    NDCG@10={np.mean(ndcg):.4f}  IPS-DCG={np.mean(ips):.4f}")
        probe = compute_position_classifier_probe_simple(enc_s, test_set, random_state=seed)
        simple_ranker_probe_results["accuracy_linear"].append(probe["accuracy_linear"])
        simple_ranker_probe_results["accuracy_nonlinear"].append(probe["accuracy_nonlinear"])
        simple_ranker_probe_results["chance_level"].append(probe["chance_level"])
        del enc_s, rank_head_s
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print(f"  [reconstruction + adversarial, no explicit z_b]")
        enc_nz, rank_head_nz = train_reconstruction_adversarial_no_zb(train_set, vali_set, epochs=epochs)
        ndcg, ips = evaluate_ranker_fn(make_score_fn_no_zb(enc_nz, rank_head_nz), test_set, k=10, n_runs_ips=5)
        per_variant_results["reconstruction + adversarial, no explicit z_b"]["NDCG10"].append(float(np.mean(ndcg)))
        per_variant_results["reconstruction + adversarial, no explicit z_b"]["IPSDCG"].append(float(np.mean(ips)))
        print(f"    NDCG@10={np.mean(ndcg):.4f}  IPS-DCG={np.mean(ips):.4f}")
        probe = compute_position_classifier_probe_simple(_MuExtractorWrapper(enc_nz), test_set, random_state=seed)
        no_zb_probe_results["accuracy_linear"].append(probe["accuracy_linear"])
        no_zb_probe_results["accuracy_nonlinear"].append(probe["accuracy_nonlinear"])
        no_zb_probe_results["chance_level"].append(probe["chance_level"])
        del enc_nz, rank_head_nz
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "-" * 80)
    print(f"MULTI-SEED ABLATION SUMMARY (mean +/- std across {len(ABLATION_SEEDS)} seeds)")
    print("-" * 80)
    summary = {}
    full_ndcg_mean = np.mean(per_variant_results["Full model"]["NDCG10"])
    full_ips_mean = np.mean(per_variant_results["Full model"]["IPSDCG"])
    for name, _ in variants + [("single encoder (no separation)", None),
                                ("simple position-adversarial ranker", None),
                                ("reconstruction + adversarial, no explicit z_b", None)]:
        ndcgs = per_variant_results[name]["NDCG10"]
        ipss = per_variant_results[name]["IPSDCG"]
        ndcg_mean, ndcg_std = float(np.mean(ndcgs)), float(np.std(ndcgs, ddof=1))
        ips_mean, ips_std = float(np.mean(ipss)), float(np.std(ipss, ddof=1))
        delta_ndcg_pct = 100 * (ndcg_mean - full_ndcg_mean) / full_ndcg_mean if name != "Full model" else 0.0
        delta_ips_pct = 100 * (ips_mean - full_ips_mean) / full_ips_mean if name != "Full model" else 0.0
        summary[name] = {"NDCG10_mean": ndcg_mean, "NDCG10_std": ndcg_std,
                          "IPSDCG_mean": ips_mean, "IPSDCG_std": ips_std,
                          "delta_NDCG10_pct": delta_ndcg_pct, "delta_IPSDCG_pct": delta_ips_pct,
                          "seeds": ABLATION_SEEDS, "n_seeds": len(ABLATION_SEEDS)}
        print(f"  {name:32s}  NDCG@10={ndcg_mean:.4f}+/-{ndcg_std:.4f} ({delta_ndcg_pct:+.2f}%)  "
              f"IPS-DCG={ips_mean:.4f}+/-{ips_std:.4f} ({delta_ips_pct:+.2f}%)")

    print("\n" + "-" * 80)
    print("SIMPLE POSITION-ADVERSARIAL RANKER: POSITION-CLASSIFIER PROBE "
          f"(mean +/- std across {len(ABLATION_SEEDS)} seeds)")
    print("-" * 80)
    lin = simple_ranker_probe_results["accuracy_linear"]
    nl = simple_ranker_probe_results["accuracy_nonlinear"]
    chance_mean = float(np.mean(simple_ranker_probe_results["chance_level"]))
    probe_summary = {
        "accuracy_linear_mean": float(np.mean(lin)), "accuracy_linear_std": float(np.std(lin, ddof=1)),
        "accuracy_nonlinear_mean": float(np.mean(nl)), "accuracy_nonlinear_std": float(np.std(nl, ddof=1)),
        "chance_level": chance_mean, "seeds": ABLATION_SEEDS, "n_seeds": len(ABLATION_SEEDS),
    }
    print(f"  Chance level: {chance_mean:.4f}")
    print(f"  Logistic Regression accuracy: {probe_summary['accuracy_linear_mean']:.4f}"
          f"+/-{probe_summary['accuracy_linear_std']:.4f}")
    print(f"  Random Forest accuracy:       {probe_summary['accuracy_nonlinear_mean']:.4f}"
          f"+/-{probe_summary['accuracy_nonlinear_std']:.4f}")
    print("  Compare directly against the main model's frozen-z_r probe (Section 6.4) to assess "
          "whether this simpler architecture achieves comparable position-invariance, not just "
          "comparable ranking performance.")
    summary["simple_position_adversarial_ranker_probe"] = probe_summary

    print("\n" + "-" * 80)
    print("RECONSTRUCTION + ADVERSARIAL, NO EXPLICIT z_b: POSITION-CLASSIFIER PROBE "
          f"(mean +/- std across {len(ABLATION_SEEDS)} seeds)")
    print("-" * 80)
    lin_nz = no_zb_probe_results["accuracy_linear"]
    nl_nz = no_zb_probe_results["accuracy_nonlinear"]
    chance_mean_nz = float(np.mean(no_zb_probe_results["chance_level"]))
    probe_summary_nz = {
        "accuracy_linear_mean": float(np.mean(lin_nz)), "accuracy_linear_std": float(np.std(lin_nz, ddof=1)),
        "accuracy_nonlinear_mean": float(np.mean(nl_nz)), "accuracy_nonlinear_std": float(np.std(nl_nz, ddof=1)),
        "chance_level": chance_mean_nz, "seeds": ABLATION_SEEDS, "n_seeds": len(ABLATION_SEEDS),
    }
    print(f"  Chance level: {chance_mean_nz:.4f}")
    print(f"  Logistic Regression accuracy: {probe_summary_nz['accuracy_linear_mean']:.4f}"
          f"+/-{probe_summary_nz['accuracy_linear_std']:.4f}")
    print(f"  Random Forest accuracy:       {probe_summary_nz['accuracy_nonlinear_mean']:.4f}"
          f"+/-{probe_summary_nz['accuracy_nonlinear_std']:.4f}")
    print("  This variant retains adversarial training and a reconstruction objective, but has no")
    print("  separate z_b -- compare against both the full model's probe (Section 6.4) and the")
    print("  simple ranker's probe above to isolate whether an explicit bias latent specifically")
    print("  matters, independent of whether reconstruction and adversarial training are present.")
    summary["reconstruction_adversarial_no_zb_probe"] = probe_summary_nz

    sweep_file = f"multiseed_ablation_{DATASET}.json"
    with open(sweep_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved multi-seed ablation results to {sweep_file}")
    return summary


# ============================================================================
# PART 15.7 — MISMATCHED TRAIN/EVAL BIAS FUNCTION (Reviewer #9, Point 4)
# ============================================================================
MISMATCH_EVAL_SETTINGS = [
    ("power", 0.5, "theta_eval = k^-0.5 (mismatched, mild)"),
    ("power", 0.9, "theta_eval = k^-0.9 (matched, sanity check)"),
    ("power", 1.3, "theta_eval = k^-1.3 (mismatched, severe)"),
    ("log", None, "theta_eval = 1/log2(k+1) (mismatched, different functional form)"),
]


def run_mismatched_bias_evaluation(affine_model, enc_r_vae, rank_head_vae, test_set,
                                     n_runs_ips=10):
    """
    Addresses Reviewer #9, Point 4: "The click simulator used to generate
    training data and the counterfactual evaluation metric use the identical
    examination-probability function. This risks validating disentanglement
    only against the exact bias pattern it was built to remove. Please test
    with a mismatched training and evaluation bias function to demonstrate
    genuine robustness, not pattern-matching."

    Unlike run_position_bias_sensitivity_sweep() (Reviewer #5/#6), which
    RETRAINS both methods with train and eval always matched to the same
    theta_k setting, this function trains NOTHING new: it takes the already-
    trained Affine (ground-truth) and beta-VAE models (trained once, under
    the standard theta_k = k^-0.9 training-time bias, exactly as reported in
    Table 1) and evaluates IPS-DCG on the SAME trained models under several
    DIFFERENT, deliberately mismatched theta_k_eval settings. NDCG@10 does
    not depend on click simulation and is therefore identical across all
    eval settings (reported once, for reference); only IPS-DCG, which
    depends on theta_k_eval() via ips_dcg_for_query(), varies.

    If beta-VAE's IPS-DCG advantage over Affine persists across mismatched
    eval settings, not just the matched theta=0.9 case, that is evidence the
    disentanglement mechanism generalizes rather than merely pattern-matching
    the specific bias function used to generate its training clicks.

    Restores THETA_FORM_EVAL/THETA_KAPPA_EVAL to their defaults (mirroring
    THETA_FORM/THETA_KAPPA) afterward, so no other caller of
    ips_dcg_for_query() in this script is affected by having called this
    function.
    """
    global THETA_FORM_EVAL, THETA_KAPPA_EVAL
    orig_form_eval, orig_kappa_eval = THETA_FORM_EVAL, THETA_KAPPA_EVAL

    print("\n" + "=" * 80)
    print("MISMATCHED TRAIN/EVAL BIAS FUNCTION (Reviewer #9, Point 4)")
    print(f"NOTE: Affine and beta-VAE are NOT retrained here -- both were trained once "
          f"under theta_k = k^-{THETA_KAPPA} (Section 4.6.5, matching Table 1). Only the "
          f"EVALUATION-time bias assumption (theta_k_eval, used by IPS-DCG) varies below, "
          f"decoupled entirely from what either model was trained under.")
    print("=" * 80)

    affine_score_fn = make_score_fn_simple(affine_model)
    vae_score_fn = make_score_fn_vae(enc_r_vae, rank_head_vae)

    results = []
    for form, kappa, label in MISMATCH_EVAL_SETTINGS:
        THETA_FORM_EVAL = form
        THETA_KAPPA_EVAL = kappa if kappa is not None else THETA_KAPPA_EVAL
        print(f"\n--- Eval setting: {label} ---")

        affine_ndcg, affine_ips = evaluate_ranker_fn(affine_score_fn, test_set, k=10, n_runs_ips=n_runs_ips)
        vae_ndcg, vae_ips = evaluate_ranker_fn(vae_score_fn, test_set, k=10, n_runs_ips=n_runs_ips)

        row = {
            "eval_setting": label, "theta_form_eval": form, "theta_kappa_eval": THETA_KAPPA_EVAL,
            "training_theta_kappa": THETA_KAPPA, "is_matched": (form == THETA_FORM and THETA_KAPPA_EVAL == THETA_KAPPA),
            "Affine_NDCG10": float(np.mean(affine_ndcg)), "Affine_IPSDCG": float(np.mean(affine_ips)),
            "betaVAE_NDCG10": float(np.mean(vae_ndcg)), "betaVAE_IPSDCG": float(np.mean(vae_ips)),
        }
        row["betaVAE_wins_IPSDCG"] = row["betaVAE_IPSDCG"] > row["Affine_IPSDCG"]
        results.append(row)
        print(f"  Affine:  NDCG@10={row['Affine_NDCG10']:.4f} (ref.)  IPS-DCG={row['Affine_IPSDCG']:.4f}")
        print(f"  beta-VAE: NDCG@10={row['betaVAE_NDCG10']:.4f} (ref.)  IPS-DCG={row['betaVAE_IPSDCG']:.4f}")
        print(f"  beta-VAE wins on IPS-DCG under this eval setting: {row['betaVAE_wins_IPSDCG']}")

    THETA_FORM_EVAL, THETA_KAPPA_EVAL = orig_form_eval, orig_kappa_eval  # restore defaults

    n_wins = sum(r["betaVAE_wins_IPSDCG"] for r in results)
    n_mismatched = sum(1 for r in results if not r["is_matched"])
    n_wins_mismatched = sum(r["betaVAE_wins_IPSDCG"] for r in results if not r["is_matched"])
    print("\n" + "-" * 80)
    print("MISMATCH EVALUATION SUMMARY")
    print("-" * 80)
    print(f"beta-VAE beat Affine on IPS-DCG in {n_wins}/{len(results)} eval settings overall")
    print(f"beta-VAE beat Affine on IPS-DCG in {n_wins_mismatched}/{n_mismatched} "
          f"MISMATCHED eval settings specifically (train theta_kappa={THETA_KAPPA}, "
          f"eval theta != train theta)")

    out_file = f"mismatched_bias_eval_{DATASET}_seed{SEED}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved mismatched-bias evaluation results to {out_file}")
    return results


# ============================================================================
# MAIN
# ============================================================================
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


def main():
    print(f"Loading {DATASET.upper()} dataset...")
    train_set = LTRDataset(TRAIN_FILE)
    vali_set = LTRDataset(VALI_FILE, scaler=train_set.scaler, feature_dim=train_set.feature_dim)
    test_set = LTRDataset(TEST_FILE, scaler=train_set.scaler, feature_dim=train_set.feature_dim)
    print(f"  train queries={len(train_set.queries)}  vali queries={len(vali_set.queries)}  "
          f"test queries={len(test_set.queries)}  feature_dim={train_set.feature_dim}")

    # Train the production ranker FIRST (Joachims et al. 2017 protocol): its
    # predicted scores, not true relevance, will determine the exposure
    # ranking used by _precompute_training_clicks for every method below.
    train_production_ranker(train_set)

    # global lookup used by train_beta_vae for per-query relevance labels.
    # CRITICAL: must use the SAME exposure-ranking order as
    # _precompute_training_clicks (i.e., respect PRODUCTION_RANKER if set),
    # or dataset_rels_lookup's relevance labels would misalign with the
    # documents actually present in click_cache's ranked_feats.
    global dataset_rels_lookup
    dataset_rels_lookup = {}
    click_cache_global = _precompute_training_clicks(train_set)
    for qid, (feats, clicks) in click_cache_global.items():
        idxs = train_set.queries[qid]
        rels_full = train_set.rels[idxs]
        feats_full = train_set.features[idxs]
        if PRODUCTION_RANKER is not None:
            with torch.no_grad():
                x = torch.tensor(feats_full, dtype=torch.float32, device=DEVICE)
                exposure_scores = PRODUCTION_RANKER(x).cpu().numpy()
            order = np.argsort(-exposure_scores)[:K_MAX_POSITION]
        else:
            order = np.argsort(-rels_full)[:K_MAX_POSITION]
        dataset_rels_lookup[qid] = rels_full[order]

    results = {}

    naive_model = train_naive(train_set, vali_set)
    supervised_model = train_supervised_only(train_set, vali_set)
    ips_model = train_ips(train_set, vali_set)
    em_model, theta_est = train_regression_em(train_set, vali_set)
    bayes_model = train_bayes_ips(train_set, vali_set)
    affine_model = train_affine(train_set, vali_set, estimated=False)
    affine_est_model = train_affine(train_set, vali_set, estimated=True)
    ips_est_model, theta_hat = train_ips_estimated(train_set, vali_set)
    inforank_model = train_inforank(train_set, vali_set)

    print("\n[Timing] Starting beta-VAE training (wall-clock timer for Section 4.6.8)...")
    _train_start_time = time.time()
    enc_r_vae, rank_head_vae, enc_b_vae, pos_emb_vae, training_history = train_beta_vae(
        train_set, vali_set, track_history=True)
    _train_elapsed_seconds = time.time() - _train_start_time
    _epochs_run = len(training_history["epoch"]) if training_history else None
    print(f"[Timing] beta-VAE training finished: {_train_elapsed_seconds:.1f} sec "
          f"({_train_elapsed_seconds/60:.2f} min) over {_epochs_run} epochs "
          f"(device={DEVICE}, dataset={DATASET})")

    training_curves_path = f"training_dynamics_{DATASET}.png"
    plot_training_dynamics(training_history, training_curves_path)

    disentanglement_results = run_disentanglement_metrics(enc_r_vae, enc_b_vae, pos_emb_vae, test_set)

    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)

    methods = {
        "Naive Baseline": make_score_fn_simple(naive_model),
        "Supervised-Only": make_score_fn_simple(supervised_model),
        "IPS (ground-truth)": make_score_fn_simple(ips_model),
        "Regression-EM": make_score_fn_simple(em_model),
        "Bayes-IPS": make_score_fn_simple(bayes_model),
        "Affine (ground-truth)": make_score_fn_simple(affine_model),
        "Affine (estimated)": make_score_fn_simple(affine_est_model),
        "IPS (estimated)": make_score_fn_simple(ips_est_model),
        "InfoRank": make_score_fn_inforank(inforank_model),
        "beta-VAE (Ours)": make_score_fn_vae(enc_r_vae, rank_head_vae),
    }

    per_query_scores = {}
    for name, fn in methods.items():
        ndcgs, ips_dcgs = evaluate_ranker_fn(fn, test_set, k=10, n_runs_ips=20)
        per_query_scores[name] = {"ndcg": ndcgs, "ips_dcg": ips_dcgs}
        results[name] = {
            "NDCG@10": float(np.mean(ndcgs)),
            "IPS-DCG": float(np.mean(ips_dcgs)),
        }
        print(f"{name:28s}  NDCG@10={np.mean(ndcgs):.4f}  IPS-DCG={np.mean(ips_dcgs):.4f}")

    print("\nPaired t-tests vs. Naive Baseline:")
    naive_ndcg = per_query_scores["Naive Baseline"]["ndcg"]
    naive_ipsdcg = per_query_scores["Naive Baseline"]["ips_dcg"]
    ttest_results = {}
    for name in methods:
        if name == "Naive Baseline":
            continue
        n = min(len(per_query_scores[name]["ndcg"]), len(naive_ndcg))
        r_ndcg = paired_ttest_report(name, per_query_scores[name]["ndcg"][:n], naive_ndcg[:n])
        r_ips = paired_ttest_report(name, per_query_scores[name]["ips_dcg"][:n], naive_ipsdcg[:n])
        ttest_results[name] = {"NDCG@10": r_ndcg, "IPS-DCG": r_ips}
        print(f"  {name:28s}  NDCG@10: p={r_ndcg['p_value']:.4g}   "
              f"IPS-DCG: p={r_ips['p_value']:.4g}")

    # Direct comparison: beta-VAE vs. Supervised-Only (addresses R5-1 --
    # isolates the disentanglement mechanism's contribution beyond label
    # supervision alone, since Supervised-Only uses identical label
    # supervision with zero debiasing mechanism)
    if "beta-VAE (Ours)" in per_query_scores and "Supervised-Only" in per_query_scores:
        vae_ndcg = per_query_scores["beta-VAE (Ours)"]["ndcg"]
        vae_ips = per_query_scores["beta-VAE (Ours)"]["ips_dcg"]
        sup_ndcg = per_query_scores["Supervised-Only"]["ndcg"]
        sup_ips = per_query_scores["Supervised-Only"]["ips_dcg"]
        n = min(len(vae_ndcg), len(sup_ndcg))
        r_ndcg_vae_vs_sup = paired_ttest_report("beta-VAE vs Supervised-Only",
                                                 vae_ndcg[:n], sup_ndcg[:n])
        r_ips_vae_vs_sup = paired_ttest_report("beta-VAE vs Supervised-Only",
                                                vae_ips[:n], sup_ips[:n])
        ttest_results["beta-VAE vs Supervised-Only"] = {"NDCG@10": r_ndcg_vae_vs_sup,
                                                          "IPS-DCG": r_ips_vae_vs_sup}
        print("\nDirect comparison: beta-VAE vs. Supervised-Only (R5-1 -- isolates the "
              "disentanglement mechanism's contribution beyond label supervision alone):")
        print(f"  NDCG@10:  mean_diff={r_ndcg_vae_vs_sup['mean_diff']:+.4f}  "
              f"t={r_ndcg_vae_vs_sup['t_stat']:.2f}  p={r_ndcg_vae_vs_sup['p_value']:.4g}")
        print(f"  IPS-DCG:  mean_diff={r_ips_vae_vs_sup['mean_diff']:+.4f}  "
              f"t={r_ips_vae_vs_sup['t_stat']:.2f}  p={r_ips_vae_vs_sup['p_value']:.4g}")

    # Direct comparison: beta-VAE vs. Affine (ground-truth) (Reviewer #9,
    # Point 2 -- the originally submitted manuscript reported this
    # comparison as statistically indistinguishable (p=0.926) and this was
    # read as an "equivalence" claim, which a non-significant p-value does
    # not establish; this direct paired test, absent from the reconstructed
    # pipeline until now, replaces that stale result with a real one, using
    # the current pipeline's per-query scores.
    if "beta-VAE (Ours)" in per_query_scores and "Affine (ground-truth)" in per_query_scores:
        vae_ndcg = per_query_scores["beta-VAE (Ours)"]["ndcg"]
        vae_ips = per_query_scores["beta-VAE (Ours)"]["ips_dcg"]
        aff_ndcg = per_query_scores["Affine (ground-truth)"]["ndcg"]
        aff_ips = per_query_scores["Affine (ground-truth)"]["ips_dcg"]
        n = min(len(vae_ndcg), len(aff_ndcg))
        r_ndcg_vae_vs_aff = paired_ttest_report("beta-VAE vs Affine (ground-truth)",
                                                 vae_ndcg[:n], aff_ndcg[:n])
        r_ips_vae_vs_aff = paired_ttest_report("beta-VAE vs Affine (ground-truth)",
                                                vae_ips[:n], aff_ips[:n])
        ttest_results["beta-VAE vs Affine (ground-truth)"] = {"NDCG@10": r_ndcg_vae_vs_aff,
                                                                "IPS-DCG": r_ips_vae_vs_aff}
        print("\nDirect comparison: beta-VAE vs. Affine (ground-truth) (Reviewer #9, Point 2 -- "
              "replaces the stale, originally-submitted p=0.926 result with a real, current test):")
        print(f"  NDCG@10:  mean_diff={r_ndcg_vae_vs_aff['mean_diff']:+.4f}  "
              f"t={r_ndcg_vae_vs_aff['t_stat']:.2f}  p={r_ndcg_vae_vs_aff['p_value']:.4g}")
        print(f"  IPS-DCG:  mean_diff={r_ips_vae_vs_aff['mean_diff']:+.4f}  "
              f"t={r_ips_vae_vs_aff['t_stat']:.2f}  p={r_ips_vae_vs_aff['p_value']:.4g}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"dataset": DATASET, "results": results, "ttests": ttest_results,
                    "theta_est_regression_em": theta_est,
                    "theta_hat_ips_estimated": theta_hat,
                    "disentanglement_metrics": disentanglement_results,
                    "beta_vae_training_time_seconds": _train_elapsed_seconds,
                    "beta_vae_epochs_run": _epochs_run,
                    "device": DEVICE, "seed": SEED}, f, indent=2)
    print(f"\nSaved results to {RESULTS_FILE}")
    if DATASET == "yahoo":
        print("\nSanity check against published Table 1 (Yahoo! LETOR Set 2):")
        print(f"  Published beta-VAE: NDCG@10=0.6968  IPS-DCG=13.7929")
        print(f"  This run's beta-VAE: NDCG@10={results['beta-VAE (Ours)']['NDCG@10']:.4f}  "
              f"IPS-DCG={results['beta-VAE (Ours)']['IPS-DCG']:.4f}")

    if RUN_SENSITIVITY_SWEEP:
        run_position_bias_sensitivity_sweep(train_set, vali_set, test_set)

    if RUN_HYPERPARAMETER_SWEEP:
        run_hyperparameter_sensitivity_sweep(train_set, vali_set, test_set)

    if RUN_FACTORIAL_ABLATION:
        run_factorial_ablation_sweep(train_set, vali_set, test_set)

    if RUN_MULTISEED_ABLATION:
        run_multiseed_ablation(train_set, vali_set, test_set)

    if RUN_MISMATCHED_BIAS_EVAL:
        run_mismatched_bias_evaluation(affine_model, enc_r_vae, rank_head_vae, test_set)

    if RUN_MULTI_BIAS_FACTOR:
        run_multi_bias_factor_experiment(train_set, vali_set, test_set)


if __name__ == "__main__":
    main()
