"""
Yandex control experiments (real clicks, CPU friendly)
======================================================
Answers Reviewer 1 (Comment 9) and Reviewer 5 (Comments 2 and 7): on Yandex, is the
behaviour of the proposed model specific to the full framework, or do simpler models and
trivial references give the same numbers?

Everything runs on the SAME data split, the SAME hashed-ID embedding input and the SAME
evaluation as yandex_embedding_pipeline.py.  Models:

    random_init    untrained embedding ranker (probe calibration)
    supervised     Supervised-Only: ranker trained on the dwell grade only (no clicks)
    naive_click    ranker trained on clicks only (no debiasing)
    regression_em  click-only Regression-EM (propensities estimated by EM from clicks)
    vae_full       the proposed beta-VAE (adversary on)
    vae_noadv      the beta-VAE WITHOUT the adversary
    simple_adv     simple position-adversarial ranker (no VAE, no dual encoder)

References computed without training:
    logged         the original logged order (the "Naive" row of the paper)
    random (exact) expected CTR@1, CTR@5, MRR of a uniformly random ordering, computed
                   exactly per SERP (hypergeometric), plus an empirical random scorer
                   (PB-rho of a random permutation is about 0)

Metrics (as in the pipeline): PB-rho (Spearman of score vs logged position, all test rows),
CTR@1, CTR@5, MRR over test SERPs with at least one click, helped / hurt / unchanged counts,
and the frozen-representation position probe (Logistic Regression and Random Forest, chance
0.10).  The first-clicked rank of every SERP is saved so that paired tests can be run.

HOW TO RUN (Anaconda Prompt; put this file next to yandex_embedding_pipeline.py and
yandex_subsample.csv):

    set KMP_DUPLICATE_LIB_OK=TRUE
    python yandex_controls.py --time-check          (1 minute: tells you the total time)
    python yandex_controls.py                       (primary setting: 4000 sessions, 15 epochs)

    python yandex_controls.py --max-sessions 10000 --epochs 40     (extended setting, much slower)

Resumable: finished models are skipped on restart.
Output: results_yandex_controls_seed<seed>.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import math
import re
import time

ALL_MODELS = ["random_init", "supervised", "naive_click", "regression_em",
              "vae_full", "vae_noadv", "simple_adv"]

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--pipeline", default=os.path.join(here, "yandex_embedding_pipeline.py"))
ap.add_argument("--csv", default=os.path.join(here, "yandex_subsample.csv"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--max-sessions", type=int, default=4000)
ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--time-check", action="store_true")
ap.add_argument("--redo", nargs="+", default=[], help="discard saved results of these models and retrain them")
args = ap.parse_args()

with open(args.pipeline, "r", encoding="utf-8") as f:
    src = f.read()


def sub_once(pattern, repl, text):
    new, n = re.subn(pattern, lambda m: repl, text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"Could not patch pipeline pattern: {pattern}")
    return new


src = sub_once(r'^INPUT_CSV\s*=.*$', f'INPUT_CSV = {args.csv!r}', src)
src = sub_once(r'^SEED\s*=\s*\d+', f'SEED = {args.seed}', src)
src = sub_once(r'^EPOCHS\s*=\s*\d+', f'EPOCHS = {args.epochs}', src)
src = sub_once(r'^MAX_SESSIONS_TO_USE\s*=\s*\w+', f'MAX_SESSIONS_TO_USE = {args.max_sessions}', src)

# switch for the adversary (used for the "vae_noadv" control)
old_sched = "lambda_adv = 0.1 + (lambda_adv_max - 0.1) * (t / max(epochs - 1, 1))"
if src.count(old_sched) != 1:
    raise RuntimeError("adversarial schedule line not found exactly once")
src = src.replace(old_sched, old_sched + "\n        if not ADV_ENABLED:\n            lambda_adv = 0.0")
old_loop = "            for _ in range(3):\n                d_logits = disc(zr_detached)"
if src.count(old_loop) != 1:
    raise RuntimeError("discriminator loop not found exactly once")
src = src.replace(old_loop, "            for _ in range(3 if ADV_ENABLED else 0):\n                d_logits = disc(zr_detached)")
src += "\nADV_ENABLED = True\n"

ns = {"__name__": "yandex_controls_pipeline", "__file__": args.pipeline}
exec(compile(src, args.pipeline, "exec"), ns)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    n_threads = args.threads if args.threads > 0 else max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(n_threads)
except Exception:
    n_threads = None
DEVICE = ns["DEVICE"]
K = ns["K_MAX_POSITION"]
SEED = args.seed
print(f"Device: {DEVICE} | CPU threads: {n_threads} | seed: {SEED} | sessions: {args.max_sessions} | epochs: {args.epochs}")


def reseed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------- data (as in main())
reseed(SEED)
serps = ns["load_yandex_csv"](args.csv)
print(f"  {len(serps)} SERPs loaded")
all_sessions = sorted(set(s["session_id"] for s in serps))
if args.max_sessions and len(all_sessions) > args.max_sessions:
    rng = random.Random(SEED)
    rng.shuffle(all_sessions)
    keep = set(all_sessions[:args.max_sessions])
    serps = [s for s in serps if s["session_id"] in keep]
train_serps, test_serps = ns["split_by_session"](serps)
print(f"  train SERPs={len(train_serps)}  test SERPs={len(test_serps)}")
n_click_serps = sum(1 for s in test_serps if s["clicks"].sum() > 0)
print(f"  test SERPs with at least one click: {n_click_serps}")


# ---------------------------------------------------------------- baseline networks
def make_x(inp, serp):
    n = len(serp["positions"])
    return inp(np.full(n, serp["query_id"]), serp["url_ids"], serp["domain_ids"])


class EmbRanker(nn.Module):
    """hashed-ID embeddings -> MLP (same trunk as the relevance encoder) -> score."""
    def __init__(self):
        super().__init__()
        self.inp = ns["YandexInputEncoder"]()
        d = self.inp.out_dim
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(96, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)

    def forward(self, serp):
        h = self.net(make_x(self.inp, serp))
        return self.head(h).squeeze(-1), h


class SimpleAdvNet(nn.Module):
    """single deterministic encoder (24-d code) + ranking head, no decoder, no KL."""
    def __init__(self):
        super().__init__()
        self.inp = ns["YandexInputEncoder"]()
        d = self.inp.out_dim
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(96, 64), nn.ReLU())
        self.proj = nn.Linear(64, 24)
        self.head = ns["RankingHead"]()

    def forward(self, serp):
        z = self.proj(self.net(make_x(self.inp, serp)))
        return self.head(z), z


def clip_step(opt, params, loss):
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()


def rank_loss(score, serp):
    grades = serp["grades"]
    y = torch.tensor(grades, dtype=torch.float32, device=DEVICE)
    return F.mse_loss(score, y) + 0.5 * ns["pairwise_hinge_loss"](score, grades)


def fit(model, epochs, loss_fn, tag):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    params = list(model.parameters())
    for ep in range(epochs):
        model.train(); random.shuffle(train_serps); t0 = time.time(); tot = 0.0
        for serp in train_serps:
            loss = loss_fn(serp)
            clip_step(opt, params, loss); tot += float(loss.item())
        print(f"  [{tag}] epoch {ep+1}/{epochs}  loss={tot/max(len(train_serps),1):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    return model


def train_supervised():
    print("\n[Supervised-Only on the dwell grade] Training (no clicks)...")
    m = EmbRanker().to(DEVICE)
    return fit(m, args.epochs, lambda serp: rank_loss(m(serp)[0], serp), "supervised")


def train_naive_click():
    print("\n[Naive click ranker] Training (clicks only, no debiasing)...")
    m = EmbRanker().to(DEVICE)
    def loss_fn(serp):
        c = torch.tensor(serp["clicks"], dtype=torch.float32, device=DEVICE)
        return F.binary_cross_entropy_with_logits(m(serp)[0], c)
    return fit(m, args.epochs, loss_fn, "naive_click")


def train_regression_em(em_iterations=5, ema_alpha=0.5):
    print("\n[Regression-EM] Training (clicks only; propensities estimated by EM)...")
    m = EmbRanker().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-5)
    params = list(m.parameters())
    # initial propensities: observable click rate by position, normalised to position 0
    cs, ns_ = defaultdict(float), defaultdict(float)
    for s in train_serps:
        for p, c in zip(s["positions"], s["clicks"]):
            cs[int(p)] += c; ns_[int(p)] += 1
    ctr0 = max(cs[0] / max(ns_[0], 1), 1e-3)
    theta = {k: float(np.clip((cs[k] / max(ns_[k], 1)) / ctr0, 1e-3, 1.0)) for k in range(K)}
    per_em = max(1, args.epochs // em_iterations)
    for it in range(em_iterations):
        for ep in range(per_em):
            m.train(); random.shuffle(train_serps); t0 = time.time()
            for serp in train_serps:
                c = torch.tensor(serp["clicks"], dtype=torch.float32, device=DEVICE)
                # click likelihood under the position based model: P(click) = theta_k * rel(d),
                # fitted on ALL shown documents (clicked and not clicked).  An earlier version
                # used clicked documents only, which is satisfied by a constant saturated score.
                th = torch.tensor([theta[int(p)] for p in serp["positions"]],
                                  dtype=torch.float32, device=DEVICE)
                rel = torch.sigmoid(m(serp)[0])
                prob = torch.clamp(th * rel, 1e-6, 1.0 - 1e-6)
                loss = F.binary_cross_entropy(prob, c)
                clip_step(opt, params, loss)
            print(f"  [regression_em] EM iter {it+1}/{em_iterations} epoch {ep+1}/{per_em} ({time.time()-t0:.0f}s)", flush=True)
        m.eval()
        click_sum, rel_sum = defaultdict(float), defaultdict(float)
        with torch.no_grad():
            for serp in train_serps:
                pr = torch.sigmoid(m(serp)[0]).cpu().numpy()
                for p, c, r in zip(serp["positions"], serp["clicks"], pr):
                    click_sum[int(p)] += c; rel_sum[int(p)] += r
        for k in range(K):
            if rel_sum[k] > 1e-6:
                new = float(np.clip(click_sum[k] / rel_sum[k], 1e-3, 1.0))
                theta[k] = ema_alpha * new + (1 - ema_alpha) * theta[k]
        t0_ = max(theta[0], 1e-6)          # fix the scale: theta_0 = 1 (propensity and relevance are only
        for k in range(K):                 # identified up to a common factor)
            theta[k] = float(np.clip(theta[k] / t0_, 1e-3, 1.0))
        print(f"  [regression_em] theta_est = { {k: round(v, 3) for k, v in theta.items()} }")
    m.eval()
    return m, theta


def train_simple_adv(lambda_rank=2.0, lambda_adv_max=1.0):
    print("\n[Simple position-adversarial ranker] Training (no VAE, no dual encoder)...")
    m = SimpleAdvNet().to(DEVICE)
    disc = ns["Discriminator"]().to(DEVICE)
    gen = list(m.parameters())
    opt_g = torch.optim.Adam(gen, lr=3e-4, weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=6e-4, weight_decay=1e-5)
    for ep in range(args.epochs):
        lam = 0.1 + (lambda_adv_max - 0.1) * (ep / max(args.epochs - 1, 1))
        m.train(); disc.train(); random.shuffle(train_serps); t0 = time.time(); tot = 0.0
        for serp in train_serps:
            y_pos = torch.tensor(serp["positions"] >= (K // 2), dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                z_det = m(serp)[1]
            for _ in range(3):
                d_loss = F.cross_entropy(disc(z_det), y_pos)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            score, z = m(serp)
            l_adv = F.cross_entropy(disc(ns["grad_reverse"](z, lam)), y_pos)
            loss = lambda_rank * rank_loss(score, serp) + lam * l_adv
            clip_step(opt_g, gen, loss); tot += float(loss.item())
        print(f"  [simple_adv] epoch {ep+1}/{args.epochs}  loss={tot/max(len(train_serps),1):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    m.eval(); disc.eval()
    return m


# ---------------------------------------------------------------- evaluation
def probe(reps, pos):
    Xtr, Xte, ytr, yte = train_test_split(reps, pos, test_size=0.3, random_state=SEED, stratify=pos)
    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    return float(accuracy_score(yte, lr.predict(Xte))), float(accuracy_score(yte, rf.predict(Xte)))


orig_first = []        # first clicked rank in the logged order, per SERP with a click
for s in test_serps:
    if s["clicks"].sum() > 0:
        o = np.argsort(s["positions"])
        orig_first.append(int(np.where(s["clicks"][o] == 1)[0].min()))
orig_first = np.array(orig_first)


def evaluate(score_fn, rep_fn=None):
    scores, pos, first = [], [], []
    reps = []
    with torch.no_grad():
        for s in test_serps:
            sc = np.asarray(score_fn(s), dtype=float)
            scores.extend(sc.tolist()); pos.extend(s["positions"].tolist())
            if rep_fn is not None:
                reps.append(rep_fn(s))
            if s["clicks"].sum() > 0:
                order = np.argsort(-sc)
                first.append(int(np.where(s["clicks"][order] == 1)[0].min()))
    first = np.array(first)
    rho, rho_p = spearmanr(scores, pos)
    out = {"pb_rho": float(rho), "pb_rho_p": float(rho_p),
           "ctr_at_1": float(np.mean(first < 1)), "ctr_at_5": float(np.mean(first < 5)),
           "mrr": float(np.mean(1.0 / (first + 1))),
           "n_helped": int(np.sum(first < orig_first)), "n_hurt": int(np.sum(first > orig_first)),
           "n_unchanged": int(np.sum(first == orig_first)), "n_serps_with_clicks": int(len(first)),
           "first_click_rank": first.tolist()}
    if rep_fn is not None:
        lr_acc, rf_acc = probe(np.concatenate(reps, axis=0), np.array(pos))
        out["probe_lr"], out["probe_rf"] = lr_acc, rf_acc
    return out


def random_expectation():
    c1, c5, mrr = [], [], []
    for s in test_serps:
        m = int(s["clicks"].sum())
        if m == 0:
            continue
        n = len(s["clicks"])
        c1.append(m / n)
        c5.append(1.0 if (n < 5 or n - m < 5) else 1.0 - math.comb(n - m, 5) / math.comb(n, 5))
        mrr.append(sum((1.0 / r) * math.comb(n - r, m - 1) / math.comb(n, m) for r in range(1, n - m + 2)))
    return {"ctr_at_1": float(np.mean(c1)), "ctr_at_5": float(np.mean(c5)), "mrr": float(np.mean(mrr))}


out_path = f"results_yandex_controls_seed{SEED}.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}

# ---------------------------------------------------------------- references
if "_references" not in results:
    rg = np.random.RandomState(SEED + 1)
    emp = evaluate(lambda s: rg.randn(len(s["positions"])))
    results["_references"] = {
        "logged": {"ctr_at_1": float(np.mean(orig_first < 1)), "ctr_at_5": float(np.mean(orig_first < 5)),
                   "mrr": float(np.mean(1.0 / (orig_first + 1))), "pb_rho": 1.0,
                   "n_serps_with_clicks": int(len(orig_first))},
        "random_exact": random_expectation(),
        "random_empirical": {k: emp[k] for k in ("pb_rho", "ctr_at_1", "ctr_at_5", "mrr")},
        "n_train_serps": len(train_serps), "n_test_serps": len(test_serps),
        "orig_first_click_rank": orig_first.tolist(),
    }
    json.dump(results, open(out_path, "w"))
ref = results["_references"]
print("\nReferences:")
print(f"  logged order : CTR@1={ref['logged']['ctr_at_1']:.4f} CTR@5={ref['logged']['ctr_at_5']:.4f} MRR={ref['logged']['mrr']:.4f}")
print(f"  random (exact): CTR@1={ref['random_exact']['ctr_at_1']:.4f} CTR@5={ref['random_exact']['ctr_at_5']:.4f} MRR={ref['random_exact']['mrr']:.4f}")
print(f"  random scorer : PB-rho={ref['random_empirical']['pb_rho']:.4f}")

# ---------------------------------------------------------------- time check
if args.time_check:
    reseed(SEED)
    sub = train_serps[:300]
    t0 = time.time()
    ns["ADV_ENABLED"] = True
    ns["train_yandex_beta_vae"](sub, epochs=1)
    per = (time.time() - t0) / len(sub)
    est = per * len(train_serps) * args.epochs / 60
    print(f"\nTIME CHECK: {per*1000:.1f} ms per SERP for the beta-VAE.")
    print(f"  Estimated beta-VAE time for {args.epochs} epochs on {len(train_serps)} SERPs: about {est:.0f} minutes.")
    print(f"  Whole script (about 4 beta-VAE-sized trainings plus lighter baselines): about {est*3.2/60:.1f} hours.")
    raise SystemExit(0)

# ---------------------------------------------------------------- models
def vae_fns(models):
    input_enc, enc_r, enc_b, dec, rank_head, pos_emb = models
    for mod in (input_enc, enc_r, enc_b, dec, rank_head):
        mod.eval()
    def score(s):
        _, mu, _ = enc_r(make_x(input_enc, s)); return rank_head(mu).cpu().numpy()
    def rep(s):
        _, mu, _ = enc_r(make_x(input_enc, s)); return mu.cpu().numpy()
    return score, rep


def run_model(key):
    reseed(SEED)
    t0 = time.time()
    if key == "random_init":
        m = EmbRanker().to(DEVICE).eval()
        f_score = lambda s: m(s)[0].cpu().numpy(); f_rep = lambda s: m(s)[1].cpu().numpy()
    elif key == "supervised":
        m = train_supervised(); f_score = lambda s: m(s)[0].cpu().numpy(); f_rep = lambda s: m(s)[1].cpu().numpy()
    elif key == "naive_click":
        m = train_naive_click(); f_score = lambda s: m(s)[0].cpu().numpy(); f_rep = lambda s: m(s)[1].cpu().numpy()
    elif key == "regression_em":
        m, _ = train_regression_em(); f_score = lambda s: m(s)[0].cpu().numpy(); f_rep = lambda s: m(s)[1].cpu().numpy()
    elif key == "simple_adv":
        m = train_simple_adv(); f_score = lambda s: m(s)[0].cpu().numpy(); f_rep = lambda s: m(s)[1].cpu().numpy()
    else:
        ns["ADV_ENABLED"] = (key == "vae_full")
        models = ns["train_yandex_beta_vae"](train_serps, epochs=args.epochs)
        ns["ADV_ENABLED"] = True
        f_score, f_rep = vae_fns(models)
    res = evaluate(f_score, f_rep)
    res["train_seconds"] = time.time() - t0
    return res


for key in args.redo:
    results.pop(key, None)
for key in args.models:
    if key in results:
        print(f"\n[{key}] already finished, skipping"); continue
    print("\n" + "=" * 78 + f"\nMODEL: {key}\n" + "=" * 78)
    res = run_model(key)
    results[key] = res
    json.dump(results, open(out_path, "w"))
    print(f"\n  [{key}] PB-rho={res['pb_rho']:.4f}  CTR@1={res['ctr_at_1']:.4f}  CTR@5={res['ctr_at_5']:.4f}  "
          f"MRR={res['mrr']:.4f}  probe LR/RF={res['probe_lr']:.3f}/{res['probe_rf']:.3f}  "
          f"helped/hurt/same={res['n_helped']}/{res['n_hurt']}/{res['n_unchanged']}")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("SUMMARY (Yandex, test SERPs with a click; probe chance = 0.10)")
print("=" * 78)
print(f"{'model':<14}{'PB-rho':>8}{'probeLR':>9}{'probeRF':>9}{'CTR@1':>8}{'CTR@5':>8}{'MRR':>8}{'help':>6}{'hurt':>6}{'same':>6}")
L = ref["logged"]; R = ref["random_exact"]; E = ref["random_empirical"]
print(f"{'logged order':<14}{1.0:>8.3f}{'-':>9}{'-':>9}{L['ctr_at_1']:>8.4f}{L['ctr_at_5']:>8.4f}{L['mrr']:>8.4f}")
print(f"{'random (exact)':<14}{E['pb_rho']:>8.3f}{'-':>9}{'-':>9}{R['ctr_at_1']:>8.4f}{R['ctr_at_5']:>8.4f}{R['mrr']:>8.4f}")
for k in ALL_MODELS:
    if k in results:
        v = results[k]
        print(f"{k:<14}{v['pb_rho']:>8.3f}{v['probe_lr']:>9.3f}{v['probe_rf']:>9.3f}{v['ctr_at_1']:>8.4f}"
              f"{v['ctr_at_5']:>8.4f}{v['mrr']:>8.4f}{v['n_helped']:>6}{v['n_hurt']:>6}{v['n_unchanged']:>6}")
print(f"\nSaved: {out_path}")
