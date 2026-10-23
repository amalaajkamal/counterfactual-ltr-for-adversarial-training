"""
Biased training targets experiment (Yahoo! LETOR Set 2, CPU friendly)
=====================================================================
Reviewer 5 (Comments 1 and 7) and Reviewer 1 (Comment 3) point out that, with clean
human labels, there is no position bias for the adversarial mechanism to remove, so no
setting exists in which the architecture could show its value.  This script creates that
setting.

Idea
----
In many real systems the "labels" are derived from user behaviour (dwell time, clicks) and
therefore inherit position bias.  We mimic this: the graded labels used for TRAINING (and
for validation / model selection) are inflated for documents that the production ranker
placed near the top of the list:

    y_obs = clip( stochastic_round( y + s * 2 * (1 - pos / K) ), 0, 4 )     for pos < K

where pos = 0..K-1 is the exposure position (K = 10) and s is the bias strength
(s = 0 is the clean case; s = 1 inflates the top position by up to 2 grades).
The TEST labels stay clean, so test NDCG@10 measures real relevance.
Clicks are still simulated from the TRUE relevance (unchanged), so the click signal is
unbiased in the sense of the click model, while the labels are biased.

NOTE: the production ranker is a function of the document features, so this bias is
partly learnable from x alone.  That is exactly the point: a ranker that fits the biased
labels reproduces the production order, and only a mechanism that removes position
information from the relevance representation can resist it.

Models compared (identical data, identical budget, identical seed):
    supervised   Supervised Only on the biased labels (all candidate documents of each query)
    supervised_topk  Supervised Only on the SAME top-K exposed documents the VAE-type models see
                 (control: the VAE-type models only train on the exposed documents)
    vae_full     full beta-VAE (adversary on), biased labels + clicks
    vae_noadv    beta-VAE without the adversary
    simple_adv   simple position-adversarial ranker
    simple_noadv the SAME simple ranker with the adversary switched off (isolates the adversary)

Output per model: test NDCG@10 on CLEAN labels, std, and the per-query NDCG array
(for paired tests).  Results go to results_biased_labels_yahoo_seed<seed>.json.

HOW TO RUN (Anaconda Prompt, same folder as beta_vae_unified_pipeline.py):

    set KMP_DUPLICATE_LIB_OK=TRUE
    python biased_label_experiment.py --strengths 1.0
    python biased_label_experiment.py --strengths 2.0        (optional, stronger bias)

Each strength takes about 1 to 1.5 hours (four models, 30 epochs).  Resumable.
Quick test:  python biased_label_experiment.py --models supervised --strengths 1.0
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import re
import time

ALL_MODELS = ["supervised", "supervised_topk", "vae_full", "vae_noadv", "simple_adv", "simple_noadv"]

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "beta_vae_unified_pipeline.py"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--strengths", nargs="+", type=float, default=[1.0])
ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
ap.add_argument("--include-clean", action="store_true",
                help="also run the clean reference (strength 0) for the same models")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--train", default=None)
ap.add_argument("--vali", default=None)
ap.add_argument("--test", default=None)
args = ap.parse_args()

with open(args.pipeline, "r", encoding="utf-8") as f:
    src = f.read()


def sub_once(pattern, repl, text):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"Could not patch pipeline pattern: {pattern}")
    return new


src = sub_once(r'^SEED\s*=\s*\d+', f'SEED = {args.seed}', src)
src = sub_once(r'^DATASET\s*=\s*"[a-z]+"', 'DATASET = "yahoo"', src)
if args.train:
    src = sub_once(r'^YAHOO_TRAIN_FILE\s*=.*$', f'YAHOO_TRAIN_FILE = r"{args.train}"', src)
if args.vali:
    src = sub_once(r'^YAHOO_VALI_FILE\s*=.*$', f'YAHOO_VALI_FILE = r"{args.vali}"', src)
if args.test:
    src = sub_once(r'^YAHOO_TEST_FILE\s*=.*$', f'YAHOO_TEST_FILE = r"{args.test}"', src)
src = src.replace(', multi_class="multinomial"', "")
_m = re.search(r"def train_simple_adversarial_ranker\(.*?(?=\n# =+\n)", src, re.S)
if _m is None:
    raise RuntimeError("train_simple_adversarial_ranker not found")
_fn = _m.group(0)
_sig = "pairwise_loss_enabled=True, click_cache_override=None):"
_lam = "lambda_adv = 0.1 + (lambda_adv_max - 0.1) * (t / max(epochs - 1, 1))"
_loop = "            for _ in range(3):"
for _s in (_sig, _lam, _loop):
    if _fn.count(_s) != 1:
        raise RuntimeError("cannot patch simple adversarial trainer: " + _s)
_fn2 = (_fn.replace(_sig, "pairwise_loss_enabled=True, click_cache_override=None, adv_enabled=True):")
          .replace(_lam, _lam + "\n        if not adv_enabled:\n            lambda_adv = 0.0")
          .replace(_loop, "            for _ in range(3 if adv_enabled else 0):"))
src = src.replace(_fn, _fn2)

ns = {"__name__": "pipeline_biased", "__file__": args.pipeline}
exec(compile(src, args.pipeline, "exec"), ns)

import random
import numpy as np
import torch

try:
    n_threads = args.threads if args.threads > 0 else max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(n_threads)
except Exception:
    n_threads = None
DEVICE = ns["DEVICE"]
K = ns["K_MAX_POSITION"]
print(f"Device: {DEVICE} | CPU threads: {n_threads} | seed: {args.seed}")


def reseed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ------------------------------------------------------------------------- data
reseed(args.seed)
train_set = ns["LTRDataset"](ns["TRAIN_FILE"])
vali_set = ns["LTRDataset"](ns["VALI_FILE"], scaler=train_set.scaler, feature_dim=train_set.feature_dim)
test_set = ns["LTRDataset"](ns["TEST_FILE"], scaler=train_set.scaler, feature_dim=train_set.feature_dim)
print(f"  train={len(train_set.queries)} vali={len(vali_set.queries)} test={len(test_set.queries)} "
      f"features={train_set.feature_dim}")
ns["train_production_ranker"](train_set)
# clicks are simulated from the TRUE relevance, before any label is touched
click_cache = ns["_precompute_training_clicks"](train_set)

true_train_rels = train_set.rels.copy()
true_vali_rels = vali_set.rels.copy()
max_grade = int(max(true_train_rels.max(), 4))


def exposure_positions(dataset, qid):
    """exposure position (0..K-1, or -1 outside the top K) of every doc of a query."""
    idxs = dataset.queries[qid]
    feats = dataset.features[idxs]
    with torch.no_grad():
        x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
        sc = ns["PRODUCTION_RANKER"](x).cpu().numpy()
    order = np.argsort(-sc)
    pos = -np.ones(len(idxs), dtype=int)
    for p, d in enumerate(order[:K]):
        pos[d] = p
    return pos


def apply_bias(dataset, true_rels, strength, rng):
    """overwrite dataset.rels with position biased labels; return mean absolute change."""
    new = true_rels.copy().astype(float)
    changed, total = 0.0, 0
    for qid, idxs in dataset.queries.items():
        pos = exposure_positions(dataset, qid)
        top = pos >= 0
        boost = np.zeros(len(idxs))
        boost[top] = strength * 2.0 * (1.0 - pos[top] / K)
        val = true_rels[idxs].astype(float) + boost
        fl = np.floor(val)
        val = fl + (rng.rand(len(val)) < (val - fl))          # stochastic rounding
        val = np.clip(val, 0, max_grade)
        new[idxs] = val
        changed += np.abs(val - true_rels[idxs]).sum(); total += len(idxs)
    dataset.rels = new.astype(true_rels.dtype)
    return changed / max(total, 1)


def rebuild_lookup():
    lookup = {}
    for qid in click_cache:
        idxs = train_set.queries[qid]
        rels_full = train_set.rels[idxs]
        feats_full = train_set.features[idxs]
        with torch.no_grad():
            x = torch.tensor(feats_full, dtype=torch.float32, device=DEVICE)
            sc = ns["PRODUCTION_RANKER"](x).cpu().numpy()
        lookup[qid] = rels_full[np.argsort(-sc)[:K]]
    ns["dataset_rels_lookup"] = lookup


def eval_clean(score_fn):
    """test NDCG@10 on the CLEAN test labels (test_set.rels is never modified)."""
    nd, _ = ns["evaluate_ranker_fn"](score_fn, test_set, k=10, n_runs_ips=1)
    nd = np.asarray(nd, dtype=float)
    return float(nd.mean()), float(nd.std()), nd.tolist()


out_path = f"results_biased_labels_yahoo_seed{args.seed}.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}

strengths = ([0.0] if args.include_clean else []) + [s for s in args.strengths if s != 0.0]
for s in strengths:
    tag = f"s{s:g}"
    todo = [m for m in args.models if f"{tag}_{m}" not in results]
    if not todo:
        print(f"\n[{tag}] all requested models finished, skipping"); continue
    # restore clean labels, then bias them for this strength
    train_set.rels = true_train_rels.copy(); vali_set.rels = true_vali_rels.copy()
    if s > 0:
        rng = np.random.RandomState(args.seed + int(1000 * s))
        d_tr = apply_bias(train_set, true_train_rels, s, rng)
        d_va = apply_bias(vali_set, true_vali_rels, s, rng)
        print(f"\n### bias strength s={s:g}: mean |label change| train={d_tr:.3f} vali={d_va:.3f} grades/doc")
    else:
        print("\n### clean reference (s=0)")
    rebuild_lookup()

    for key in todo:
        print("\n" + "=" * 78 + f"\n{tag}  MODEL: {key}\n" + "=" * 78)
        reseed(args.seed)
        t0 = time.time()
        if key == "supervised":
            model = ns["train_supervised_only"](train_set, vali_set); model.eval()
            sf = ns["make_score_fn_simple"](model)
        elif key == "supervised_topk":
            from types import SimpleNamespace
            tq = {}
            for qid in click_cache:
                idxs = np.asarray(train_set.queries[qid])
                with torch.no_grad():
                    x = torch.tensor(train_set.features[idxs], dtype=torch.float32, device=DEVICE)
                    sc = ns["PRODUCTION_RANKER"](x).cpu().numpy()
                tq[qid] = idxs[np.argsort(-sc)[:K]]
            sub = SimpleNamespace(queries=tq, features=train_set.features, rels=train_set.rels,
                                  feature_dim=train_set.feature_dim)
            model = ns["train_supervised_only"](sub, vali_set); model.eval()
            sf = ns["make_score_fn_simple"](model)
        elif key in ("vae_full", "vae_noadv"):
            kw = {} if key == "vae_full" else {"adv_enabled": False}
            enc_r, head, _, _, _ = ns["train_beta_vae"](train_set, vali_set, epochs=args.epochs,
                                                         click_cache_override=click_cache, **kw)
            enc_r.eval(); sf = ns["make_score_fn_vae"](enc_r, head)
        else:
            enc, head = ns["train_simple_adversarial_ranker"](train_set, vali_set, epochs=args.epochs,
                                                              click_cache_override=click_cache,
                                                              adv_enabled=(key == "simple_adv"))
            enc.eval(); sf = ns["make_score_fn_simple_adversarial"](enc, head)
        m, sd, per_q = eval_clean(sf)
        results[f"{tag}_{key}"] = {"strength": s, "model": key, "test_NDCG@10_clean": m,
                                   "per_query_std": sd, "per_query_ndcg": per_q,
                                   "train_seconds": time.time() - t0}
        json.dump(results, open(out_path, "w"))
        print(f"\n  [{tag} {key}] clean-test NDCG@10 = {m:.4f}  (per-query SD {sd:.3f})")

# restore clean labels (not needed afterwards, kept for tidiness)
train_set.rels = true_train_rels; vali_set.rels = true_vali_rels

print("\n" + "=" * 78)
print("SUMMARY: clean-test NDCG@10 under biased training labels")
print("=" * 78)
tags = sorted({v["strength"] for v in results.values()})
print(f"{'model':<16}" + "".join(f"{('s=%g' % t):>10}" for t in tags))
for mkey in ALL_MODELS:
    row = []
    for t in tags:
        v = results.get(f"s{t:g}_{mkey}")
        row.append(f"{v['test_NDCG@10_clean']:>10.4f}" if v else f"{'-':>10}")
    if any(r.strip() != "-" for r in row):
        print(f"{mkey:<16}" + "".join(row))
print(f"\nSaved: {out_path}")
