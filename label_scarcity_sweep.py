"""
Label-scarcity sweep for the adversarial beta-VAE (Yahoo! LETOR Set 2, CPU friendly)
====================================================================================
Answers Reviewer 5 (Comment 1) and Reviewer 6 (Comment 1): does the click-based
reconstruction + disentanglement objective help when relevance labels exist for only
a FRACTION of the training queries (clicks are available for ALL training queries)?

For every label fraction f in LABEL_FRACTIONS the script trains and evaluates:
  * Supervised-Only  : labels of the labelled queries only, no clicks, no debiasing
  * beta-VAE (full)  : ranking loss ONLY on the labelled queries, click reconstruction,
                       adversarial and all other losses on ALL training queries
  * beta-VAE w/o adversarial training (optional, RUN_NO_ADVERSARIAL)

It reuses your existing pipeline file (beta_vae_unified_pipeline.py) WITHOUT modifying
it: the pipeline source is loaded, and train_beta_vae is patched in memory so the
ranking term is switched off for unlabelled queries.

HOW TO RUN (Jupyter notebook or terminal, put this file next to
beta_vae_unified_pipeline.py):

    # 1) timing check (about one epoch, tells you how long the sweep will take)
    !python label_scarcity_sweep.py --time-check

    # 2) the sweep (resumable: finished runs are skipped if you stop and restart)
    !python label_scarcity_sweep.py

Optional arguments:
    --pipeline PATH     path to beta_vae_unified_pipeline.py (default: same folder)
    --seed 42           random seed (default 42, matches Table 2)
    --fractions 0.05 0.25    label fractions (default 0.05 0.25)
    --epochs 30         beta-VAE epoch budget (default 30, as in the ablation tables)
    --threads 4         CPU threads for PyTorch (default: half of your cores)
    --no-adv            also train the beta-VAE without adversarial training
    --include-full      also run 100% labels as a reference point (slow)
    --train/--vali/--test FILE   override the Yahoo! file paths

Output: results_label_scarcity_yahoo_seed<seed>.json  (written after every run)
"""

import os
# Anaconda on Windows often loads two OpenMP runtimes; this lets PyTorch start anyway.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import re
import sys
import time
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "beta_vae_unified_pipeline.py"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.25])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--no-adv", action="store_true")
ap.add_argument("--include-full", action="store_true")
ap.add_argument("--time-check", action="store_true")
ap.add_argument("--train", default=None)
ap.add_argument("--vali", default=None)
ap.add_argument("--test", default=None)
ap.add_argument("--min-vali-queries", type=int, default=100)
args = ap.parse_args()

# ---------------------------------------------------------------------------
# 1. Load the pipeline source into its own namespace (main() is NOT executed)
# ---------------------------------------------------------------------------
with open(args.pipeline, "r", encoding="utf-8") as f:
    src = f.read()


def sub_once(pattern, repl, text, flags=re.M):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
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

# Patch train_beta_vae: add labelled_qids; the ranking term is dropped for unlabelled queries
m = re.search(r"def train_beta_vae\(.*?(?=\n# =+\n)", src, re.S)
if not m:
    raise RuntimeError("train_beta_vae not found in pipeline")
fn_src = m.group(0)
fn_new = fn_src.replace("track_history=False, click_cache_override=None):",
                        "track_history=False, click_cache_override=None, labelled_qids=None):", 1)
old_line = "            loss = (l_rec + lambda_rank * l_rank + lambda_r * l_kl_r +"
new_line = ("            rank_w = lambda_rank if (labelled_qids is None or qid in labelled_qids) else 0.0\n"
            "            loss = (l_rec + rank_w * l_rank + lambda_r * l_kl_r +")
if fn_new.count(old_line) != 1 or "labelled_qids=None" not in fn_new:
    raise RuntimeError("Could not patch train_beta_vae (pipeline file changed?)")
fn_new = fn_new.replace(old_line, new_line, 1)
src = src.replace(fn_src, fn_new, 1)
# the fixed early-stopping epoch count must not be lowered below 0 etc.; nothing else changed

ns = {"__name__": "pipeline_patched", "__file__": args.pipeline}
exec(compile(src, args.pipeline, "exec"), ns)

import numpy as np
import torch
import random
from scipy.stats import ttest_rel

try:
    n_threads = args.threads if args.threads > 0 else max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(n_threads)
except Exception:
    n_threads = None
print(f"Device: {ns['DEVICE']} | CPU threads: {n_threads} | seed: {args.seed}")


def reseed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------------------
# 2. Data, production ranker, relevance lookup (same steps as main())
# ---------------------------------------------------------------------------
reseed(args.seed)
LTRDataset = ns["LTRDataset"]
print("Loading Yahoo! LETOR Set 2 ...")
train_set = LTRDataset(ns["TRAIN_FILE"])
vali_full = LTRDataset(ns["VALI_FILE"], scaler=train_set.scaler, feature_dim=train_set.feature_dim)
test_set = LTRDataset(ns["TEST_FILE"], scaler=train_set.scaler, feature_dim=train_set.feature_dim)
print(f"  train queries={len(train_set.queries)}  vali={len(vali_full.queries)}  "
      f"test={len(test_set.queries)}  features={train_set.feature_dim}")

ns["train_production_ranker"](train_set)
K = ns["K_MAX_POSITION"]
click_cache = ns["_precompute_training_clicks"](train_set)   # simulated clicks, ALL training queries
lookup = {}
for qid in click_cache:
    idxs = train_set.queries[qid]
    rels_full = train_set.rels[idxs]
    feats_full = train_set.features[idxs]
    with torch.no_grad():
        x = torch.tensor(feats_full, dtype=torch.float32, device=ns["DEVICE"])
        exposure_scores = ns["PRODUCTION_RANKER"](x).cpu().numpy()
    order = np.argsort(-exposure_scores)[:K]
    lookup[qid] = rels_full[order]
ns["dataset_rels_lookup"] = lookup          # global used inside train_beta_vae

# nested labelled subsets: the 5% set is contained in the 25% set
rng = np.random.RandomState(args.seed)
all_qids = [q for q in click_cache.keys()]
perm = rng.permutation(len(all_qids))
ordered_qids = [all_qids[i] for i in perm]


def labelled_subset(frac):
    n = max(2, int(round(frac * len(ordered_qids))))
    return set(ordered_qids[:n])


def sub_dataset(base, qids):
    return SimpleNamespace(queries={q: base.queries[q] for q in base.queries if q in qids},
                           features=base.features, rels=base.rels, feature_dim=base.feature_dim)


def vali_for_fraction(frac):
    """Validation labels also scale with the label fraction (model selection must not
    use more labels than the training budget allows)."""
    vq = list(vali_full.queries.keys())
    r = np.random.RandomState(args.seed + 1)
    r.shuffle(vq)
    n = min(len(vq), max(args.min_vali_queries, int(round(frac * len(vq)))))
    return SimpleNamespace(queries={q: vali_full.queries[q] for q in vq[:n]},
                           features=vali_full.features, rels=vali_full.rels,
                           feature_dim=vali_full.feature_dim)


# ---------------------------------------------------------------------------
# 3. Time check mode
# ---------------------------------------------------------------------------
if args.time_check:
    print("\nTIME CHECK: training the full beta-VAE for 1 epoch on all queries ...")
    t0 = time.time()
    ns["train_beta_vae"](train_set, vali_for_fraction(1.0), epochs=1,
                         click_cache_override=click_cache, labelled_qids=None)
    dt = time.time() - t0
    print(f"\n1 epoch of beta-VAE took {dt:.1f} s.")
    print(f"Estimated full beta-VAE run at {args.epochs} epochs (no early stop): {dt*args.epochs/60:.0f} min")
    runs = len(args.fractions) + (1 if args.include_full else 0)
    est = runs * dt * args.epochs / 60 * (2 if args.no_adv else 1)
    print(f"Estimated total beta-VAE time for {runs} fractions"
          f"{' incl. no-adversarial' if args.no_adv else ''}: about {est:.0f} min "
          f"(+ Supervised-Only, a few minutes each).")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. Sweep
# ---------------------------------------------------------------------------
out_path = f"results_label_scarcity_yahoo_seed{args.seed}.json"
results = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    print(f"Resuming: found {len(results)} finished runs in {out_path}")


def save():
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def evaluate(score_fn):
    return ns["evaluate_ranker_fn"](score_fn, test_set, k=10, n_runs_ips=20)


fractions = list(args.fractions) + ([1.0] if args.include_full else [])
per_query = {}   # in-memory per-query scores for paired tests within this session

for frac in fractions:
    lab = labelled_subset(frac)
    vali_f = vali_for_fraction(frac)
    tag = f"{int(round(frac*100))}pct"
    print("\n" + "=" * 78)
    print(f"LABEL FRACTION {frac:.0%}: {len(lab)} labelled training queries "
          f"(of {len(ordered_qids)}), validation queries {len(vali_f.queries)}")
    print("=" * 78)

    # ---- Supervised-Only
    key = f"Supervised-Only@{tag}"
    if key not in results:
        reseed(args.seed)
        t0 = time.time()
        model = ns["train_supervised_only"](sub_dataset(train_set, lab), vali_f)
        nd, ip = evaluate(ns["make_score_fn_simple"](model))
        results[key] = {"label_fraction": frac, "NDCG@10": float(nd.mean()),
                        "IPS-DCG": float(ip.mean()), "train_seconds": time.time() - t0,
                        "n_labelled_queries": len(lab)}
        per_query[key] = (nd, ip); save()
        print(f"  {key}: NDCG@10={nd.mean():.4f}  IPS-DCG={ip.mean():.4f}")

    # ---- beta-VAE (full)
    key = f"beta-VAE@{tag}"
    if key not in results:
        reseed(args.seed)
        t0 = time.time()
        enc_r, head, _, _, hist = ns["train_beta_vae"](
            train_set, vali_f, epochs=args.epochs, track_history=True,
            click_cache_override=click_cache, labelled_qids=lab)
        nd, ip = evaluate(ns["make_score_fn_vae"](enc_r, head))
        results[key] = {"label_fraction": frac, "NDCG@10": float(nd.mean()),
                        "IPS-DCG": float(ip.mean()), "train_seconds": time.time() - t0,
                        "epochs_run": len(hist["epoch"]), "n_labelled_queries": len(lab)}
        per_query[key] = (nd, ip); save()
        print(f"  {key}: NDCG@10={nd.mean():.4f}  IPS-DCG={ip.mean():.4f}")

    # ---- beta-VAE without adversarial training (optional)
    if args.no_adv:
        key = f"beta-VAE w/o adversarial@{tag}"
        if key not in results:
            reseed(args.seed)
            t0 = time.time()
            enc_r, head, _, _, hist = ns["train_beta_vae"](
                train_set, vali_f, epochs=args.epochs, adv_enabled=False, track_history=True,
                click_cache_override=click_cache, labelled_qids=lab)
            nd, ip = evaluate(ns["make_score_fn_vae"](enc_r, head))
            results[key] = {"label_fraction": frac, "NDCG@10": float(nd.mean()),
                            "IPS-DCG": float(ip.mean()), "train_seconds": time.time() - t0,
                            "epochs_run": len(hist["epoch"]), "n_labelled_queries": len(lab)}
            per_query[key] = (nd, ip); save()
            print(f"  {key}: NDCG@10={nd.mean():.4f}  IPS-DCG={ip.mean():.4f}")

    # ---- paired comparison beta-VAE vs Supervised-Only (needs both in this session)
    ks, kv = f"Supervised-Only@{tag}", f"beta-VAE@{tag}"
    if ks in per_query and kv in per_query:
        for i, name in enumerate(["NDCG@10", "IPS-DCG"]):
            a, b = per_query[kv][i], per_query[ks][i]
            n = min(len(a), len(b))
            t, p = ttest_rel(a[:n], b[:n])
            d = float(np.mean(a[:n] - b[:n]))
            results[kv][f"diff_vs_supervised_{name}"] = {"mean_diff": d, "t": float(t), "p": float(p)}
            print(f"  beta-VAE minus Supervised-Only, {name}: {d:+.4f}  (t={t:.2f}, p={p:.3g})")
        save()

# ---------------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"SUMMARY (Yahoo! LETOR Set 2, seed {args.seed}, {args.epochs}-epoch beta-VAE budget)")
print("=" * 78)
print(f"{'Labels':>7} | {'Method':<30} | {'NDCG@10':>8} | {'IPS-DCG':>8} | {'minutes':>7}")
for k, v in sorted(results.items(), key=lambda kv: (kv[1]['label_fraction'], kv[0])):
    print(f"{v['label_fraction']:>7.0%} | {k.split('@')[0]:<30} | {v['NDCG@10']:>8.4f} | "
          f"{v['IPS-DCG']:>8.4f} | {v['train_seconds']/60:>7.1f}")
print(f"\nSaved: {out_path}")
