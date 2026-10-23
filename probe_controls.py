"""
Position-probe controls (Yahoo! LETOR Set 2, CPU friendly)
==========================================================
Answers Reviewer 5 (Comment 2) and Reviewer 1 (Comment 8): are the position probes
of the relevance representation (about 12% accuracy against 10% chance, whole-block
R2 about 0.06) specific to ADVERSARIAL training, or would any model show them?

The identical frozen-representation probes are applied to the hidden representation of

    random_init   an UNTRAINED relevance encoder (what the raw features give)
    naive         Naive ranker trained on clicks (64-d hidden layer)
    supervised    Supervised-Only ranker trained on labels only (64-d hidden layer)
    vae_full      full beta-VAE, z_r (24-d)
    vae_noadv     beta-VAE WITHOUT adversarial training, z_r (24-d)
    single_enc    single-encoder ablation (40-d)
    simple_adv    simple position-adversarial ranker (24-d)

Three definitions of "position" are probed, because the original protocol is confounded:
    relsorted   the 10 best documents of a query ordered by TRUE relevance (original
                protocol of the pipeline: position is then strongly tied to relevance)
    exposure    the 10 top documents ordered by the production ranker, i.e. the order in
                which documents were actually presented when clicks were simulated
    random      the same documents as "exposure" but with positions randomly shuffled
                (null control: any probe above chance here is a probe artefact)

Probes (identical to the pipeline's): position R2 (Ridge and Random Forest), position
classification accuracy (Logistic Regression and Random Forest, chance = 10%), and,
on the relsorted documents, relevance R2.

HOW TO RUN (Anaconda Prompt, same folder as beta_vae_unified_pipeline.py):

    set KMP_DUPLICATE_LIB_OK=TRUE
    python probe_controls.py --models random_init naive supervised
    python probe_controls.py            (all models, about 1.5 hours)

It is resumable: finished models are skipped if you stop and restart.
Output: results_probe_controls_yahoo_seed<seed>.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import re
import time

ALL_MODELS = ["random_init", "naive", "supervised", "vae_full", "vae_noadv", "single_enc", "simple_adv"]

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "beta_vae_unified_pipeline.py"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=30, help="epoch budget for the VAE-style models")
ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
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

# newer scikit-learn removed the multi_class argument (multinomial is the default anyway)
src = src.replace(', multi_class="multinomial"', "")

ns = {"__name__": "pipeline_probe", "__file__": args.pipeline}
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


# ---------------------------------------------------------------- data (as in main())
reseed(args.seed)
train_set = ns["LTRDataset"](ns["TRAIN_FILE"])
vali_set = ns["LTRDataset"](ns["VALI_FILE"], scaler=train_set.scaler, feature_dim=train_set.feature_dim)
test_set = ns["LTRDataset"](ns["TEST_FILE"], scaler=train_set.scaler, feature_dim=train_set.feature_dim)
print(f"  train={len(train_set.queries)} vali={len(vali_set.queries)} test={len(test_set.queries)} "
      f"features={train_set.feature_dim}")
ns["train_production_ranker"](train_set)
click_cache = ns["_precompute_training_clicks"](train_set)
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

# ------------------------------------------------- test documents for the three protocols
rng = np.random.RandomState(args.seed + 7)
test_docs = []     # list of (feats_all_docs, order_rel, order_exp, rels_all)
with torch.no_grad():
    for qid, idxs in test_set.queries.items():
        feats = test_set.features[idxs]
        rels = test_set.rels[idxs]
        if len(rels) < 2:
            continue
        x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
        exp_scores = ns["PRODUCTION_RANKER"](x).cpu().numpy()
        test_docs.append((feats, np.argsort(-rels)[:K], np.argsort(-exp_scores)[:K], rels))
print(f"  probe documents from {len(test_docs)} test queries")


def collect(rep_fn):
    """rep_fn(feats_np)->(n,d) representation. Returns arrays for the three protocols."""
    out = {"relsorted": ([], [], []), "exposure": ([], [], []), "random": ([], [], [])}
    for feats, o_rel, o_exp, rels in test_docs:
        Z = rep_fn(feats)
        n = len(o_rel)
        out["relsorted"][0].append(Z[o_rel]); out["relsorted"][1].append(np.arange(n))
        out["relsorted"][2].append(rels[o_rel])
        m = len(o_exp)
        out["exposure"][0].append(Z[o_exp]); out["exposure"][1].append(np.arange(m))
        out["exposure"][2].append(rels[o_exp])
        out["random"][0].append(Z[o_exp]); out["random"][1].append(rng.permutation(m))
        out["random"][2].append(rels[o_exp])
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]).astype(int), np.concatenate(v[2]).astype(int))
            for k, v in out.items()}


def run_probes(rep_fn):
    data = collect(rep_fn)
    res = {}
    for name, (Z, pos, rel) in data.items():
        pr = ns["compute_block_informativeness"](Z, pos)
        pc = ns["compute_position_classifier_accuracy"](Z, pos)
        res[name] = {"n_docs": int(len(pos)), "dim": int(Z.shape[1]),
                     "pos_R2_linear": pr["R2_linear"], "pos_R2_rf": pr["R2_nonlinear"],
                     "pos_acc_linear": pc["accuracy_linear"], "pos_acc_rf": pc["accuracy_nonlinear"],
                     "chance": pc["chance_level"]}
        if name == "relsorted":
            rr = ns["compute_block_informativeness"](Z, rel)
            res[name]["rel_R2_linear"] = rr["R2_linear"]; res[name]["rel_R2_rf"] = rr["R2_nonlinear"]
    return res


def np_rep(fn):
    def f(feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            return fn(x).cpu().numpy()
    return f


def eval_ndcg(score_fn):
    nd, _ = ns["evaluate_ranker_fn"](score_fn, test_set, k=10, n_runs_ips=1)
    return float(np.mean(nd))


out_path = f"results_probe_controls_yahoo_seed{args.seed}.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}


def finish(key, rep_fn, score_fn, t0):
    res = run_probes(rep_fn)
    res["test_NDCG@10"] = eval_ndcg(score_fn) if score_fn is not None else None
    res["train_seconds"] = time.time() - t0
    results[key] = res
    json.dump(results, open(out_path, "w"), indent=2)
    r = res["relsorted"]; e = res["exposure"]; n = res["random"]
    print(f"\n  [{key}] dim={r['dim']}  NDCG@10={res['test_NDCG@10']}")
    print(f"     position acc (RF)  relsorted={r['pos_acc_rf']:.3f}  exposure={e['pos_acc_rf']:.3f}  "
          f"random(null)={n['pos_acc_rf']:.3f}   (chance {r['chance']:.2f})")
    print(f"     position R2 (RF)   relsorted={r['pos_R2_rf']:.3f}  exposure={e['pos_R2_rf']:.3f}  "
          f"random(null)={n['pos_R2_rf']:.3f}")
    print(f"     relevance R2 (RF, relsorted) = {r['rel_R2_rf']:.3f}")


for key in args.models:
    if key in results:
        print(f"\n[{key}] already finished, skipping"); continue
    print("\n" + "=" * 78 + f"\nMODEL: {key}\n" + "=" * 78)
    reseed(args.seed)
    t0 = time.time()
    d = train_set.feature_dim
    if key == "random_init":
        enc = ns["RelevanceEncoder"](d).to(DEVICE).eval()
        finish(key, np_rep(lambda x: enc(x)[1]), None, t0)
    elif key in ("naive", "supervised"):
        model = (ns["train_naive"] if key == "naive" else ns["train_supervised_only"])(train_set, vali_set)
        model.eval()
        finish(key, np_rep(lambda x: model.net[:5](x)), ns["make_score_fn_simple"](model), t0)
    elif key in ("vae_full", "vae_noadv"):
        kw = {} if key == "vae_full" else {"adv_enabled": False}
        enc_r, head, _, _, _ = ns["train_beta_vae"](train_set, vali_set, epochs=args.epochs,
                                                     click_cache_override=click_cache, **kw)
        enc_r.eval()
        finish(key, np_rep(lambda x: enc_r(x)[1]), ns["make_score_fn_vae"](enc_r, head), t0)
    elif key == "single_enc":
        enc, head = ns["train_single_encoder_ablation"](train_set, vali_set, epochs=args.epochs)
        enc.eval()
        finish(key, np_rep(lambda x: enc(x)[1]), ns["make_score_fn_single_encoder"](enc, head), t0)
    elif key == "simple_adv":
        enc, head = ns["train_simple_adversarial_ranker"](train_set, vali_set, epochs=args.epochs,
                                                          click_cache_override=click_cache)
        enc.eval()
        finish(key, np_rep(lambda x: enc(x)), ns["make_score_fn_simple_adversarial"](enc, head), t0)

print("\n" + "=" * 78)
print("SUMMARY: random forest position accuracy (chance 0.10) / position R2 / relevance R2")
print("=" * 78)
print(f"{'model':<12}{'dim':>4} {'NDCG@10':>8} | {'acc rel':>7}{'acc exp':>8}{'acc null':>9} | "
      f"{'R2 rel':>7}{'R2 exp':>8}{'R2 null':>8} | {'relevR2':>7}")
for k in ALL_MODELS:
    if k in results:
        v = results[k]; r, e, n = v["relsorted"], v["exposure"], v["random"]
        nd = f"{v['test_NDCG@10']:.4f}" if v["test_NDCG@10"] is not None else "   -   "
        print(f"{k:<12}{r['dim']:>4} {nd:>8} | {r['pos_acc_rf']:>7.3f}{e['pos_acc_rf']:>8.3f}{n['pos_acc_rf']:>9.3f} | "
              f"{r['pos_R2_rf']:>7.3f}{e['pos_R2_rf']:>8.3f}{n['pos_R2_rf']:>8.3f} | {r['rel_R2_rf']:>7.3f}")
print(f"\nSaved: {out_path}")
