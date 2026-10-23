"""
Unseen identifier check for the Yandex experiments (Reviewer 5, Comment 7).

Question: how many query / URL / domain IDs in the test split never occur in the training
split, and how are they embedded?  The pipeline hashes every ID with (id % n_buckets), so an
unseen ID is never "random": it shares the embedding row of whichever training IDs fall into
the same bucket, or an untouched initial row if no training ID falls there.

This script only reads the CSV (no training).  Same data path as yandex_controls.py.

Run (Anaconda Prompt, in the folder with yandex_embedding_pipeline.py and yandex_subsample.csv):
    python check_unseen_ids.py                      (primary setting, 4000 sessions)
    python check_unseen_ids.py --max-sessions 10000
Output: yandex_unseen_ids_seed42.json   (send it back)
"""
import os, re, json, random, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--pipeline", default=os.path.join(here, "yandex_embedding_pipeline.py"))
ap.add_argument("--csv", default=os.path.join(here, "yandex_subsample.csv"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-sessions", type=int, default=4000)
args = ap.parse_args()

src = open(args.pipeline, "r", encoding="utf-8").read()
def sub_once(p, r, t):
    n, c = re.subn(p, lambda m: r, t, count=1, flags=re.M)
    assert c == 1, p
    return n
src = sub_once(r'^INPUT_CSV\s*=.*$', f'INPUT_CSV = {args.csv!r}', src)
src = sub_once(r'^SEED\s*=\s*\d+', f'SEED = {args.seed}', src)
src = src.split("# ============================================================================\n# PART 2")[0]   # data part only
ns = {"__name__": "unseen_check", "__file__": args.pipeline}
exec(compile(src, args.pipeline, "exec"), ns)

random.seed(args.seed)
serps = ns["load_yandex_csv"](args.csv)
sessions = sorted(set(s["session_id"] for s in serps))
if args.max_sessions and len(sessions) > args.max_sessions:
    rng = random.Random(args.seed); rng.shuffle(sessions)
    keep = set(sessions[:args.max_sessions]); serps = [s for s in serps if s["session_id"] in keep]
train, test = ns["split_by_session"](serps)
QB, UB, DB = ns["QUERY_HASH_BUCKETS"], ns["URL_HASH_BUCKETS"], ns["DOMAIN_HASH_BUCKETS"]

tr_q = {s["query_id"] for s in train}
tr_u = {int(u) for s in train for u in s["url_ids"]}
tr_d = {int(d) for s in train for d in s["domain_ids"]}
tr_qb = {q % QB for q in tr_q}; tr_ub = {u % UB for u in tr_u}; tr_db = {d % DB for d in tr_d}

def frac(vals, seen): 
    vals = list(vals); return float(np.mean([v in seen for v in vals])) if vals else float("nan")

res = {"n_train_serps": len(train), "n_test_serps": len(test),
       "n_train_unique": {"query": len(tr_q), "url": len(tr_u), "domain": len(tr_d)},
       "buckets": {"query": QB, "url": UB, "domain": DB},
       "train_buckets_used": {"query": len(tr_qb), "url": len(tr_ub), "domain": len(tr_db)}}
te_docs_u = [int(u) for s in test for u in s["url_ids"]]
te_docs_d = [int(d) for s in test for d in s["domain_ids"]]
res["test_doc_fraction_seen_in_train"] = {
    "query": frac([s["query_id"] for s in test for _ in s["url_ids"]], tr_q),
    "url": frac(te_docs_u, tr_u), "domain": frac(te_docs_d, tr_d)}
res["test_doc_fraction_bucket_trained"] = {
    "query": frac([s["query_id"] % QB for s in test for _ in s["url_ids"]], tr_qb),
    "url": frac([u % UB for u in te_docs_u], tr_ub), "domain": frac([d % DB for d in te_docs_d], tr_db)}

# per clicked test SERP (same order as the evaluation in yandex_controls.py)
per = []
for s in test:
    if s["clicks"].sum() > 0:
        urls = [int(u) for u in s["url_ids"]]
        per.append({"query_seen": int(s["query_id"] in tr_q),
                    "n_url_unseen": int(sum(u not in tr_u for u in urls)),
                    "n_docs": len(urls)})
res["per_clicked_serp"] = per
res["clicked_serps_all_urls_seen"] = int(sum(p["n_url_unseen"] == 0 for p in per))
res["clicked_serps_query_seen"] = int(sum(p["query_seen"] for p in per))
res["n_clicked_serps"] = len(per)
out = f"yandex_unseen_ids_seed{args.seed}_s{args.max_sessions}.json"
json.dump(res, open(out, "w"))
print(json.dumps({k: v for k, v in res.items() if k != "per_clicked_serp"}, indent=1))
print("Saved:", out)
