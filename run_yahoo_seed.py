"""
run_yahoo_seed.py -- rerun the Yahoo! LETOR pipeline for ONE seed, in its own folder.

Put this file in the same folder as beta_vae_unified_pipeline.py (your local copy,
with your data paths already set), then run from that folder:

    python run_yahoo_seed.py --seed 42 --yahoo-dir "E:\\Learning to Rank Challenge\\ltrc_yahoo"
    python run_yahoo_seed.py --seed 7 --yahoo-dir "..." --enable RUN_MISMATCHED_BIAS_EVAL
    python run_yahoo_seed.py --seed 42 --yahoo-dir "..." --dry-run   # only writes the modified script

What it does
  * copies the pipeline into  rerun_yahoo_seed<N>/  with SEED and DATASET set,
  * switches every RUN_* sweep flag OFF (except the ones given with --enable),
  * makes the result and figure file names seed-specific, so nothing you already
    have (results_yahoo.json, tsne_yahoo.png, ...) can be overwritten,
  * runs it, showing the console output and saving it to console_log.txt.

Your original pipeline file is never modified.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--script", default="beta_vae_unified_pipeline.py")
ap.add_argument("--enable", nargs="*", default=[],
                help="RUN_* flags to switch ON (every other RUN_* flag is forced OFF)")
ap.add_argument("--yahoo-dir", default=None,
                help="folder holding set2.train.txt, set2.valid.txt, set2.test.txt "
                     "(overrides the data paths written in the pipeline)")
ap.add_argument("--dry-run", action="store_true",
                help="write the modified script and stop, without running it")
args = ap.parse_args()

seed = args.seed
src_path = Path(args.script).resolve()
if not src_path.exists():
    sys.exit(f"Cannot find {src_path}. Run this from the folder that holds the pipeline.")
src = src_path.read_text(encoding="utf-8")


def sub_once(pattern, replacement, text):
    new, n = re.subn(pattern, lambda m: replacement, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"Could not find this line in {src_path.name}: {pattern}")
    return new


src = sub_once(r'^SEED\s*=\s*\d+', f'SEED = {seed}', src)
src = sub_once(r'^DATASET\s*=\s*"[a-z]+"', 'DATASET = "yahoo"', src)
src = sub_once(r'^RESULTS_FILE\s*=.*$', f'RESULTS_FILE = "results_yahoo_seed{seed}.json"', src)

# data paths: optional override, then always verify the three files exist
if args.yahoo_dir:
    d = Path(args.yahoo_dir)
    for var, fname in (("YAHOO_TRAIN_FILE", "set2.train.txt"),
                       ("YAHOO_VALI_FILE", "set2.valid.txt"),
                       ("YAHOO_TEST_FILE", "set2.test.txt")):
        src = sub_once(rf'^{var}\s*=.*$', f'{var} = r"{d / fname}"', src)

missing = []
for var in ("YAHOO_TRAIN_FILE", "YAHOO_VALI_FILE", "YAHOO_TEST_FILE"):
    m = re.search(rf'^{var}\s*=\s*r"(.*)"', src, flags=re.M)
    if not m:
        sys.exit(f"Could not read {var} from {src_path.name}")
    if not Path(m.group(1)).exists():
        missing.append(m.group(1))
if missing:
    sys.exit("These Yahoo! data files were not found:\n  " + "\n  ".join(missing) +
             "\nRe-run with  --yahoo-dir \"<folder that holds set2.train.txt>\"")
print("Data files found: OK")

# sweep flags: everything off unless explicitly enabled
found = set()


def set_flag(m):
    name = m.group(1)
    found.add(name)
    return f"{name}{m.group(2)}{'True' if name in args.enable else 'False'}"


src = re.sub(r'^(RUN_[A-Z_]+)(\s*=\s*)(?:True|False)', set_flag, src, flags=re.M)
unknown = set(args.enable) - found
if unknown:
    sys.exit(f"Unknown flag(s): {sorted(unknown)}. Available: {sorted(found)}")

# seed-specific figure names
for old, new in [("tsne_{DATASET}.png", "tsne_{DATASET}_seed{SEED}.png"),
                 ("training_dynamics_{DATASET}.png", "training_dynamics_{DATASET}_seed{SEED}.png")]:
    if old not in src:
        sys.exit(f"Could not find the output name {old} in {src_path.name}")
    src = src.replace(old, new)

run_dir = Path(f"rerun_yahoo_seed{seed}")
run_dir.mkdir(exist_ok=True)
out_script = run_dir / f"pipeline_yahoo_seed{seed}.py"
out_script.write_text(src, encoding="utf-8")
print(f"Wrote {out_script}")
print(f"  SEED = {seed}, DATASET = yahoo, results -> results_yahoo_seed{seed}.json")
print(f"  flags ON: {sorted(args.enable) if args.enable else 'none'}")
if args.dry_run:
    sys.exit(0)

# PYTHONIOENCODING avoids a crash on Windows when the script prints the beta symbol
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
log_path = run_dir / "console_log.txt"
t0 = time.time()
def emit(line):
    """Show a log line without ever crashing on a character the console cannot
    display (Windows cp1252 consoles cannot show the beta symbol). The log file
    is written separately in UTF-8 and always keeps the exact text."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        sys.stdout.write(line.encode(enc, errors="replace").decode(enc))
    sys.stdout.flush()


with open(log_path, "w", encoding="utf-8") as log:
    proc = subprocess.Popen([sys.executable, out_script.name], cwd=run_dir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in proc.stdout:
            log.write(line)       # log first, so it is complete even if display fails
            log.flush()
            emit(line)
        code = proc.wait()
    except BaseException:
        proc.kill()               # never leave an orphaned run behind
        raise

print(f"\nFinished with exit code {code} in {(time.time() - t0) / 60:.1f} minutes")
print(f"Send me:  {run_dir / f'results_yahoo_seed{seed}.json'}")
print(f"     and: {log_path}")
