# Adversarial Disentanglement of Relevance and Position in Learning to Rank from Biased Clicks

Code and result files for the manuscript submitted to Expert Systems with Applications (ESWA-D-26-05900).

The method is an adversarial beta-VAE with two encoders. The ranking head is trained with relevance labels together with a click based reconstruction and disentanglement objective. It is not a click-only method. The paper reports that it does not improve on a label-only ranker (Supervised-Only) on Yahoo!, and that no component is shown to be necessary.

## Layout
- `scripts/` all code (see the table below)
- `results/` the result files used for the tables and figures in the paper
- `archive/` earlier exploratory notebooks, kept for provenance only, not used for any reported number

## Setup
1. Python 3.9 or newer, then `pip install -r requirements.txt` (fill in exact versions; see the note in that file).
2. Datasets (not redistributed here):
   - Yahoo! LETOR Set 2 (`set2.train.txt`, `set2.valid.txt`, `set2.test.txt`)
   - MSLR-WEB30K (Fold 1)
   - Yandex Personalized Web Search Challenge raw logs (Kaggle)
3. Set the data paths at the top of `scripts/beta_vae_unified_pipeline.py`, or pass `--yahoo-dir` to `run_yahoo_seed.py`.

## What produces what
| Paper item | Script | Result file |
|---|---|---|
| Table 2, main results, Yahoo!, per seed | `run_yahoo_seed.py --seed {42,123,7}` (calls `beta_vae_unified_pipeline.py`) | `results/results_yahoo_seed{42,123,7}.json` |
| Table 2, MSLR-WEB30K | `mslr_only_pipeline.py` | `results/results_mslr_seed{7,123}.json` |
| Table 3, multi seed summary | computed from the per seed files above | same |
| Direct comparison with Supervised-Only, seed level tests | computed from the per seed files above | same |
| Five seed ablation | `beta_vae_unified_pipeline.py` with `RUN_MULTISEED_ABLATION=True` | `results/multiseed_ablation_yahoo.json` |
| Factorial ablation (Appendix E) | `beta_vae_unified_pipeline.py` with `RUN_FACTORIAL_ABLATION=True` | `results/factorial_ablation_{yahoo,mslr}.json` |
| Hyperparameter sensitivity (Appendix A) | `beta_vae_unified_pipeline.py` with `RUN_HYPERPARAMETER_SWEEP=True` | `results/hyperparam_sweep_yahoo.json` |
| Mismatched train and evaluation bias (Appendix C) | `beta_vae_unified_pipeline.py` with `RUN_MISMATCHED_BIAS_EVAL=True` | `results/mismatched_bias_eval_yahoo_seed{7,123}.json` |
| Probe controls (Appendix F) | `probe_controls.py` | `results/results_probe_controls_yahoo_seed42.json` |
| Biased label experiment (Appendix G) | `biased_label_experiment.py` | `results/results_biased_labels_yahoo_seed42.json` |
| Label scarcity (Appendix H) | `label_scarcity_sweep.py` | `results/results_label_scarcity_yahoo_seed42.json` |
| Yandex main results | `yandex_parser.py`, `yandex_embedding_pipeline.py` | `results/results_yandex_seed42.json` |
| Yandex controls (Section 6.5) | `yandex_controls.py` | `results/results_yandex_controls_seed42.json` |
| Yandex unseen identifiers | `check_unseen_ids.py` | `results/yandex_unseen_ids_seed42_s4000.json` |
| Parameter counts and latency (Appendix B) | `compute_complexity_measurement.py` | printed output |
| Extended ranking depth comparison | `extended_depth_ablation_comparison.py` | printed output |
| Figures 3 and 4 | `regenerate_performance_figures.py` | uses `results/results_yahoo_seed42.json` |

The click simulation (affine position and trust bias model) and all baselines (Naive, IPS, Regression-EM, Bayes-IPS, Affine, InfoRank, Supervised-Only, simple adversarial ranker) are implemented inside `beta_vae_unified_pipeline.py`.

## Example commands
```
python scripts/run_yahoo_seed.py --seed 42 --yahoo-dir /path/to/ltrc_yahoo
python scripts/probe_controls.py --seed 42 --epochs 30 --train set2.train.txt --vali set2.valid.txt --test set2.test.txt
python scripts/yandex_controls.py --threads 6 --max-sessions 4000
python scripts/check_unseen_ids.py
```
Run each script with `--help` for its options.

## Reproducibility notes
- Seeds are fixed (42, 123, 7), but training is not bit-exact under a fixed seed (parallel and GPU nondeterminism). Repeating the default configuration in the hyperparameter sweep gave NDCG@10 of 0.7246, 0.7219 and 0.7204. Expect differences of this size when rerunning.
- Training times in the paper (969.66 s on Yahoo!, 17,573.91 s on MSLR) were measured on a CUDA GPU.
- The Yandex experiments use a 4,000 session subsample and hashed identifier embeddings; most test identifiers are unseen (see the paper).
