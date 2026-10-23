"""
================================================================================
Computational complexity measurement: parameter counts + inference latency
================================================================================
Fills in the long-pending [TBD] in Section 4.6.8 (Computational Complexity
and Implementation Details), and directly answers a reviewer question on
production inference latency.

Needs NO training data and NO trained weights -- parameter counts are a
property of the architecture alone, and inference latency is measured on
randomly-initialized models with random input tensors, which is valid for
latency characterization (forward-pass cost depends on architecture and
tensor shapes, not on learned weight VALUES).

Run this once; paste its printed output back to fill in Section 4.6.8 and
to answer the reviewer's inference-latency question with real numbers.
================================================================================
"""

import time
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ---- architecture definitions (must match the main pipeline exactly) ----
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
        return self.mu(h)  # inference uses mu only (no sampling)


class BiasEncoder(nn.Module):
    def __init__(self, in_dim, pos_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + pos_dim, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 48), nn.ReLU(),
        )
        self.mu = nn.Linear(48, 16)
        self.logvar = nn.Linear(48, 16)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )


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


class PositionPredictor(nn.Module):
    def __init__(self, zb_dim=16, n_positions=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zb_dim, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_positions),
        )


class SimpleRanker(nn.Module):
    """The baseline architecture (Naive, IPS, Affine, Bayes-IPS, Regression-EM, Supervised-Only)."""
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
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.obs_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.rel_head = nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))


class InferenceModel(nn.Module):
    """What actually runs at deployment: Enc_r (mu only) + RankingHead.
    No decoder, no discriminator, no bias encoder, no position predictor,
    no position input at all -- matches Section 4.7's stated inference path."""
    def __init__(self, in_dim):
        super().__init__()
        self.enc_r = RelevanceEncoder(in_dim)
        self.rank_head = RankingHead()

    def forward(self, x):
        mu_r = self.enc_r(x)
        return self.rank_head(mu_r)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_inference_latency(model, in_dim, batch_sizes=(1, 10, 100, 1000), n_repeats=100, warmup=10):
    model.eval().to(DEVICE)
    results = {}
    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.randn(bs, in_dim, device=DEVICE)
            for _ in range(warmup):
                _ = model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_repeats):
                _ = model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            per_call_ms = (elapsed / n_repeats) * 1000
            per_doc_us = (elapsed / n_repeats / bs) * 1e6
            results[bs] = {"per_call_ms": per_call_ms, "per_doc_us": per_doc_us}
    return results


def run_for_feature_dim(in_dim, label):
    print("=" * 80)
    print(f"FEATURE DIMENSION: {in_dim}  ({label})")
    print("=" * 80)

    enc_r = RelevanceEncoder(in_dim)
    enc_b = BiasEncoder(in_dim)
    dec = Decoder()
    rank_head = RankingHead()
    disc = Discriminator()
    pos_pred = PositionPredictor()
    simple_ranker = SimpleRanker(in_dim)
    inforank = InfoRankNet(in_dim)
    inference_model = InferenceModel(in_dim)

    print("\nParameter counts:")
    print(f"  RelevanceEncoder (Enc_r):    {count_params(enc_r):>10,}")
    print(f"  BiasEncoder (Enc_b):         {count_params(enc_b):>10,}")
    print(f"  Decoder:                     {count_params(dec):>10,}")
    print(f"  RankingHead:                 {count_params(rank_head):>10,}")
    print(f"  Discriminator:               {count_params(disc):>10,}")
    print(f"  PositionPredictor:           {count_params(pos_pred):>10,}")
    total_train_time = (count_params(enc_r) + count_params(enc_b) + count_params(dec) +
                         count_params(rank_head) + count_params(disc) + count_params(pos_pred))
    print(f"  ---")
    print(f"  TOTAL (all components, training-time):  {total_train_time:>10,}")
    print(f"  TOTAL (inference-time: Enc_r + RankingHead only): {count_params(inference_model):>10,}")
    print(f"\n  Baseline SimpleRanker (Naive/IPS/Affine/etc.): {count_params(simple_ranker):>10,}")
    print(f"  InfoRank baseline:                             {count_params(inforank):>10,}")
    print(f"\n  Inference-time param ratio vs. SimpleRanker baseline: "
          f"{count_params(inference_model) / count_params(simple_ranker):.2f}x")

    print("\nInference latency (CPU/GPU forward pass only, randomly-initialized weights,")
    print("batched documents per call):")
    for name, model in [("Ours (Enc_r + RankingHead)", inference_model),
                         ("Baseline (SimpleRanker)", simple_ranker)]:
        print(f"\n  {name}:")
        latencies = measure_inference_latency(model, in_dim)
        for bs, stats in latencies.items():
            print(f"    batch={bs:>5}  {stats['per_call_ms']:>8.3f} ms/call  "
                  f"{stats['per_doc_us']:>8.2f} us/document")
    print()


if __name__ == "__main__":
    # Yahoo! LETOR: ~700 features. MSLR-WEB30K: ~136 features.
    run_for_feature_dim(700, "Yahoo! LETOR Set 2")
    run_for_feature_dim(136, "MSLR-WEB30K")
