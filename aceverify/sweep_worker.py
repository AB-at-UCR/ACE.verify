"""Indexed-job worker for NRP hyperparameter sweeps.

Reads JOB_COMPLETION_INDEX (set by Kubernetes completionMode: Indexed)
and launches aceverify.train with the matching (lr, weight_decay) pair.
"""
import os
import sys

from aceverify.train import main as train_main

CONFIGS = [
    {"lr": "1e-5", "weight_decay": "1e-4"},
    {"lr": "5e-5", "weight_decay": "1e-4"},
    {"lr": "1e-4", "weight_decay": "1e-4"},
    {"lr": "5e-5", "weight_decay": "1e-3"},
]


def main():
    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"JOB_COMPLETION_INDEX={idx} out of range 0..{len(CONFIGS) - 1}")
    cfg = CONFIGS[idx]
    ckpt = f"/workspace/results/sweep_{idx}_lr{cfg['lr']}_wd{cfg['weight_decay']}.pth"
    sys.argv = [
        "aceverify.train",
        "--train_path", "/workspace/data/train_data.h5",
        "--test_path", "/workspace/data/test_data.h5",
        "--checkpoint-path", ckpt,
        "--epochs", "5",
        "--batch-size", "8",
        "--lr", cfg["lr"],
        "--weight-decay", cfg["weight_decay"],
        "--train-n", "1000",
        "--test-n", "200",
        "--num-workers", "2",
    ]
    train_main()


if __name__ == "__main__":
    main()
