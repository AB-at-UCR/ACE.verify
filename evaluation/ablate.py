"""Attribute the enhanced model's short-batch behaviour to individual components.

Each configuration is trained from the same seeds under the same budget, so the
AUC column isolates one change at a time.

    python -m evaluation.ablate --train /data/train_data.h5 --test /data/test_data.h5 --seeds 1337 7
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

PLAIN_LOSS = ["--contrastive-weight", "0", "--artifact-weight", "0", "--label-smoothing", "0"]

CONFIGS = {
    "baseline": ["--model", "baseline", *PLAIN_LOSS],
    "enhanced-full": ["--model", "enhanced"],
    "enhanced-plain-loss": ["--model", "enhanced", *PLAIN_LOSS],
    "enhanced-no-frequency": ["--model", "enhanced", "--no-frequency"],
    "enhanced-no-motion": ["--model", "enhanced", "--no-motion"],
    "enhanced-spatial-only": ["--model", "enhanced", "--no-frequency", "--no-motion", *PLAIN_LOSS],
}


def run(name, seed, args, out_dir):
    tag = f"{name}_seed{seed}"
    metrics_path = os.path.join(out_dir, f"{tag}.json")
    cmd = [
        sys.executable, "-u", "-m", "aceverify.train",
        "--train_path", args.train,
        "--test_path", args.test,
        "--checkpoint-path", os.path.join(out_dir, f"{tag}.pth"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--train-n", str(args.train_n),
        "--test-n", str(args.test_n),
        "--num-workers", str(args.num_workers),
        "--seed", str(seed),
        "--no-resume", "--no-save",
        "--metrics-json", metrics_path,
        *CONFIGS[name],
    ]
    print(f"  {tag} ...", flush=True)
    log_path = os.path.join(out_dir, f"{tag}.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log_path) as log:
            print(log.read()[-3000:])
        raise SystemExit(f"{tag} failed; see {log_path}")
    with open(metrics_path) as f:
        return json.load(f)["metrics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-n", type=int, default=960)
    parser.add_argument("--test-n", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 7])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        name: {seed: run(name, seed, args, args.out_dir) for seed in args.seeds}
        for name in args.configs
    }
    with open(os.path.join(args.out_dir, "ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    width = max(len(n) for n in args.configs) + 2
    print(f"\n{'config'.ljust(width)}{'test AUC':>18}{'test acc %':>18}{'sec/epoch':>11}")
    print("-" * (width + 47))
    reference = None
    for name in args.configs:
        aucs = [results[name][s]["test_auc"][-1] for s in args.seeds]
        accs = [results[name][s]["test_accuracies"][-1] for s in args.seeds]
        secs = [statistics.mean(results[name][s]["epoch_seconds"]) for s in args.seeds]
        mean_auc = statistics.mean(aucs)
        reference = mean_auc if reference is None else reference
        spread = f"±{statistics.stdev(aucs):.4f}" if len(aucs) > 1 else ""
        print(
            f"{name.ljust(width)}{f'{mean_auc:.4f}{spread}':>18}"
            f"{f'{statistics.mean(accs):.2f}':>18}{statistics.mean(secs):>11.1f}"
        )
    print(f"\nSeeds: {args.seeds}. AUC is the comparable metric across configs.")


if __name__ == "__main__":
    main()
