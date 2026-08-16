"""Short-batch A/B: baseline architecture vs. the enhanced multi-domain model.

Runs both variants under identical data budgets, seeds and batch sizes so the
metric delta reflects the architecture and loss changes rather than the schedule.
This is a pre-flight convergence check before committing to full-scale training,
not a final result.

AUC is the headline metric. Train/test loss is *not* comparable across variants:
the enhanced objective adds contrastive and artifact terms and uses label
smoothing, which raises its floor regardless of how well it classifies.

    python -m evaluation.short_batch_ab --train /data/train_data.h5 --test /data/test_data.h5 --seeds 1337 7 99
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

VARIANTS = ("baseline", "enhanced")


def run_variant(name, seed, args, out_dir):
    tag = f"{name}_seed{seed}"
    metrics_path = os.path.join(out_dir, f"{tag}_metrics.json")
    cmd = [
        sys.executable, "-u", "-m", "aceverify.train",
        "--train_path", args.train,
        "--test_path", args.test,
        "--checkpoint-path", os.path.join(out_dir, f"{tag}.pth"),
        "--model", name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--train-n", str(args.train_n),
        "--test-n", str(args.test_n),
        "--num-workers", str(args.num_workers),
        "--seed", str(seed),
        "--no-resume",
        "--metrics-json", metrics_path,
    ]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if not args.save:
        cmd += ["--no-save"]
    if name == "baseline":
        # Isolate the architecture: the baseline keeps the plain BCE objective.
        cmd += ["--contrastive-weight", "0", "--artifact-weight", "0", "--label-smoothing", "0"]

    print(f"  running {tag} ...", flush=True)
    log_path = os.path.join(out_dir, f"{tag}.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log_path) as log:
            print(log.read()[-4000:])
        raise SystemExit(f"{tag} failed with exit code {proc.returncode}; see {log_path}")

    with open(metrics_path) as f:
        return json.load(f)["metrics"]


def scalars(metrics):
    losses = metrics["train_losses"]
    return {
        "final train acc %": metrics["train_accuracies"][-1],
        "final test acc %": metrics["test_accuracies"][-1],
        "best test acc %": max(metrics["test_accuracies"]),
        "final test AUC": metrics["test_auc"][-1],
        "best test AUC": max(metrics["test_auc"]),
        "final test F1": metrics["test_f1"][-1],
        "train loss drop": losses[0] - losses[-1],
        "peak VRAM GiB": metrics["peak_vram_gib"][-1],
        "sec / epoch": sum(metrics["epoch_seconds"]) / len(metrics["epoch_seconds"]),
    }


def summarize(results, seeds):
    keys = list(scalars(results["baseline"][seeds[0]]).keys())
    per_variant = {
        v: {k: [scalars(results[v][s])[k] for s in seeds] for k in keys} for v in VARIANTS
    }

    width = max(len(k) for k in keys) + 2
    spread = "sd" if len(seeds) > 1 else ""
    header = f"{'metric'.ljust(width)}{'baseline':>18}{'enhanced':>18}{'delta':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for key in keys:
        b, e = per_variant["baseline"][key], per_variant["enhanced"][key]
        mb, me = statistics.mean(b), statistics.mean(e)
        if len(seeds) > 1:
            sb = f"{mb:.4f}±{statistics.stdev(b):.4f}"
            se = f"{me:.4f}±{statistics.stdev(e):.4f}"
        else:
            sb, se = f"{mb:.4f}", f"{me:.4f}"
        print(f"{key.ljust(width)}{sb:>18}{se:>18}{me - mb:>10.4f}")

    if len(seeds) > 1:
        deltas = [
            scalars(results["enhanced"][s])["final test AUC"]
            - scalars(results["baseline"][s])["final test AUC"]
            for s in seeds
        ]
        wins = sum(d > 0 for d in deltas)
        print(
            f"\nAUC delta per seed: {['%+.4f' % d for d in deltas]}  "
            f"(enhanced wins {wins}/{len(seeds)})"
        )
    print(f"\n{spread and 'sd over %d seeds. ' % len(seeds)}"
          "Loss is not comparable across variants (different objective).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-n", type=int, default=960)
    parser.add_argument("--test-n", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--save", action="store_true", help="Keep the checkpoints from every run")
    args = parser.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="aceverify_ab_")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Artifacts -> {out_dir}")
    print(f"{len(VARIANTS) * len(args.seeds)} runs: variants={VARIANTS} seeds={args.seeds}")

    results = {v: {} for v in VARIANTS}
    for seed in args.seeds:
        for variant in VARIANTS:
            results[variant][seed] = run_variant(variant, seed, args, out_dir)

    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    summarize(results, args.seeds)


if __name__ == "__main__":
    main()
