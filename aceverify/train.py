import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import ACEDataset, read_labels
from .losses import DeepfakeCriterion
from .model import ACEVerifyModel

logger = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO", force: bool = False):
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        force=force,
    )
    logging.getLogger().setLevel(numeric_level)
    logger.setLevel(numeric_level)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "enhanced":
        return ACEVerifyModel(**kwargs)
    if name == "baseline":
        from .baseline_model import ACEVerifyBaseline

        return ACEVerifyBaseline()
    raise ValueError(f"Unknown model '{name}'; expected 'enhanced' or 'baseline'.")


def load_data(indices, path, n, training, shuffle_data=True, dataset_class=ACEDataset):
    if len(indices) == 0:
        logger.debug("Loading data from %s (n=%s, training=%s)", path, n, training)
        num_each = n // 2

        all_labels = read_labels(path)
        real_indices = np.where(all_labels == 0)[0]
        fake_indices = np.where(all_labels == 1)[0]

        if len(real_indices) == 0 or len(fake_indices) == 0:
            raise ValueError(
                f"Expected both real(0) and fake(1) labels in {path}, "
                f"found real={len(real_indices)}, fake={len(fake_indices)}"
            )

        sel_real = np.random.choice(real_indices, min(len(real_indices), num_each), replace=False)
        sel_fake = np.random.choice(fake_indices, min(len(fake_indices), num_each), replace=False)

        sub_indices = np.concatenate([sel_real, sel_fake])
    else:
        logger.debug("Using provided indices for data loading from %s (n=%s)", path, n)
        sub_indices = np.array(indices)

    if len(sub_indices) == 0:
        raise ValueError(f"No samples selected from {path}. Check file contents and requested n={n}.")
    if shuffle_data:
        np.random.shuffle(sub_indices)

    dataset = dataset_class(h5_path=path, indices=sub_indices, is_training=training)
    logger.debug("Built dataset with %d samples from %s", len(sub_indices), path)
    return dataset


def _log_gpu_memory(tag: str):
    if not torch.cuda.is_available():
        return
    logger.info(
        "GPU memory [%s]: allocated=%.2f GiB reserved=%.2f GiB peak=%.2f GiB",
        tag,
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
        torch.cuda.max_memory_allocated() / 1024**3,
    )


def _make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def find_safe_batch_size(model, device, start_bs: int, use_amp: bool, amp_dtype=torch.float16) -> int:
    """Halve batch size until a dummy forward+backward fits in VRAM."""
    if device.type != "cuda":
        return start_bs

    model.train()
    # BatchNorm layers must not see the random probe tensors: updating their running
    # statistics with noise poisons the pretrained EfficientNet before step 0.
    batch_norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for module in batch_norms:
        module.eval()

    try:
        bs = max(1, int(start_bs))
        while bs >= 1:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                videos = torch.randn(bs, 3, 16, 224, 224, device=device)
                specs = torch.randn(bs, 1, 224, 224, device=device)
                labels = torch.zeros(bs, 1, device=device)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = model(videos, specs)
                    loss = nn.functional.binary_cross_entropy_with_logits(outputs.float(), labels)
                loss.backward()
                del videos, specs, labels, outputs, loss
                _log_gpu_memory(f"probe bs={bs}")
                logger.info("Selected batch size %d after VRAM probe", bs)
                return bs
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "OOM during VRAM probe at batch_size=%d; trying %d", bs, max(1, bs // 2)
                )
                if bs == 1:
                    raise
                bs //= 2
            finally:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        return 1
    finally:
        for module in batch_norms:
            module.train()


def _run_validation(model, loader, device, criterion, use_amp, amp_dtype):
    model.eval()
    probabilities, targets, losses = [], [], []

    with torch.no_grad():
        for videos, specs, labels in loader:
            videos = videos.to(device, non_blocking=True)
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                outputs, aux = model(videos, specs, return_aux=True)

            loss, _ = criterion(outputs, labels, aux)
            losses.append(loss.item())
            probabilities.extend(torch.sigmoid(outputs.float()).view(-1).cpu().tolist())
            targets.extend(labels.view(-1).cpu().tolist())

    predictions = [1.0 if p > 0.5 else 0.0 for p in probabilities]
    accuracy = 100.0 * float(np.mean(np.array(predictions) == np.array(targets)))
    try:
        auc = roc_auc_score(targets, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "auc": float(auc),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "predictions": predictions,
        "targets": targets,
    }


def train_model(config):
    model = config["model"]
    device = config["device"]
    epochs = config["epochs"]
    criterion = config["criterion"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    test_path = config["test_path"]
    train_path = config["train_path"]
    batch_size = config["batch_size"]
    shuffle_data = config.get("shuffle_data", True)
    checkpoint_path = config["checkpoint_path"]
    train_indices = config.get("train_indices", [])
    test_indices = config.get("test_indices", [])
    dataset_class = config.get("dataset_class", ACEDataset)
    train_n = config.get("train_n", 1000)
    test_n = config.get("test_n", 200)
    num_workers = config.get("num_workers", 2)
    max_steps = config.get("max_steps", 0)
    grad_clip = config.get("grad_clip", 1.0)
    save_checkpoint = config.get("save_checkpoint", True)
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    auto_batch = config.get("auto_batch", True)

    model = model.to(device)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

    if auto_batch:
        batch_size = find_safe_batch_size(model, device, batch_size, use_amp, amp_dtype)
        config["batch_size"] = batch_size

    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger.info(
        "AMP=%s dtype=%s scaler=%s auto_batch=%s effective_batch_size=%d max_steps=%s",
        use_amp,
        amp_dtype,
        use_scaler,
        auto_batch,
        batch_size,
        max_steps or "all",
    )

    test_dataset = load_data(
        test_indices, test_path, test_n, False, shuffle_data=shuffle_data, dataset_class=dataset_class
    )
    test_loader = _make_loader(test_dataset, batch_size, False, num_workers, device)

    if config.get("resume", True):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state)
            logger.info("Resumed weights from %s", checkpoint_path)
        except FileNotFoundError:
            logger.info("No checkpoint at %s. Starting from pretrained backbone weights.", checkpoint_path)
        except (RuntimeError, KeyError) as exc:
            # Architecture changed under an old checkpoint; pretrained backbones are
            # a valid starting point, so warn rather than abort.
            logger.warning(
                "Checkpoint %s is incompatible with the current architecture "
                "(%s); starting from pretrained backbone weights.",
                checkpoint_path,
                str(exc).splitlines()[0],
            )

    logger.info("Training start on %s", device)
    metrics = {
        "epochs": [],
        "train_losses": [],
        "train_accuracies": [],
        "test_accuracies": [],
        "test_auc": [],
        "test_f1": [],
        "test_losses": [],
        "learning_rate": [],
        "batch_size": [],
        "peak_vram_gib": [],
        "epoch_seconds": [],
    }
    validation = None

    for epoch in range(epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)
        epoch_start = time.time()

        train_dataset = load_data(
            train_indices,
            train_path,
            train_n,
            True,
            shuffle_data=shuffle_data,
            dataset_class=dataset_class,
        )

        total_loss = 0.0
        steps = 0
        train_correct = 0
        train_total = 0

        # An OOM halves the batch size and restarts the epoch, so that the reported
        # loss always covers a full pass at a single batch size.
        while steps == 0:
            train_loader = _make_loader(train_dataset, batch_size, True, num_workers, device)
            if len(train_loader) == 0:
                raise ValueError(
                    f"Empty training loader for epoch {epoch + 1}. "
                    f"Check train_path={train_path} and sampled dataset contents."
                )

            model.train()
            total_loss = 0.0
            steps = 0
            train_correct = 0
            train_total = 0
            oom = False

            for i, (videos, specs, labels) in enumerate(train_loader):
                if max_steps and i >= max_steps:
                    logger.info("Reached max_steps=%d; ending epoch early", max_steps)
                    break

                videos = videos.to(device, non_blocking=True)
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                        outputs, aux = model(videos, specs, return_aux=True)
                    loss, parts = criterion(outputs, labels, aux)

                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "OOM at epoch %d step %d; restarting epoch with batch_size %d -> %d",
                        epoch + 1,
                        i,
                        batch_size,
                        max(1, batch_size // 2),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    del videos, specs, labels
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    config["batch_size"] = batch_size
                    test_loader = _make_loader(test_dataset, batch_size, False, num_workers, device)
                    oom = True
                    break

                if scheduler is not None:
                    scheduler.step()

                predictions = (torch.sigmoid(outputs.float()) > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                total_loss += loss.item()
                steps += 1

                if i % 10 == 0:
                    detail = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                    logger.info(
                        "Epoch [%d/%d], Step [%d/%d], Loss: %.4f (%s), Train Acc: %.2f%%, LR: %.2e",
                        epoch + 1,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        detail,
                        100 * train_correct / max(1, train_total),
                        optimizer.param_groups[0]["lr"],
                    )
                    _log_gpu_memory(f"epoch {epoch + 1} step {i}")

            if oom:
                steps = 0
                continue

            if steps == 0:
                raise ValueError(
                    f"No training steps ran in epoch {epoch + 1} despite a non-empty loader."
                )

        avg_train_loss = total_loss / steps
        train_accuracy = 100 * (train_correct / train_total)

        validation = _run_validation(model, test_loader, device, criterion, use_amp, amp_dtype)
        if not validation["targets"]:
            raise ValueError(
                f"No validation samples were evaluated in epoch {epoch + 1}. Check test_path={test_path}."
            )

        peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        metrics["epochs"].append(epoch + 1)
        metrics["train_losses"].append(avg_train_loss)
        metrics["train_accuracies"].append(train_accuracy)
        metrics["test_accuracies"].append(validation["accuracy"])
        metrics["test_auc"].append(validation["auc"])
        metrics["test_f1"].append(validation["f1"])
        metrics["test_losses"].append(validation["loss"])
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])
        metrics["batch_size"].append(batch_size)
        metrics["peak_vram_gib"].append(round(peak_vram, 3))
        metrics["epoch_seconds"].append(round(time.time() - epoch_start, 1))

        logger.info("---Epoch %d Summary---", epoch + 1)
        logger.info("    Average Training Loss: %.4f", avg_train_loss)
        logger.info("    Training Accuracy: %.2f%%", train_accuracy)
        logger.info("    Test Loss: %.4f", validation["loss"])
        logger.info("    Test Accuracy: %.2f%%", validation["accuracy"])
        logger.info("    Test AUC: %.4f  F1: %.4f", validation["auc"], validation["f1"])
        logger.info("    Epoch time: %.1fs  Peak VRAM: %.2f GiB", metrics["epoch_seconds"][-1], peak_vram)
        logger.info("------------------------------------------------")

    if validation is not None:
        logger.info("Final Classification Report:")
        logger.info(
            "\n%s",
            classification_report(
                validation["targets"],
                validation["predictions"],
                target_names=["Real", "Fake"],
                zero_division=0,
            ),
        )

    if save_checkpoint:
        try:
            torch.save(model.state_dict(), checkpoint_path)
            metrics_path = checkpoint_path.replace(".pth", "_metrics.csv")
            pd.DataFrame(metrics).to_csv(metrics_path, index=False)
            logger.info("Saved checkpoint to %s and metrics to %s", checkpoint_path, metrics_path)
        except Exception:
            logger.exception("Failed to save model checkpoint or metrics")
            raise

    return model, metrics


def build_param_groups(model, lr: float, weight_decay: float, backbone_lr_scale: float):
    """Pretrained backbones get a smaller step than the randomly initialized heads."""
    backbone, head, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        elif name.startswith("video_model.") or "audio_encoder.backbone" in name:
            backbone.append(param)
        else:
            head.append(param)

    return [
        {"params": head, "lr": lr, "weight_decay": weight_decay},
        {"params": backbone, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, total_steps: int, warmup_fraction: float = 0.05):
    warmup = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train ACEVerifyModel")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training HDF5 file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test HDF5 file")
    parser.add_argument("--checkpoint-path", type=str, default="results/aceverify_final.pth", help="Where to save the checkpoint")
    parser.add_argument("--model", choices=["enhanced", "baseline"], default="enhanced", help="Architecture to train (default: enhanced)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Requested batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate for new heads (default: 2e-4)")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.05, help="LR multiplier for pretrained backbones (default: 0.05)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay (default: 0.05)")
    parser.add_argument("--train-n", type=int, default=1000, help="Balanced train samples per epoch (default: 1000)")
    parser.add_argument("--test-n", type=int, default=200, help="Balanced validation samples (default: 200)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--max-steps", type=int, default=0, help="Cap training steps per epoch (0 = full epoch); used for smoke tests")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient-norm clip (default: 1.0)")
    parser.add_argument("--pos-weight", type=float, default=None, help="BCE positive class weight; leave unset for the balanced sampler")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.2, help="Supervised contrastive weight (0 disables)")
    parser.add_argument("--artifact-weight", type=float, default=0.1, help="Patch artifact sparsity weight (0 disables)")
    parser.add_argument("--no-frequency", action="store_true", help="Ablate the DCT/SRM frequency stream")
    parser.add_argument("--no-motion", action="store_true", help="Ablate the patch temporal-incoherence stream")
    parser.add_argument("--no-audio", action="store_true", help="Ablate the audio spectrogram stream")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training")
    parser.add_argument("--no-auto-batch", action="store_true", help="Disable VRAM probe / OOM batch backoff")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the checkpoint (smoke tests)")
    parser.add_argument("--metrics-json", type=str, default=None, help="Also write the metric history to this JSON path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    seed_everything(args.seed)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        logger.info(
            "Training config: model=%s device=%s batch_size=%d lr=%s epochs=%d train_n=%d test_n=%d amp=%s seed=%d",
            args.model,
            device,
            args.batch_size,
            args.lr,
            args.epochs,
            args.train_n,
            args.test_n,
            not args.no_amp,
            args.seed,
        )

        model_kwargs = {}
        if args.model == "enhanced":
            model_kwargs = {
                "use_frequency": not args.no_frequency,
                "use_motion": not args.no_motion,
                "use_audio": not args.no_audio,
            }
        model = build_model(args.model, **model_kwargs).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model %s: %.1fM total params, %.1fM trainable",
            args.model,
            sum(p.numel() for p in model.parameters()) / 1e6,
            trainable / 1e6,
        )

        # The sampler draws an equal number of real and fake clips, so a pos_weight
        # above 1 would bias the decision threshold toward "Fake" for no reason.
        criterion = DeepfakeCriterion(
            label_smoothing=args.label_smoothing,
            contrastive_weight=args.contrastive_weight,
            artifact_weight=args.artifact_weight if args.model == "enhanced" else 0.0,
            pos_weight=args.pos_weight,
        ).to(device)

        optimizer = optim.AdamW(
            build_param_groups(model, args.lr, args.weight_decay, args.backbone_lr_scale)
        )

        steps_per_epoch = max(1, math.ceil(args.train_n / max(1, args.batch_size)))
        if args.max_steps:
            steps_per_epoch = min(steps_per_epoch, args.max_steps)
        scheduler = build_scheduler(optimizer, steps_per_epoch * args.epochs)

        os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)

        config = {
            "device": device,
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "checkpoint_path": args.checkpoint_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_path": args.test_path,
            "train_path": args.train_path,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "num_workers": args.num_workers,
            "max_steps": args.max_steps,
            "grad_clip": args.grad_clip,
            "amp": not args.no_amp,
            "auto_batch": not args.no_auto_batch,
            "resume": not args.no_resume,
            "save_checkpoint": not args.no_save,
        }

        _, metrics = train_model(config)

        if args.metrics_json:
            with open(args.metrics_json, "w") as f:
                json.dump({"model": args.model, "args": vars(args), "metrics": metrics}, f, indent=2, default=str)
            logger.info("Wrote metric history to %s", args.metrics_json)
    except Exception:
        logger.exception("Fatal error in training pipeline")
        raise


if __name__ == "__main__":
    main()
