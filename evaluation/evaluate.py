"""Standalone evaluation script that can run inside a pod.

Usage: python -m evaluation.evaluate --h5 /workspace/data/test.h5 --checkpoint /workspace/results/aceverify_final.pth
"""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aceverify.dataset import ACEDataset
from aceverify.model import load_from_checkpoint
from aceverify.train import build_model

logger = logging.getLogger(__name__)


def configure_logging(level="INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s | %(levelname)s | %(message)s')


def _equal_error_rate(labels, scores):
    """EER is the threshold-free operating point usually reported for deepfake detection."""
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(labels)[order]
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    false_negative = 1.0 - np.cumsum(labels) / positives
    false_positive = np.cumsum(1 - labels) / negatives
    crossing = np.nanargmin(np.abs(false_negative - false_positive))
    return float((false_negative[crossing] + false_positive[crossing]) / 2.0)


def evaluate(h5_path, checkpoint_path, batch_size=8, device=None, num_workers=0, model_name="auto", amp=True):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    use_amp = amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    ds = ACEDataset(h5_path, is_training=False)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logger.info("Evaluating %d samples from %s (batch_size=%d workers=%d)", len(ds), h5_path, batch_size, num_workers)

    try:
        if model_name == "auto":
            model, model_name = load_from_checkpoint(checkpoint_path, map_location=device)
            logger.info("Detected %s architecture in %s", model_name, checkpoint_path)
        else:
            model = build_model(model_name)
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
        model.to(device)
    except Exception:
        logger.exception("Failed to load checkpoint %s", checkpoint_path)
        raise

    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for i, (videos, specs, labels) in enumerate(loader):
            videos, specs = videos.to(device), specs.to(device)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                outputs = model(videos, specs)
            all_scores.extend(torch.sigmoid(outputs.float()).view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).tolist())
            if i % 10 == 0:
                logger.info("Eval step %d/%d (n=%d)", i, len(loader), len(all_labels))

    all_preds = [int(s > 0.5) for s in all_scores]
    df = pd.DataFrame({'label': all_labels, 'pred': all_preds, 'score': all_scores})
    out_csv = os.path.splitext(checkpoint_path)[0] + '_eval.csv'
    df.to_csv(out_csv, index=False)

    try:
        auc = float(roc_auc_score(all_labels, all_scores))
        ap = float(average_precision_score(all_labels, all_scores))
    except ValueError:
        auc = ap = float("nan")

    summary = {
        "n": len(all_labels),
        "model": model_name,
        "accuracy": float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0,
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": auc,
        "average_precision": ap,
        "eer": _equal_error_rate(all_labels, all_scores),
        "checkpoint": checkpoint_path,
        "h5": h5_path,
    }
    out_json = os.path.splitext(checkpoint_path)[0] + '_eval.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info('Saved evaluation predictions to %s', out_csv)
    logger.info(
        'Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f AUC=%.4f EER=%.4f (n=%d)',
        summary["accuracy"], summary["precision"], summary["recall"],
        summary["f1"], summary["auc"], summary["eer"], summary["n"],
    )
    logger.info('\n%s', classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], zero_division=0))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', choices=['auto', 'enhanced', 'baseline'], default='auto',
                        help="Architecture to load; 'auto' infers it from the checkpoint")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0, help="DataLoader workers (0 avoids h5py fork deadlocks)")
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()
    configure_logging(args.log_level)
    evaluate(
        args.h5,
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model,
        amp=not args.no_amp,
    )


if __name__ == '__main__':
    main()
