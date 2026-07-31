"""Standalone evaluation script that can run inside a pod.

Usage: python -m ops.evaluate --h5 /workspace/data/test.h5 --checkpoint /workspace/results/aceverify_final.pth
"""
import argparse
import logging
import os
import torch
import pandas as pd
from aceverify.dataset import ACEDataset
from aceverify.model import ACEVerifyModel

logger = logging.getLogger(__name__)

def configure_logging(level="INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s | %(levelname)s | %(message)s')


def evaluate(h5_path, checkpoint_path, batch_size=8, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    ds = ACEDataset(h5_path, is_training=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = ACEVerifyModel()
    model.to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
    except Exception:
        logger.exception("Failed to load checkpoint %s", checkpoint_path)
        raise

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos, specs, labels in loader:
            videos, specs = videos.to(device), specs.to(device)
            outputs = model(videos, specs)
            preds = (torch.sigmoid(outputs) > 0.5).long().view(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.view(-1).tolist())

    df = pd.DataFrame({'label': all_labels, 'pred': all_preds})
    out_csv = os.path.splitext(checkpoint_path)[0] + '_eval.csv'
    df.to_csv(out_csv, index=False)
    logger.info('Saved evaluation predictions to %s', out_csv)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()
    configure_logging(args.log_level)
    evaluate(args.h5, args.checkpoint, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
