import argparse
import os
import logging
import h5py
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
from .dataset import ACEDataset
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

def load_data(indices, path, n, training, shuffle_data=True, dataset_class=ACEDataset):
    sub_indices = []
    if len(indices) == 0:
        logger.debug("Loading data from %s (n=%s, training=%s)", path, n, training)
        num_each = n // 2
        all_labels = []

        try:
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    all_labels.append(f[key].attrs['label'])
        except Exception:
            logger.exception("Failed to read labels from HDF5 file: %s", path)
            raise

        all_labels = np.array(all_labels)
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
        logger.debug("Using provided indices for data loading from %s (n=%s, training=%s)", path, n, training)
        sub_indices = np.array(indices)

    if len(sub_indices) == 0:
        raise ValueError(f"No samples selected from {path}. Check file contents and requested n={n}.")
    if shuffle_data:
        np.random.shuffle(sub_indices)

    dataset = dataset_class(h5_path=path, indices=sub_indices, is_training=training)
    logger.debug("Built dataset with %d samples from %s", len(sub_indices), path)
    return dataset

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

    # Move model and criterion to the same device as inputs
    model = model.to(device)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

    test_dataset = load_data(test_indices, test_path, 200, False, shuffle_data=shuffle_data, dataset_class=dataset_class)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Resuming from dict checkpoint: %s", checkpoint_path)
        else:
            model.load_state_dict(checkpoint)
            logger.info("Resuming from state_dict: %s", checkpoint_path)
    except FileNotFoundError:
        logger.warning("No checkpoint found. Starting from pretrained ViT weights.")
    except Exception:
        logger.exception("Failed to load checkpoint from %s", checkpoint_path)
        raise
    
    # Train loop
    logger.info("Training start on %s", device)
    metrics = {}
    metrics["test_accuracies"] = []
    metrics["train_accuracies"] = []
    metrics["epochs"] = []
    for epoch in range(epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)
        train_dataset = load_data(train_indices, train_path, 1000, True, shuffle_data=shuffle_data, dataset_class=dataset_class)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        if len(train_loader) == 0:
            raise ValueError(
                f"Empty training loader for epoch {epoch + 1}. "
                f"Check train_path={train_path} and sampled dataset contents."
            )
        
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        try:
            for i, (videos, specs, labels) in enumerate(train_loader):
                videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                outputs = model(videos, specs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                total_loss += loss.item()
                if i % 10 == 0:
                    current_accuracy = 100 * (train_correct / train_total)
                    logger.info(
                        "Epoch [%d/%d], Step [%d], Loss: %.4f, Training Accuracy: %.2f%%",
                        epoch + 1,
                        epochs,
                        i,
                        loss.item(),
                        current_accuracy,
                    )
        except Exception:
            logger.exception("Training step failed in epoch %d", epoch + 1)
            raise

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * (train_correct / train_total)
        metrics["epochs"].append(epoch+1)
        metrics["train_accuracies"].append(train_accuracy)
        scheduler.step()

        # validation
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        try:
            with torch.no_grad():
                for videos, specs, labels in test_loader:
                    videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(videos, specs)
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels.extend(labels.view(-1).cpu().tolist())
        except Exception:
            logger.exception("Validation failed in epoch %d", epoch + 1)
            raise

        if test_total == 0:
            raise ValueError(
                f"No validation samples were evaluated in epoch {epoch + 1}. "
                f"Check test_path={test_path}."
            )

        test_accuracy = 100 * (test_correct / test_total)
        metrics["test_accuracies"].append(test_accuracy)

        logger.info("---Epoch %d Summary---", epoch + 1)
        logger.info("    Average Training Loss: %.4f", avg_train_loss)
        logger.info("    Training Accuracy: %.2f%%", train_accuracy)
        logger.info("    Test Accuracy: %.2f%%", test_accuracy)
        logger.info("------------------------------------------------")

    classification_report_final = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])

    logger.info("Final Classification Report:")
    logger.info("\n%s", classification_report_final)
    try:
        torch.save(model.state_dict(), checkpoint_path)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(checkpoint_path.replace('.pth', '_metrics.csv'), index=False)
        logger.info("Saved checkpoint to %s and metrics to %s", checkpoint_path, checkpoint_path.replace('.pth', '_metrics.csv'))
    except Exception:
        logger.exception("Failed to save model checkpoint or metrics")
        raise

    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Train ACEVerifyModel")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training HDF5 file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test HDF5 file")
    parser.add_argument("--checkpoint-path", type=str, default="results/aceverify_final.pth", help="Where to save the checkpoint (default: results/aceverify_final.pth)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    train_path = args.train_path
    test_path = args.test_path
    checkpoint_path = args.checkpoint_path
    epochs = args.epochs
    batch_size = args.batch_size

    try:
        # Hyperparameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = 5e-5
        logger.info(
            "Training config: device=%s batch_size=%d lr=%s epochs=%d",
            device,
            batch_size,
            learning_rate,
            epochs,
        )

        # Model
        model = ACEVerifyModel().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        # Ensure results directory exists (works inside a pod mounting /workspace)
        ckpt_dir = os.path.dirname(checkpoint_path) or "."
        os.makedirs(ckpt_dir, exist_ok=True)

        config = {
            "device": device,
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "checkpoint_path": checkpoint_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "test_path": test_path,
            "train_path": train_path,
        }

        train_model(config)
    except Exception:
        logger.exception("Fatal error in training pipeline")
        raise

if __name__ == '__main__':
    main()
