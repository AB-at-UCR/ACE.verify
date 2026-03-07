import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report


def load_data(path, n, training):
    num_each = n // 2
    all_labels = []

    with h5py.File(path, 'r') as f:
        for key in f.keys():
            all_labels.append(f[key].attrs['label'])

    all_labels = np.array(all_labels)
    real_indices = np.where(all_labels == 0)[0]
    fake_indices = np.where(all_labels == 1)[0]

    sel_real = np.random.choice(real_indices, min(len(real_indices), num_each), replace=False)
    sel_fake = np.random.choice(fake_indices, min(len(fake_indices), num_each), replace=False)

    sub_indices = np.concatenate([sel_real, sel_fake])
    np.random.shuffle(sub_indices)

    dataset = ACEDataset(h5_path=path, indices=sub_indices, is_training=training)
    return dataset

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
learning_rate = 5e-5
epochs = 10

test_dataset = load_data(test_path, 200, False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

# Model
model = ACEVerifyModel().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

checkpoint_path = "aceverify_final.pth" 
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resuming from dict checkpoint: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Resuming from state_dict: {checkpoint_path}")
except FileNotFoundError:
    print("No checkpoint found. Starting from pretrained ViT weights.")

# Train loop
print(f"Training Start! On {device}!")
for epoch in range(epochs):
    train_dataset = load_data(train_path, 1000, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for i, (videos, specs, labels) in enumerate(train_loader):
        videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)

        outputs = model(videos, specs)
        loss = criterion(outputs, labels.float().view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

        total_loss += loss.item()
        if i % 10 == 0:
            current_accuracy = 100 * (train_correct / train_total)
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}, Training Accuracy: {current_accuracy:.2f}%")

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * (train_correct / train_total)
    scheduler.step()

    # validation
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos, specs, labels in test_loader:
            videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(videos, specs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = 100 * (test_correct / test_total)

    print(f"---Epoch {epoch+1} Summary---")
    print(f"    Average Training Loss: {avg_train_loss:.4f}")
    print(f"    Training Accuracy: {train_accuracy:.2f}%")
    print(f"    Test Accuracy: {test_accuracy:.2f}%")
    print("------------------------------------------------")


print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
torch.save(model.state_dict(), "aceverify_final.pth")