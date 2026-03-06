import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from dataset import ACEDataset
from model import ACEVerifyModel

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
steps = 4
learning_rate = 0.0001
epochs = 10
scalar = GradScaler('cuda')

# Prepare data
train_data_raw = ['data/train_data.h5']
test_data_raw = ['data/test_data.h5']

train_dataset = ACEDataset(train_data_raw)
test_dataset = ACEDataset(test_data_raw)
# num_138 = 0
# num_151 = 0
# for i in range(len(train_dataset)):
#     video, spec, label = train_dataset[i]
#     print(f"{spec.shape}, label: {label}")
#     if spec.shape[2] == 138:
#         num_138 += 1
#     else:
#         num_151 += 1
# print(f"num_138:{num_138}, num_151:{num_151}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
model = ACEVerifyModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

best_val_acc = 0.0
model_save_path = "aceverify_best.pth"

# Train loop
print(f"Training Start! On {device}!")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for i, (videos, specs, labels) in enumerate(train_loader):
        videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)

        with autocast('cuda'):
            outputs = model(videos, specs)
            loss = criterion(outputs, labels)
            loss = loss / steps

        scalar.scale(loss).backward()

        if (i+1) % steps == 0:
            scalar.step(optimizer)
            scalar.update()
            optimizer.zero_grad()

        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

        total_loss += loss.item()
        if i % 10 == 0:
            current_accuracy = 100 * (train_correct / train_total)
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}, Training Accuracy: {current_accuracy:.2f}%")

            if current_accuracy > best_val_acc:
                best_val_acc = current_accuracy
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': current_accuracy,
                }, model_save_path)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * (train_correct / train_total)

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
            all_preds.extend(preds.cpu().numpy())
            all_labelsl.extend(labels.cpu().numpy())

    test_accuracy = 100 * (test_correct / test_total)

    print(f"---Epoch {epoch+1} Summary---")
    print(f"    Average Training Loss: {avg_train_loss:.4f}")
    print(f"    Training Accuracy: {train_accuracy:.2f}%")
    print(f"    Test Accuracy: {test_accuracy:.2f}%")
    print("------------------------------------------------")


print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
torch.save(model.state_dict(), "aceverify_final.pth")
