import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ACEDataset
from model import ACEVerifyModel

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
learning_rate = 0.0001
epochs = 10

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train loop
print("Training Start!")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for i, (videos, specs, labels) in enumerate(train_loader):
        videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)

        outputs = model(videos, specs)
        loss = criterion(outputs, labels)

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

    # validation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for videos, specs, labels in test_loader:
            videos, specs, labels = videos.to(device), specs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(videos, specs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    test_accuracy = 100 * (test_correct / test_total)

    print(f"---Epoch {epoch+1} Summary---")
    print(f"    Average Training Loss: {avg_train_loss:.4f}")
    print(f"    Training Accuracy: {train_accuracy:.2f}%")
    print(f"    Test Accuracy: {test_accuracy:.2f}%")
    print("------------------------------------------------")


torch.save(model.state_dict(), "aceverify_final.pth")


def predict_video(video_tensor, audio_tensor):
    model.eval()
    with torch.no_grad():
        output = model(video_tensor.unsqueeze(0).to(device), audio_tensor.unsqueeze(0).to(device))
        prob = torch.sigmoid(output).item()
        return "Fake" if prob > 0.5 else "Real", prob