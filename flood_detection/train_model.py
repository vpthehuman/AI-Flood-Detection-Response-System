import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FloodDetectionModel
from data_preprocessing import FloodNetDataset, get_data_transforms
from sklearn.metrics import f1_score, roc_auc_score
import os

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FloodDetectionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_transforms = get_data_transforms()

    data_dir = 'data/ColorMasks-FloodNetv1'
    train_dataset = FloodNetDataset(os.path.join(data_dir, 'ColorMask-TrainSet'), data_transforms['train'])
    val_dataset = FloodNetDataset(os.path.join(data_dir, 'ColorMask-ValSet'), data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        f1 = f1_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_preds)
        print(f'Epoch {epoch+1}/{num_epochs}, F1: {f1:.4f}, AUC: {auc:.4f}')

    torch.save(model.state_dict(), 'flood_detection_model.pth')

if __name__ == '__main__':
    train_model()
