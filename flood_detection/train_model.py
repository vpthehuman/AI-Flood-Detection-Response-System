import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FloodDetectionModel
from data_preprocessing import FloodNetDataset, get_data_transforms
from sklearn.metrics import f1_score
import os

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FloodDetectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_transforms = get_data_transforms()

    data_dir = 'data/ColorMasks-FloodNetv1'
    train_dataset = FloodNetDataset(os.path.join(data_dir, 'ColorMask-TrainSet'), data_transforms['train'])
    val_dataset = FloodNetDataset(os.path.join(data_dir, 'ColorMask-ValSet'), data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                val_loss += dice_loss(outputs, masks).item()
                val_f1 += f1_score(masks.cpu().numpy().flatten(), (outputs > 0.5).cpu().numpy().flatten())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1/len(val_loader):.4f}')

    torch.save(model.state_dict(), 'flood_detection_model.pth')

if __name__ == '__main__':
    train_model()
