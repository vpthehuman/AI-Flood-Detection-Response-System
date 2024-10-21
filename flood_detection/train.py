from typing import Optional
import copy
import datetime
import glob
import os
import time
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

from PIL import Image
from skimage import io
from skimage.transform import resize as io_resize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from model import FloodDetectionModel
from dataset import FloodDataset, get_data_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# load dataset
flooded_files = glob.glob("../data/processed/train/labeled/flooded/img/*.jpg")
flooded_y = [1]*len(flooded_files)

non_flooded_files = glob.glob("../data/processed/train/labeled/non-flooded/img/*.jpg")
non_flooded_y = [0]*len(non_flooded_files)

X = flooded_files + non_flooded_files
y = flooded_y + non_flooded_y

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.5, stratify=y)
print(len(X_train), Counter(y_train), len(X_val), Counter(y_val))





# create sampler
unique_labels, counts = np.unique(y_train, return_counts=True)
class_weights = [1/c for c in counts]
sample_weights = [0] * len(y_train)
for idx, lbl in enumerate(y_train):
    sample_weights[idx] = class_weights[lbl]
sampler_train = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

transforms = get_data_transforms()
train_set = FloodDataset(X_train, y_train, transform=transforms['train'])
val_set = FloodDataset(X_val, y_val, transform=transforms['val'])

BATCH_SIZE = 64
dataloaders = {
    'train': torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler_train),
    'val': torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
}
dataset_size = {
    'train':len(train_set),
    'val':len(val_set),
}





## Load unlabeled data
unlabeled_files = glob.glob("../data/processed/train/unlabeled/img/*.jpg")

# Adding validation set to unlabeled data
for mode in ['validation', 'test']:
    unlabeled_files += glob.glob(f"../data/processed/{mode}/img/*.jpg")

unlabeled_files.sort()
pseudo_labels = np.array([-1]*len(unlabeled_files))

unlabeled_set = FloodDataset(unlabeled_files, None, transform=transforms['val'])
unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

dataloaders['unlabeled'] = unlabeled_loader
dataset_size['unlabeled'] = len(unlabeled_set)





# >>> hyperparameters >>>
lr = 0.01
num_epochs = 100
start_alpha_from = 5
reach_max_alpha_in = 15
# <<< hyperparameters <<<

model = FloodDetectionModel()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters())
scheduler = None





alphas = np.linspace(0, 1, reach_max_alpha_in - start_alpha_from)

since = time.time()
current_time = datetime.datetime.now()
save_time_prefix = f"{current_time.year}{current_time.month}{current_time.day}_{current_time.hour}{current_time.minute}{current_time.second}"
model_save_path = None

writer = SummaryWriter()
os.makedirs("checkpoints", exist_ok=True)

best_model_wts = copy.deepcopy(model.state_dict())
best_f1 = 0.0

for epoch in range(num_epochs):

    if epoch < start_alpha_from:
        alpha = 0
    elif epoch - start_alpha_from >= len(alphas):
        alpha = alphas[-1]
    else:
        alpha = alphas[max(0,epoch - start_alpha_from)]

    for phase in ['train', 'unlabeled', 'val']:

        if alpha == 0 and phase == 'unlabeled':
            continue
        if phase in ['train', 'unlabeled']:
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # for confusion matrix
        epoch_preds = []
        epoch_lbls = []

        for batch_id, (img, lbl) in enumerate(tqdm(dataloaders[phase], desc=f"Epoch {epoch}/{num_epochs}, {phase}, alpha:{alpha:.2f}")):
            img = img.to(device)
            if phase in ['train', 'val']:
                lbl = lbl.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase in ['train', 'unlabeled']):
                output = model(img)
                preds = torch.argmax(output, 1)

                if phase in ['train', 'val']:
                    loss = criterion(output, lbl)
                else:
                    loss = alpha * criterion(output, torch.tensor(pseudo_labels[lbl], dtype=torch.int64, device=device))

                if phase in ['train', 'val']:
                    epoch_preds.extend(preds.detach().cpu())
                    epoch_lbls.extend(lbl.detach().cpu())

                if phase in ['train', 'unlabeled']:
                    loss.backward()
                    optimizer.step()

            if phase in ['train', 'val']:
                writer.add_scalars(f'Step-Loss/{phase}', {'loss':loss.item(), 'alpha':alpha}, (len(dataloaders[phase])*epoch)+batch_id)
                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(preds == lbl.data)

        # if phase in ['train', 'val']:
        #   print(confusion_matrix(epoch_lbls, epoch_preds), np.array([0, 1]))

        if scheduler is not None and phase == 'train':
            scheduler.step()

        if phase in ['train', 'val']:
            epoch_f1 = f1_score(epoch_lbls, epoch_preds)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

        if phase in ['train', 'val']:
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalars(f'F1/{phase}', {'f1':epoch_f1, 'alpha':alpha}, epoch)
            writer.add_scalars(f'Accuracy/{phase}', {'accu':epoch_acc, 'alpha':alpha}, epoch)

        print(f'Epoch {epoch}/{num_epochs}, {phase} Loss: {epoch_loss:.4f}, Acc:{epoch_acc:.4f}, F1:{epoch_f1:.4f}')

        if phase == 'train' and epoch >= start_alpha_from - 1:
            model.eval()
            for img, lbl in tqdm(dataloaders['unlabeled'], desc="Predicting pseudo labels"):
                img = img.to(device)
                preds = torch.argmax(model(img), 1)
                pseudo_labels[lbl] = preds.detach().cpu()

        if phase == 'val' and epoch_f1 > best_f1:
            best_epoch = epoch
            best_f1 = epoch_f1
            best_model_wts = copy.deepcopy(model.state_dict())

            if model_save_path:
                os.remove(model_save_path)
            model_save_path = f"checkpoints/{save_time_prefix}_epoch{epoch:03d}_fscore{best_f1:.3f}.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)

time_elapsed = time.time() - since
writer.close()

torch.save({
    'epoch': num_epochs - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f"checkpoints/{save_time_prefix}_last.pt")
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Saved model is from {best_epoch} epoch with f1 {best_f1}')
model.load_state_dict(best_model_wts)