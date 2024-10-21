from typing import Optional
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage import io


transforms = {
    'train' : T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val' : T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def get_data_transforms():
    return transforms


class FloodDataset(torch.utils.data.Dataset):
  """
  This is to use with array of files and array of labels as an input
  """
  def __init__(self, X: list[str], y: Optional[list[int]], transform=None):
    self.X = X
    self.y = y
    self.transform = transform
    if y is not None:
       assert len(X) == len(y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx: int):
    image = io.imread(self.X[idx])
    if self.transform is not None:
        image = self.transform(image)
    if self.y is None:
        return (image, idx)
    return (image, self.y[idx])



# class FloodNetDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.images = [f for f in os.listdir(root_dir) if f.endswith('.png')]
#         self.color_map = {
#             (0, 0, 0): 0,  # background
#             (255, 0, 0): 1,  # building-flooded
#             (180, 120, 120): 2,  # building-non-flooded
#             (160, 150, 20): 3,  # road-flooded
#             (140, 140, 140): 4,  # road-non-flooded
#             (61, 230, 250): 5,  # water
#             (0, 82, 255): 6,  # tree
#             (255, 0, 245): 7,  # vehicle
#             (255, 235, 0): 8,  # pool
#             (4, 250, 7): 9  # grass
#         }

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.images[idx])
#         image = Image.open(img_name)

#         # Convert image to numpy array
#         image_np = np.array(image)

#         # Create a mask for flooded areas (building-flooded and road-flooded)
#         flood_mask = np.logical_or(
#             np.all(image_np == [255, 0, 0], axis=-1),
#             np.all(image_np == [160, 150, 20], axis=-1)
#         )

#         # Convert mask to tensor
#         flood_mask = torch.tensor(flood_mask, dtype=torch.float32)

#         if self.transform:
#             image = self.transform(image)

#         return image, flood_mask