import torch
import torchvision.models as models
import torch.nn as nn

class FloodDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FloodDetectionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
