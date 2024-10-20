import torch
import torch.nn as nn
import torchvision


class FloodDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        final_layer_in = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(final_layer_in, 2)

    def forward(self, x):
        return self.model(x)


# class FloodDetectionModel(nn.Module):
#     def __init__(self):
#         super(FloodDetectionModel, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         self.resnet.fc = nn.Identity()

#         self.upsample = nn.Sequential(
#             nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)

#         x = self.upsample(x)
#         return x
