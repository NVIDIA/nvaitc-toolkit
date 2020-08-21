import torch.nn as nn
import torch.nn.functional as F

from model.network.nvidiaresnet import build_resnet


class MyCNN(nn.Module):
    name = 'alex'

    def __init__(self):
        super().__init__()
        self.avg_pool_dim = 64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(64 * 1 * 1, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.to_last_conv(x)

    def to_last_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.mpool1(x)
        self.features_conv = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mpool1(x)
        self.features_conv = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        h = x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = x.view(x.size(0), 64 * 1 * 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def NVIDIAResnet18():
    return build_resnet('resnet18', 'classic')

def NVIDIAResnet50():
    return build_resnet('resnet50', 'classic')

NVIDIAResnet18.name = 'resnet18'
NVIDIAResnet50.name = 'resnet50'
