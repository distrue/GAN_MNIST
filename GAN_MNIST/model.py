import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class GAN_MNIST_GENERATOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh(),
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = x.view(x.size(0), 100)
        x = self.model(x)
        return x

class GAN_MNIST_DISCRIMINATOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.DropoutRate = 0.3
        
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(self.DropoutRate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.DropoutRate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.DropoutRate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 784)
        x = self.model(x)
        return x
