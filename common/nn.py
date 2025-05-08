'''

This file defines the neural networks used.

Author: Jakub Čoček (xcocek00)

'''

# -- IMPORTS --

import os
path_to_module = os.path.abspath(os.path.join('..', 'flowmind'))
sys.path.append(path_to_module)

# torch imports
import torch.nn as nn
import torch.nn.functional as F

# others
import csv
import sys

# sets csv limit
csv.field_size_limit(sys.maxsize)

# -- CNN architecture for 32x32 FlowPics --     
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder f()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
        )

        # projection head g()
        self.projection = nn.Sequential(
            nn.Linear(120,120),
            nn.ReLU(),
            nn.Linear(120,30),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
# -- CNN architecture for 64x64 FlowPics --     
class CNN_64(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder f()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(2704, 120),
            nn.ReLU(),
        )

        # projection head g()
        self.projection = nn.Sequential(
            nn.Linear(120,120),
            nn.ReLU(),
            nn.Linear(120,30),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
# -- fine tuning using linear classifier (also called classification head) --
class MLP(nn.Module):
    def __init__(self, output_size: int, input_size=120):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.fc(x)
        return y
    