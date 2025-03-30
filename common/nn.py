# -- IMPORTS --

import os
os.chdir('/workplace/flowmind/')

# torch imports
import torch.nn as nn
import torch.nn.functional as F

# others
import csv
import sys

# sets csv limit
csv.field_size_limit(sys.maxsize)

# -- CNN architecture --     
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

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
            #nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.projection = nn.Sequential(
            nn.Linear(120,120),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(120,30),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
# -- fine tuning using linear classifier --
class MLP(nn.Module):
    def __init__(self, output_size: int, input_size=120):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.fc(x)
        return y
    