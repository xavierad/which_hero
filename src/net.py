from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size: List[int], number_classes: int) -> None:
        super().__init__()
        self.conv1: nn.Module = nn.Conv2d(3, *input_size)
        self.pool: nn.Module = nn.MaxPool2d(2, 2)
        self.conv2: nn.Module = nn.Conv2d(6, 16, 5)
        self.fc1: nn.Module = nn.Linear(16 * 5 * 5, 120)
        self.fc2: nn.Module = nn.Linear(120, 84)
        self.fc3: nn.Module = nn.Linear(84, number_classes)

    def forward(self, x):
        x: torch.Tensor = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x