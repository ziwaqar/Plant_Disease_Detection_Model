import torch
import torch.nn as nn

class PlantNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantNet, self).__init__()
        # layer 1: input 128x128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # layer 2: output 64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # layer 3: output 32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5) # prevent overfitting
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # flatten for fc layers
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x