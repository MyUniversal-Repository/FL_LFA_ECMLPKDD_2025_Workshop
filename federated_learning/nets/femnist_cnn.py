import torch
import torch.nn as nn
import torch.nn.functional as F

class FEMNISTCNN(nn.Module):

    def __init__(self):
        super(FEMNISTCNN, self).__init__()

        # First Convolutional Block
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Output: 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 32 x 14 x 14
        )
        
        # Second Convolutional Block
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 64 x 7 x 7
        )
        
        # Third Convolutional Block
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 128 x 3 x 3
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Flattened input to 256 units
        self.fc2 = nn.Linear(256, 62)  # FEMNIST has 62 classes (26 lowercase, 26 uppercase, 10 digits)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)  # Output probabilities for 62 classes

        return x
