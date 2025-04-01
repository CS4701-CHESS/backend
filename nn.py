import torch
import torch.nn as nn
import torch.utils.data.dataset


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batch(x)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x


class Dataset(torch.utils.data.Dataset):
    # color is either white or black, the only type of positions that this dataset will hold
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
