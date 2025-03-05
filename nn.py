import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset
import helper
import pandas as pd
import numpy as np
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, conv_size, conv_depth):
        super(Model, self).__init__()
        self.convs = []
        self.bns = []
        self.conv_depth = conv_depth
        self.conv_size = conv_size

        self.inputConv = nn.Conv2d(14, conv_size, kernel_size=3, stride=1, padding=1)
        for i in range(conv_depth):
            self.convs.append(
                nn.Conv2d(conv_size, conv_size, kernel_size=3, stride=1, padding=1)
            )
            self.convs.append(
                nn.Conv2d(conv_size, conv_size, kernel_size=3, stride=1, padding=1)
            )
            self.bns.append(nn.BatchNorm2d(conv_size))
            self.bns.append(nn.BatchNorm2d(conv_size))

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(conv_size * 64, 64)
        self.lin2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.inputConv(x)
        for i in range(self.conv_depth):
            previous = x
            x = self.convs[2 * i](x)
            x = self.bns[2 * i](x)
            x = F.relu(x)
            x = self.convs[2 * i + 1](x)
            x = self.bns[2 * i](x)
            x += previous
            x = F.relu(x)

        x = self.flatten(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x


class Dataset(torch.utils.data.Dataset):
    # color is either white or black, the only type of positions that this dataset will hold
    def __init__(self, csv_file, isWhite, read=100000):

        # read only a subset of the data
        raws = pd.read_csv(csv_file, nrows=read)

        # process data using funcs
        x = []
        y = []

        # 2000 items/ sec
        rows = raws.values.tolist()
        with tqdm(total=len(rows)) as progress_bar:
            for row in rows:
                progress_bar.update(1)

                tempX = helper.fen2vec(row[0], isWhite)
                tempY = helper.cp2val(row[1])
                if tempX is None or tempY == 0:  # remove draws and incorrect sides
                    continue

                x.append(tempX)
                y.append(tempY)

        self.x = torch.stack(x)
        self.y = (torch.Tensor(y)).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def collate_fn(self, batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y
