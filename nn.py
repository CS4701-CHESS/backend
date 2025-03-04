import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset
import helper
import pandas as pd
import numpy as np
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.selu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = F.selu(x)
        return x


class Dataset(torch.utils.data.Dataset):
    # color is either white or black, the only type of positions that this dataset will hold
    def __init__(self, csv_file, isWhite, read=100000):

        # read only a subset of the data
        raws = pd.read_csv(csv_file, nrows=read)

        # process data using funcs
        x = []
        y = []

        # 7000 items/ sec, about 2.2 min for a million datapoints
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
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # likely unneccesary
    # def collate_fn(self, batch):
    #    pass
