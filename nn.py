import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset
import helper
import pandas as pd
import numpy as np
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
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
