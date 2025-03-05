import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset
import nn as defs
from tqdm import tqdm
from matplotlib import pyplot as plt

# Note that the dataset is not included in the github due to size.
# Also note that two neural networks may be required if we want the model
# to play as black and white

# dataset gets killed by kernel for a million reads, might need to move to colab
dataset = defs.Dataset("data/chessData.csv", isWhite=True, read=500000)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
print("Successfully Loaded Datasets!\n")
# print(dataset.x[0])
# print(dataset.y[0])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=True)

LAWChess = defs.Model(32, 4)
# add to nvidia GPU, might not work on macs
LAWChess = LAWChess.to("cuda")

# define loss
criterion = nn.MSELoss()

# define an optimizer
optimizer = torch.optim.Adam(LAWChess.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=10,
    threshold=0.0001,
    threshold_mode="abs",
)
num_epochs = 1000


train_losses = []
train_lossAvg = 0
train_numBatch = len(train_loader)
test_losses = []
test_lossAvg = 0
test_numBatch = len(test_loader)

# train the model
print("Training:")
with tqdm(total=num_epochs) as progress_bar:
    for epoch in range(num_epochs):

        train_lossAvg = 0
        test_lossAvg = 0
        progress_bar.update(1)

        # train
        LAWChess.train()
        for x, y in train_loader:
            x = x.to("cuda")
            y = y.to("cuda")

            optimizer.zero_grad()
            pred = LAWChess(x)
            loss = criterion(pred, y)
            loss.backward()

            train_lossAvg += loss.item()

            # update parameters with gd
            optimizer.step()

        # validate
        LAWChess.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to("cuda")
                y = y.to("cuda")

                pred = LAWChess(x)
                loss = criterion(pred, y)

                test_lossAvg += loss.item()

        scheduler.step(train_lossAvg / train_numBatch)
        train_losses.append(train_lossAvg / train_numBatch)
        test_losses.append(test_lossAvg / test_numBatch)

# save model
torch.save(LAWChess.state_dict(), "model.pth")

# visualize data
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
