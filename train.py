import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset
import nn as defs

# Note that the dataset is not included in the github due to size.
# Also note that two neural networks may be required if we want the model
# to play as black and white

dataset = defs.Dataset("data/chessData.csv", isWhite=True, read=100)
print("Successfully Loaded Dataset!\n")
print(dataset.x[0])
# print(dataset.y[0])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

model = defs.Model()
# add to nvidia GPU, might not work on macs
model = model.to("cuda")

# define loss
criterion = nn.MSELoss()

# define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
for epoch in range(100):
    for batch in dataset:
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        # print loss
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # update parameters with gd
        optimizer.step()


# save model
torch.save(model.state_dict(), "model.pth")
