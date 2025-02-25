# Note that the dataset is not included in the github due to size

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset
import nn as defs

x = torch.randn(100, 10)
y = torch.randn(100, 1)

dataset = defs.Dataset(x, y)
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
