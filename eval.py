import nn as defs
import torch

# load in trained model
model = defs.Model()
model.load_state_dict(torch.load("model.pth"))

# make prediction on input
x = torch.rand(10)
pred = model(x)
print(f"Prediction Successful: {pred}")
