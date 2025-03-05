import nn as defs
import torch
import helper

# load in trained model
model = defs.Model(32, 4)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# make prediction on input
pred = model(
    (
        helper.fen2vec(
            "r3b2k/p5p1/4pq2/1p1p4/2n2P2/P2B4/1P2Q2P/1K1R2R1 w - - 4 26", isWhite=True
        )
    ).unsqueeze(0)
)
print(f"Prediction: {pred}")
