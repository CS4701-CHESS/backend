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
            "2r4r/p4pk1/1p2p1p1/4Nn1q/3P2R1/P1P2Q2/5PP1/3R2K1 w - - 7 27",
            isWhite=True,
        )
    ).unsqueeze(0)
)
print(f"Prediction: {pred}")
