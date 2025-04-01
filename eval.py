from chess import Board
import helper
import torch
from nn import Model
import pickle
import numpy as np

# Load the mapping for models being used
model_path = "models/validation_optimized_model.pth"
mti_path = "models/vom_mti"


def prepare_input(board: Board):
    matrix = helper.fen2vec(board.fen(), True)
    X_tensor = matrix.unsqueeze(0)
    return X_tensor


with open(mti_path, "rb") as file:
    move_to_int = pickle.load(file)

# Load the model
model = Model(num_classes=len(move_to_int))
model.load_state_dict(torch.load(model_path))
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}


# Function to make predictions
def predict_move(board: Board):
    X_tensor = prepare_input(board).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        logits = model(X_tensor)

    logits = logits.squeeze(0)  # Remove batch dimension

    probabilities = (
        torch.softmax(logits, dim=0).cpu().numpy()
    )  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move

    return None


def predict_move_fen(fen):
    return predict_move(Board(fen))
