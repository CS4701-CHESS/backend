from chess import Board
import helper
import torch
from nn import Model
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load the mapping for models being used
model_path = "models/validation_optimized_model.pth"
mti_path = "models/vom_mti"


def prepare_input(board: Board):
    # Get the actual side from the board
    is_white = board.turn  # chess.WHITE is True, chess.BLACK is False

    # The fen2vec function now works regardless of side, but we still pass is_white for compatibility
    matrix = helper.fen2vec(board.fen(), is_white)

    # Add error handling in case something else goes wrong
    if matrix is None:
        logger.error(f"fen2vec returned None for board: {board.fen()}")
        # Create a fallback zero tensor with the right dimensions
        return torch.zeros((14, 8, 8), dtype=torch.float32).unsqueeze(0)

    X_tensor = matrix.unsqueeze(0)
    return X_tensor


try:
    with open(mti_path, "rb") as file:
        move_to_int = pickle.load(file)

    # Load the model
    model = Model(num_classes=len(move_to_int))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode

    int_to_move = {v: k for k, v in move_to_int.items()}
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    model = None
    move_to_int = {}
    int_to_move = {}


def predict_move(board: Board):
    try:
        if model is None:
            logger.error("Model not initialized")
            return None, None

        X_tensor = prepare_input(board).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        with torch.no_grad():
            logits = model(X_tensor)

        logits = logits.squeeze(0)  # Remove batch dimension

        probabilities = (
            torch.softmax(logits, dim=0).cpu().numpy()
        )  # Convert to probabilities
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]

        if not legal_moves:
            logger.warning(f"No legal moves for board: {board.fen()}")
            return None, None

        # Find the best legal move according to the model
        best_move = None
        best_prob = -1
        best_move_idx = -1

        sorted_indices = np.argsort(probabilities)[::-1]
        for move_index in sorted_indices:
            if move_index >= len(int_to_move):
                continue  # Skip if index is out of bounds
            move = int_to_move.get(move_index)
            if move and move in legal_moves_uci:
                best_move = move
                best_prob = probabilities[move_index]
                best_move_idx = move_index
                break

        # Calculate a simple evaluation based on the confidence of the model
        # Map the probability to a range like -1.0 to 1.0 where higher is better for the current player
        eval_score = (best_prob * 2) - 1.0 if best_prob > 0 else 0

        # Fallback: return first legal move if no predicted move is legal
        if best_move is None:
            logger.warning("No valid moves found from model prediction")
            best_move = legal_moves[0].uci() if legal_moves else None
            eval_score = 0  # Neutral evaluation for fallback move

        return best_move, eval_score

    except Exception as e:
        logger.error(f"Error in predict_move: {e}")
        # Fallback to first legal move
        try:
            legal_moves = list(board.legal_moves)
            return legal_moves[0].uci() if legal_moves else None, None
        except Exception as inner_e:
            logger.error(f"Error in fallback: {inner_e}")
            return None, None


# Modify predict_move_fen to return both move and evaluation
def predict_move_fen(fen):
    try:
        board = Board(fen)
        move, eval_score = predict_move(board)
        return move, eval_score
    except Exception as e:
        logger.error(f"Error in predict_move_fen: {e}")
        return None, None
