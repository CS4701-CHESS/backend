from chess import Board
import helper
import torch
from nn import Model
import pickle
import numpy as np
import logging
import chess

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

# Material values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 100000
}

# Piece-square tables (from White's perspective)
PAWN_PST = [
     0,  5,  5, -10, -10,  5,  5,  0,
     0, 10, -5,   0,   0, -5, 10,  0,
     0, 10, 10,  20,  20, 10, 10,  0,
     5, 20, 20,  30,  30, 20, 20,  5,
    10, 20, 20,  30,  30, 20, 20, 10,
    50, 50, 50,  50,  50, 50, 50, 50,
    80, 80, 80,  80,  80, 80, 80, 80,
     0,  0,  0,   0,   0,  0,  0,  0
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

ROOK_PST = [
     0,   0,   5,  10,  10,   5,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     5,  10,  10,  10,  10,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
]
KING_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20
]

# Mapping piece type to PST
PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST
}

def evaluate_board(board: chess.Board) -> int:
    """
    Evaluates the board position using material and piece-square tables.
    Positive = White advantage. Negative = Black advantage.
    """
    score = 0

    for piece_type in PIECE_VALUES:
        white_squares = board.pieces(piece_type, chess.WHITE)
        black_squares = board.pieces(piece_type, chess.BLACK)

        # Material
        score += PIECE_VALUES[piece_type] * (len(white_squares) - len(black_squares))

        # Positional (PST)
        pst = PIECE_SQUARE_TABLES.get(piece_type)
        if pst:
            for square in white_squares:
                score += pst[square]
            for square in black_squares:
                score -= pst[chess.square_mirror(square)]  # mirror for black

    return score
