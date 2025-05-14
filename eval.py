from chess import Board
import helper
import torch
from nn import Model
import pickle
import numpy as np
import logging
import chess
import random


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Model paths
white_model_path = "models/validation_optimized_model_white.pth"
white_mti_path = "models/vom_mti_white"
black_model_path = "models/validation_optimized_model_black.pth"
black_mti_path = "models/vom_mti_black"


def prepare_input(board: Board):
    """Prepare input tensor for the neural network based on current board state."""
    is_white = board.turn == chess.WHITE

    matrix = helper.fen2vec(board.fen(), is_white)

    if matrix is None:
        logger.error(f"fen2vec returned None for board: {board.fen()}")
        return torch.zeros((14, 8, 8), dtype=torch.float32).unsqueeze(0)

    X_tensor = matrix.unsqueeze(0)
    return X_tensor


# Initialize models
try:
    # Load white model data
    with open(white_mti_path, "rb") as file:
        move_to_int_white = pickle.load(file)
    int_to_move_white = {v: k for k, v in move_to_int_white.items()}

    # Load black model data
    with open(black_mti_path, "rb") as file:
        move_to_int_black = pickle.load(file)
    int_to_move_black = {v: k for k, v in move_to_int_black.items()}

    # Initialize white model
    white_model = Model(num_classes=len(move_to_int_white))
    white_model.load_state_dict(
        torch.load(white_model_path, map_location=torch.device("cpu"))
    )
    white_model.to("cuda" if torch.cuda.is_available() else "cpu")
    white_model.eval()

    # Initialize black model
    black_model = Model(num_classes=len(move_to_int_black))
    black_model.load_state_dict(
        torch.load(black_model_path, map_location=torch.device("cpu"))
    )
    black_model.to("cuda" if torch.cuda.is_available() else "cpu")
    black_model.eval()

except Exception as e:
    logger.critical(f"Failed to initialize models: {e}")
    white_model = None
    black_model = None
    move_to_int_white = {}
    move_to_int_black = {}
    int_to_move_white = {}
    int_to_move_black = {}


def predict_move(board: Board):
    """Predict the best move using the appropriate model based on whose turn it is."""
    try:
        is_white = board.turn == chess.WHITE
        model = white_model if is_white else black_model
        move_to_int = move_to_int_white if is_white else move_to_int_black
        int_to_move = int_to_move_white if is_white else int_to_move_black

        if model is None:
            logger.error(f"{'White' if is_white else 'Black'} model not initialized")
            return None, None

        X_tensor = prepare_input(board).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        with torch.no_grad():
            logits = model(X_tensor)

        logits = logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]

        if not legal_moves:
            logger.warning(f"No legal moves for board: {board.fen()}")
            return None, None

        best_move = None
        best_prob = -1
        best_move_idx = -1

        sorted_indices = np.argsort(probabilities)[::-1]
        for move_index in sorted_indices:
            if move_index >= len(int_to_move):
                continue
            move = int_to_move.get(move_index)
            if move and move in legal_moves_uci:
                best_move = move
                best_prob = probabilities[move_index]
                best_move_idx = move_index
                break

        eval_score = (best_prob * 2) - 1.0 if best_prob > 0 else 0

        if best_move is None:
            logger.warning("No valid moves found from model prediction")
            best_move = legal_moves[0].uci() if legal_moves else None
            eval_score = 0

        return best_move, eval_score

    except Exception as e:
        logger.error(f"Error in predict_move: {e}")

        try:
            legal_moves = list(board.legal_moves)
            return legal_moves[0].uci() if legal_moves else None, None
        except Exception as inner_e:
            logger.error(f"Error in fallback: {inner_e}")
            return None, None


# Piece Square Tables
PAWN_PST = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    10,
    -5,
    0,
    0,
    -5,
    10,
    0,
    0,
    10,
    10,
    20,
    20,
    10,
    10,
    0,
    5,
    20,
    20,
    30,
    30,
    20,
    20,
    5,
    10,
    20,
    20,
    30,
    30,
    20,
    20,
    10,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    80,
    80,
    80,
    80,
    80,
    80,
    80,
    80,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

KNIGHT_PST = [
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
    -40,
    -20,
    0,
    0,
    0,
    0,
    -20,
    -40,
    -30,
    0,
    10,
    15,
    15,
    10,
    0,
    -30,
    -30,
    5,
    15,
    20,
    20,
    15,
    5,
    -30,
    -30,
    0,
    15,
    20,
    20,
    15,
    0,
    -30,
    -30,
    5,
    10,
    15,
    15,
    10,
    5,
    -30,
    -40,
    -20,
    0,
    5,
    5,
    0,
    -20,
    -40,
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
]

BISHOP_PST = [
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    10,
    10,
    5,
    0,
    -10,
    -10,
    5,
    5,
    10,
    10,
    5,
    5,
    -10,
    -10,
    0,
    10,
    10,
    10,
    10,
    0,
    -10,
    -10,
    10,
    10,
    10,
    10,
    10,
    10,
    -10,
    -10,
    5,
    0,
    0,
    0,
    0,
    5,
    -10,
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
]

ROOK_PST = [
    0,
    0,
    5,
    10,
    10,
    5,
    0,
    0,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    5,
    10,
    10,
    10,
    10,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

QUEEN_PST = [
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    5,
    5,
    5,
    0,
    -10,
    -5,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    0,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -10,
    5,
    5,
    5,
    5,
    5,
    0,
    -10,
    -10,
    0,
    5,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
]

KING_PST = [
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -20,
    -30,
    -30,
    -40,
    -40,
    -30,
    -30,
    -20,
    -10,
    -20,
    -20,
    -20,
    -20,
    -20,
    -20,
    -10,
    20,
    20,
    0,
    0,
    0,
    0,
    20,
    20,
    20,
    30,
    10,
    0,
    0,
    10,
    30,
    20,
]


PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 100000,
}


def get_game_phase(board):
    """
    Returns a value between 0 and 1 indicating game phase
    0 = opening, 1 = endgame
    """
    total_material = sum(
        len(board.pieces(piece, color)) * PIECE_VALUES[piece]
        for color in [chess.WHITE, chess.BLACK]
        for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    )

    max_material = (
        4 * PIECE_VALUES[chess.KNIGHT]
        + 4 * PIECE_VALUES[chess.BISHOP]
        + 4 * PIECE_VALUES[chess.ROOK]
        + 2 * PIECE_VALUES[chess.QUEEN]
    )

    phase = 1.0 - min(1.0, total_material / max_material)
    return phase


def evaluate_board(board: chess.Board) -> int:
    """
    Evaluates the board position with multiple factors.
    Positive = White advantage. Negative = Black advantage.
    """
    if board.is_checkmate():
        return -100000 if board.turn else 100000

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = get_game_phase(board)

    score = 0

    for piece_type in PIECE_VALUES:
        white_squares = board.pieces(piece_type, chess.WHITE)
        black_squares = board.pieces(piece_type, chess.BLACK)

        score += PIECE_VALUES[piece_type] * (len(white_squares) - len(black_squares))

        pst = PIECE_SQUARE_TABLES.get(piece_type)
        if pst:
            for square in white_squares:
                score += pst[square]
            for square in black_squares:
                score -= pst[chess.square_mirror(square)]

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 50

    orig_turn = board.turn

    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))

    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))

    board.turn = orig_turn

    score += 5 * (white_mobility - black_mobility)

    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    white_doubled_pawns = 0
    black_doubled_pawns = 0

    for file in range(8):
        white_pawns_in_file = 0
        black_pawns_in_file = 0

        for rank in range(8):
            square = rank * 8 + file
            if square in white_pawns:
                white_pawns_in_file += 1
            if square in black_pawns:
                black_pawns_in_file += 1

        if white_pawns_in_file > 1:
            white_doubled_pawns += white_pawns_in_file - 1
        if black_pawns_in_file > 1:
            black_doubled_pawns += black_pawns_in_file - 1

    score -= 20 * (white_doubled_pawns - black_doubled_pawns)

    white_isolated_pawns = 0
    black_isolated_pawns = 0

    for file in range(8):
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)

        white_pawns_in_file = 0
        black_pawns_in_file = 0
        white_pawns_in_adjacent = 0
        black_pawns_in_adjacent = 0

        for rank in range(8):
            square = rank * 8 + file
            if square in white_pawns:
                white_pawns_in_file += 1
            if square in black_pawns:
                black_pawns_in_file += 1

        for adj_file in adjacent_files:
            for rank in range(8):
                square = rank * 8 + adj_file
                if square in white_pawns:
                    white_pawns_in_adjacent += 1
                if square in black_pawns:
                    black_pawns_in_adjacent += 1

        if white_pawns_in_file > 0 and white_pawns_in_adjacent == 0:
            white_isolated_pawns += 1
        if black_pawns_in_file > 0 and black_pawns_in_adjacent == 0:
            black_isolated_pawns += 1

    score -= 15 * (white_isolated_pawns - black_isolated_pawns)

    white_rooks = board.pieces(chess.ROOK, chess.WHITE)
    black_rooks = board.pieces(chess.ROOK, chess.BLACK)

    white_rooks_open_files = 0
    black_rooks_open_files = 0

    for file in range(8):
        file_has_white_pawn = False
        file_has_black_pawn = False

        for rank in range(8):
            square = rank * 8 + file
            if square in white_pawns:
                file_has_white_pawn = True
            if square in black_pawns:
                file_has_black_pawn = True

        for square in white_rooks:
            if file == square % 8:
                if not file_has_white_pawn:
                    white_rooks_open_files += 1
                if not file_has_white_pawn and not file_has_black_pawn:
                    white_rooks_open_files += 1

        for square in black_rooks:
            if file == square % 8:
                if not file_has_black_pawn:
                    black_rooks_open_files += 1
                if not file_has_white_pawn and not file_has_black_pawn:
                    black_rooks_open_files += 1

    score += 25 * (white_rooks_open_files - black_rooks_open_files)

    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    king_center_penalty = (1.0 - phase) * 50

    white_king_file = white_king_square % 8
    white_king_rank = white_king_square // 8
    black_king_file = black_king_square % 8
    black_king_rank = black_king_square // 8

    if white_king_file < 4:
        score -= int(king_center_penalty)
    if white_king_file > 5:
        score += int(king_center_penalty * 0.5)

    if black_king_file < 4:
        score += int(king_center_penalty)
    if black_king_file > 5:
        score -= int(king_center_penalty * 0.5)

    if phase > 0.5:
        white_king_center_distance = max(
            3 - white_king_file, white_king_file - 4
        ) + max(3 - white_king_rank, white_king_rank - 4)
        black_king_center_distance = max(
            3 - black_king_file, black_king_file - 4
        ) + max(3 - black_king_rank, black_king_rank - 4)

        endgame_king_activity = int(phase * 10)
        score -= endgame_king_activity * (
            white_king_center_distance - black_king_center_distance
        )

    return score


def mvv_lva_score(board, move):
    """
    Score captures for move ordering using MVV-LVA
    Higher score = better capture
    """
    if not board.is_capture(move):
        return 0

    to_square = move.to_square
    victim_piece = board.piece_at(to_square)

    if board.is_en_passant(move):
        victim_value = PIECE_VALUES[chess.PAWN]
    else:
        victim_value = PIECE_VALUES.get(victim_piece.piece_type, 0)

    from_square = move.from_square
    attacker_piece = board.piece_at(from_square)
    attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0)

    return victim_value * 10 - attacker_value


def order_moves(board):
    """
    Order moves for alpha-beta pruning efficiency.
    Order: 1) Captures (by MVV-LVA), 2) Checks, 3) Others
    """
    captures = []
    checks = []
    others = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append((move, mvv_lva_score(board, move)))
        elif board.gives_check(move):
            checks.append(move)
        else:
            others.append(move)

    captures.sort(key=lambda x: x[1], reverse=True)

    return [move for move, _ in captures] + checks + others


# Zobrist hashing implementation

# Initialize the Zobrist tables
random.seed(42)  # For reproducibility
ZOBRIST_PIECE_SQUARE = {}
ZOBRIST_CASTLING = {}
ZOBRIST_EP = {}
ZOBRIST_TURN = random.getrandbits(64)

# Initialize piece-square table
for piece in range(1, 7):  # 1=pawn, 2=knight, ..., 6=king
    for color in [True, False]:  # True=White, False=Black
        for square in range(64):
            ZOBRIST_PIECE_SQUARE[(piece, color, square)] = random.getrandbits(64)

# Initialize castling rights table
for i in range(16):  # 4 bits, representing KQkq
    ZOBRIST_CASTLING[i] = random.getrandbits(64)

# Initialize en passant table
for file in range(8):
    ZOBRIST_EP[file] = random.getrandbits(64)


def zobrist_hash(board):
    """
    Calculate Zobrist hash for the current board position
    """
    h = 0

    # Hash the pieces
    for square in range(64):
        piece = board.piece_at(chess.SQUARES[square])
        if piece:
            h ^= ZOBRIST_PIECE_SQUARE[(piece.piece_type, piece.color, square)]

    # Hash the castling rights
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    h ^= ZOBRIST_CASTLING[castling]

    # Hash the en passant square
    if board.ep_square:
        file = chess.square_file(board.ep_square)
        h ^= ZOBRIST_EP[file]

    # Hash the turn
    if board.turn == chess.WHITE:
        h ^= ZOBRIST_TURN

    return h


# Transposition Table implementation


class TranspositionEntry:
    """
    Entry in the transposition table.
    """

    # Flag types
    EXACT = 0  # Exact score
    LOWERBOUND = 1  # Beta cutoff, actual score might be higher
    UPPERBOUND = 2  # Alpha cutoff, actual score might be lower

    def __init__(self, hash_key, depth, score, flag, best_move=None):
        self.hash_key = hash_key  # Hash key for the position
        self.depth = depth  # Search depth
        self.score = score  # Evaluation score
        self.flag = flag  # Type of score (exact, lower bound, upper bound)
        self.best_move = best_move  # Best move found for this position


class TranspositionTable:
    """
    Transposition table for storing previously evaluated positions.
    """

    def __init__(self, size_mb=64):
        # Calculate the number of entries based on the size in MB
        # Each entry takes ~32 bytes
        self.max_entries = (size_mb * 1024 * 1024) // 32
        self.table = {}
        self.hits = 0
        self.collisions = 0
        self.stores = 0

    def store(self, hash_key, depth, score, flag, best_move=None):
        """Store a position in the transposition table."""
        # If we already have this position, update only if new entry has greater or equal depth
        if hash_key in self.table and self.table[hash_key].depth > depth:
            self.collisions += 1
            return

        entry = TranspositionEntry(hash_key, depth, score, flag, best_move)

        # If the table is full, we could implement a replacement strategy
        # For now, we'll just add the new entry
        if len(self.table) >= self.max_entries:
            # Simple replacement - remove a random entry
            # In a more sophisticated implementation, we could use age or depth
            if len(self.table) > 0:
                self.table.pop(next(iter(self.table)))

        self.table[hash_key] = entry
        self.stores += 1

    def lookup(self, hash_key):
        """Look up a position in the transposition table."""
        if hash_key in self.table:
            self.hits += 1
            return self.table[hash_key]
        return None

    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.collisions = 0
        self.stores = 0

    def info(self):
        """Return information about the transposition table usage."""
        return {
            "size": len(self.table),
            "max_size": self.max_entries,
            "hits": self.hits,
            "stores": self.stores,
            "collisions": self.collisions,
        }


# Initialize the transposition table
tt = TranspositionTable(size_mb=64)  # Default size of 64 MB


def quiescence_search(board, alpha, beta, depth=0, max_depth=5):
    """
    Quiescence search to evaluate captures until a quiet position is reached.
    This helps mitigate the horizon effect.
    """
    # Stand-pat score (evaluation if we don't make any moves)
    stand_pat = evaluate_board(board)

    # If the score is so good that it can't affect the result, return early
    if stand_pat >= beta:
        return beta

    # Update alpha if stand-pat is better
    if alpha < stand_pat:
        alpha = stand_pat

    # If we've reached the maximum quiescence depth, return the current evaluation
    if depth >= max_depth:
        return stand_pat

    # Only consider captures to find a "quiet" position
    captures = [move for move in board.legal_moves if board.is_capture(move)]

    # Order the moves based on MVV-LVA
    captures.sort(key=lambda move: mvv_lva_score(board, move), reverse=True)

    for move in captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, depth + 1, max_depth)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def minimax_alphabeta_with_tt(
    board, depth=3, alpha=-float("inf"), beta=float("inf"), is_root=True
):
    """
    Minimax algorithm with alpha-beta pruning and transposition table.
    """
    # Check for game over conditions
    if board.is_checkmate():
        return None, -100000 if board.turn else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return None, 0

    # Check if we've reached the maximum depth
    if depth <= 0:
        # Use quiescence search for a more stable evaluation
        eval_score = quiescence_search(board, alpha, beta)
        return None, eval_score

    # Calculate the Zobrist hash for the current position
    hash_key = zobrist_hash(board)

    # Check if the position is already in the transposition table
    tt_entry = tt.lookup(hash_key)
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == TranspositionEntry.EXACT:
            if is_root and tt_entry.best_move:
                return tt_entry.best_move, tt_entry.score
            return None, tt_entry.score
        elif tt_entry.flag == TranspositionEntry.LOWERBOUND:
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == TranspositionEntry.UPPERBOUND:
            beta = min(beta, tt_entry.score)

        if alpha >= beta:
            return None, tt_entry.score

    # Get the best move from the transposition table if available
    best_tt_move = tt_entry.best_move if tt_entry else None

    # Order moves - try the TT move first, then captures, checks, and other moves
    ordered_moves = []
    if best_tt_move:
        ordered_moves.append(best_tt_move)

    for move in order_moves(board):
        if move != best_tt_move:  # Avoid duplicating the TT move
            ordered_moves.append(move)

    best_move = None
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")

    if board.turn == chess.WHITE:
        for move in ordered_moves:
            board.push(move)
            _, score = minimax_alphabeta_with_tt(board, depth - 1, alpha, beta, False)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if beta <= alpha:
                break

        # Store the result in the transposition table
        if best_move:
            flag = TranspositionEntry.EXACT
            if best_score <= alpha:
                flag = TranspositionEntry.UPPERBOUND
            elif best_score >= beta:
                flag = TranspositionEntry.LOWERBOUND

            tt.store(hash_key, depth, best_score, flag, best_move)

        return best_move, best_score
    else:
        for move in ordered_moves:
            board.push(move)
            _, score = minimax_alphabeta_with_tt(board, depth - 1, alpha, beta, False)
            board.pop()

            if score < best_score:
                best_score = score
                best_move = move

            beta = min(beta, score)
            if beta <= alpha:
                break

        # Store the result in the transposition table
        if best_move:
            flag = TranspositionEntry.EXACT
            if best_score <= alpha:
                flag = TranspositionEntry.UPPERBOUND
            elif best_score >= beta:
                flag = TranspositionEntry.LOWERBOUND

            tt.store(hash_key, depth, best_score, flag, best_move)

        return best_move, best_score


def neural_minimax_with_tt(
    board, depth=3, alpha=-float("inf"), beta=float("inf"), top_n=4, is_root=True
):
    """
    Minimax that uses neural network predictions for move pruning with transposition table.
    """
    # Check for game over conditions
    if board.is_checkmate():
        return None, -100000 if board.turn else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return None, 0

    # Check if we've reached the maximum depth
    if depth <= 0:
        # Use quiescence search for a more stable evaluation
        eval_score = quiescence_search(board, alpha, beta)
        return None, eval_score

    # Calculate the Zobrist hash for the current position
    hash_key = zobrist_hash(board)

    # Check if the position is already in the transposition table
    tt_entry = tt.lookup(hash_key)
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == TranspositionEntry.EXACT:
            if is_root and tt_entry.best_move:
                return tt_entry.best_move, tt_entry.score
            return None, tt_entry.score
        elif tt_entry.flag == TranspositionEntry.LOWERBOUND:
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == TranspositionEntry.UPPERBOUND:
            beta = min(beta, tt_entry.score)

        if alpha >= beta:
            return None, tt_entry.score

    # Get top moves from neural network
    top_moves = get_top_n_moves(board, n=top_n)

    if not top_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, evaluate_board(board)
        move_list = [(move.uci(), 0.0) for move in legal_moves]
    else:
        move_list = top_moves

    # Get the best move from the transposition table if available
    best_tt_move = tt_entry.best_move if tt_entry else None

    # If TT move exists and is not in our move list, add it
    if best_tt_move:
        best_tt_move_uci = best_tt_move.uci()
        if best_tt_move_uci not in [m[0] for m in move_list]:
            move_list.insert(
                0, (best_tt_move_uci, 1.0)
            )  # Add at the beginning with high priority

    best_move = None
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")

    if board.turn == chess.WHITE:
        for move_uci, _ in move_list:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                _, score = neural_minimax_with_tt(
                    board, depth - 1, alpha, beta, top_n, False
                )
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            except Exception as e:
                logger.error(
                    f"Error in neural_minimax_with_tt for move {move_uci}: {e}"
                )
                if board.move_stack:
                    board.pop()

        # Store the result in the transposition table
        if best_move:
            flag = TranspositionEntry.EXACT
            if best_score <= alpha:
                flag = TranspositionEntry.UPPERBOUND
            elif best_score >= beta:
                flag = TranspositionEntry.LOWERBOUND

            tt.store(hash_key, depth, best_score, flag, best_move)

        return best_move, best_score
    else:
        for move_uci, _ in move_list:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                _, score = neural_minimax_with_tt(
                    board, depth - 1, alpha, beta, top_n, False
                )
                board.pop()

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:
                    break
            except Exception as e:
                logger.error(
                    f"Error in neural_minimax_with_tt for move {move_uci}: {e}"
                )
                if board.move_stack:
                    board.pop()

        # Store the result in the transposition table
        if best_move:
            flag = TranspositionEntry.EXACT
            if best_score <= alpha:
                flag = TranspositionEntry.UPPERBOUND
            elif best_score >= beta:
                flag = TranspositionEntry.LOWERBOUND

            tt.store(hash_key, depth, best_score, flag, best_move)

        return best_move, best_score


def get_top_n_moves(board: chess.Board, n=4):
    """
    Get the top N legal moves predicted by the neural network model.
    Returns a list of (move_uci, probability) tuples.
    """
    try:
        is_white = board.turn == chess.WHITE
        model = white_model if is_white else black_model
        int_to_move = int_to_move_white if is_white else int_to_move_black

        if model is None:
            logger.error(f"{'White' if is_white else 'Black'} model not initialized")
            return []

        X_tensor = prepare_input(board).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        with torch.no_grad():
            logits = model(X_tensor)

        logits = logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]

        if not legal_moves:
            logger.warning(f"No legal moves for board: {board.fen()}")
            return []

        top_moves = []
        sorted_indices = np.argsort(probabilities)[::-1]

        for move_index in sorted_indices:
            if len(top_moves) >= n:
                break

            if move_index >= len(int_to_move):
                continue

            move = int_to_move.get(move_index)
            if move and move in legal_moves_uci:
                top_moves.append((move, probabilities[move_index]))

        if len(top_moves) < min(n, len(legal_moves)):
            for move in legal_moves:
                move_uci = move.uci()
                if move_uci not in [m[0] for m in top_moves]:
                    top_moves.append((move_uci, 0.0))
                    if len(top_moves) >= n:
                        break

        return top_moves

    except Exception as e:
        logger.error(f"Error in get_top_n_moves: {e}")
        return []


def minimax_top_moves_with_tt(
    board, top_moves, depth=3, use_neural_minimax=True, top_n=4
):
    """
    Run minimax on each of the top N moves and select the best one, using transposition table.
    Returns the best move (in UCI format) and its evaluation.
    """
    try:
        if not top_moves:
            logger.warning("No top moves provided for minimax search")
            return None, None

        best_move = None
        best_eval = float("-inf") if board.turn == chess.WHITE else float("inf")

        # Clear transposition table before starting a new search
        tt.clear()

        for move_uci, _ in top_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)

                if use_neural_minimax:
                    _, eval_score = neural_minimax_with_tt(
                        board,
                        depth=depth - 1,
                        alpha=-float("inf"),
                        beta=float("inf"),
                        top_n=top_n,
                    )
                else:
                    _, eval_score = minimax_alphabeta_with_tt(
                        board, depth=depth - 1, alpha=-float("inf"), beta=float("inf")
                    )

                board.pop()

                if (board.turn == chess.WHITE and eval_score > best_eval) or (
                    board.turn == chess.BLACK and eval_score < best_eval
                ):
                    best_eval = eval_score
                    best_move = move_uci

            except Exception as e:
                logger.error(f"Error evaluating move {move_uci}: {e}")
                if board.move_stack:
                    board.pop()

        # Log transposition table statistics
        # logger.info(f"Transposition table stats: {tt.info()}")

        return best_move, best_eval

    except Exception as e:
        logger.error(f"Error in minimax_top_moves_with_tt: {e}")
        return None, None


def predict_move_fen(
    fen,
    depth=8,
    top_n=5,
    use_neural_minimax=True,
    first_move_all_legal=False,
):
    """
    Predict the best move for a given FEN string using the hybrid approach with transposition tables.
    """
    try:
        board = chess.Board(fen)

        if first_move_all_legal:
            legal_moves = list(board.legal_moves)
            top_moves = [(move.uci(), 0.0) for move in legal_moves]
            logger.info(f"Using all {len(top_moves)} legal moves for first ply")
        else:
            top_moves = get_top_n_moves(board, n=top_n)

        if not top_moves:
            logger.warning("No top moves found, falling back to basic prediction")
            move, eval_score = predict_move(board)
            return move, eval_score

        # Clear the transposition table before starting a new prediction
        tt.clear()

        best_move, eval_score = minimax_top_moves_with_tt(
            board,
            top_moves,
            depth=depth,
            use_neural_minimax=use_neural_minimax,
            top_n=top_n,
        )

        if best_move is None:
            logger.warning(
                "Minimax failed to find a good move, falling back to basic prediction"
            )
            move, eval_score = predict_move(board)
            return move, eval_score

        # Log transposition table statistics
        # logger.info(f"Final transposition table stats: {tt.info()}")

        return best_move, eval_score

    except Exception as e:
        logger.error(f"Error in predict_move_fen: {e}")
        return None, None
