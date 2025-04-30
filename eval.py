from chess import Board
import helper
import torch
from nn import Model
import pickle
import numpy as np
import logging
import chess


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


white_model_path = "models/validation_optimized_model_white.pth"
white_mti_path = "models/vom_mti_white"
black_model_path = "models/validation_optimized_model_black.pth"
black_mti_path = "models/vom_mti_black"


def prepare_input(board: Board):

    is_white = board.turn == chess.WHITE

    matrix = helper.fen2vec(board.fen(), is_white)

    if matrix is None:
        logger.error(f"fen2vec returned None for board: {board.fen()}")

        return torch.zeros((14, 8, 8), dtype=torch.float32).unsqueeze(0)

    X_tensor = matrix.unsqueeze(0)
    return X_tensor


try:
    with open(white_mti_path, "rb") as file:
        move_to_int_white = pickle.load(file)
    with open(black_mti_path, "rb") as file:
        move_to_int_black = pickle.load(file)

    model = Model(num_classes=len(move_to_int_white))
    model.load_state_dict(
        torch.load(white_model_path, map_location=torch.device("cpu"))
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    int_to_move = {v: k for k, v in move_to_int_white.items()}
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


def quiescence_search(board, alpha, beta, qdepth=0, max_qdepth=8):
    """
    Search captures until reaching a "quiet" position to avoid horizon effect.
    """

    stand_pat = evaluate_board(board)

    if qdepth >= max_qdepth:
        return stand_pat

    if board.is_checkmate():
        return -100000 if board.turn else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures_and_checks = []
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            captures_and_checks.append((move, mvv_lva_score(board, move)))

    if not captures_and_checks:
        return stand_pat

    captures_and_checks.sort(key=lambda x: x[1], reverse=True)

    for move, _ in captures_and_checks:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, qdepth + 1, max_qdepth)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def minimax_alphabeta(board, depth=2, alpha=-float("inf"), beta=float("inf")):
    """
    Minimax algorithm with alpha-beta pruning and quiescence search.
    """
    if depth == 0 or board.is_game_over():

        return None, quiescence_search(board, alpha, beta)

    bestMove = None
    if board.turn == chess.WHITE:
        maxEval = -float("inf")
        for move in order_moves(board):
            board.push(move)
            _, curEval = minimax_alphabeta(board, depth - 1, alpha, beta)
            board.pop()
            if curEval > maxEval:
                maxEval = curEval
                bestMove = move
            alpha = max(alpha, curEval)
            if beta <= alpha:
                break
        return bestMove, maxEval
    else:
        minEval = float("inf")
        for move in order_moves(board):
            board.push(move)
            _, curEval = minimax_alphabeta(board, depth - 1, alpha, beta)
            board.pop()
            if curEval < minEval:
                minEval = curEval
                bestMove = move
            beta = min(beta, curEval)
            if beta <= alpha:
                break
        return bestMove, minEval


def get_top_n_moves(board: chess.Board, n=4):
    """
    Get the top N legal moves predicted by the neural network model.
    Returns a list of (move_uci, probability) tuples.
    """
    try:
        if model is None:
            logger.error("Model not initialized")
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


def neural_minimax(board, depth=3, alpha=-float("inf"), beta=float("inf"), top_n=4):
    """
    Minimax that uses neural network predictions with quiescence search.
    """
    if depth == 0 or board.is_game_over():

        return None, quiescence_search(board, alpha, beta)

    top_moves = get_top_n_moves(board, n=top_n)

    if not top_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, evaluate_board(board)
        move_list = [(move.uci(), 0.0) for move in legal_moves]
    else:
        move_list = top_moves

    bestMove = None
    if board.turn == chess.WHITE:
        maxEval = -float("inf")
        for move_uci, _ in move_list:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                _, curEval = neural_minimax(board, depth - 1, alpha, beta, top_n)
                board.pop()
                if curEval > maxEval:
                    maxEval = curEval
                    bestMove = move
                alpha = max(alpha, curEval)
                if beta <= alpha:
                    break
            except Exception as e:
                logger.error(f"Error in neural_minimax for move {move_uci}: {e}")
                if board.move_stack:
                    board.pop()
        return bestMove, maxEval
    else:
        minEval = float("inf")
        for move_uci, _ in move_list:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                _, curEval = neural_minimax(board, depth - 1, alpha, beta, top_n)
                board.pop()
                if curEval < minEval:
                    minEval = curEval
                    bestMove = move
                beta = min(beta, curEval)
                if beta <= alpha:
                    break
            except Exception as e:
                logger.error(f"Error in neural_minimax for move {move_uci}: {e}")
                if board.move_stack:
                    board.pop()
        return bestMove, minEval


def minimax_top_moves(board, top_moves, depth=2, use_neural_minimax=False, top_n=4):
    """
    Run minimax on each of the top N moves and select the best one.
    Returns the best move (in UCI format) and its evaluation.

    If use_neural_minimax is True, uses neural network to prune the search at every level.
    """
    try:
        if not top_moves:
            logger.warning("No top moves provided for minimax search")
            return None, None

        best_move = None
        best_eval = float("-inf") if board.turn == chess.WHITE else float("inf")

        for move_uci, _ in top_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)

                if use_neural_minimax:
                    _, eval_score = neural_minimax(
                        board,
                        depth=depth - 1,
                        alpha=-float("inf"),
                        beta=float("inf"),
                        top_n=top_n,
                    )
                else:
                    _, eval_score = minimax_alphabeta(
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

        return best_move, best_eval

    except Exception as e:
        logger.error(f"Error in minimax_top_moves: {e}")
        return None, None


def predict_move_fen(
    fen,
    depth=8,
    top_n=5,
    use_neural_minimax=True,
    first_move_all_legal=False,
    max_qdepth=6,
):
    """
    Predict the best move for a given FEN string using the hybrid approach with quiescence search.

    Parameters:
    - depth: Regular search depth (default 8)
    - top_n: Number of moves to consider from neural network (default 5)
    - use_neural_minimax: When True, the neural network guides the search at every level
    - first_move_all_legal: When True, consider ALL legal moves at the first level
    - max_qdepth: Maximum depth for quiescence search (default 6)
    """
    try:

        global QUIESCENCE_MAX_DEPTH
        QUIESCENCE_MAX_DEPTH = max_qdepth

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

        best_move, eval_score = minimax_top_moves(
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

        return best_move, eval_score

    except Exception as e:
        logger.error(f"Error in predict_move_fen: {e}")
        return None, None
