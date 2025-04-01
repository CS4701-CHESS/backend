import nn as defs
import torch
import helper
import chess

# load in trained model
model = defs.Model(32, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()


# Minimax without alpha-beta pruning
def base_minimax(board, depth=1, isWhite=True):
    if depth == 0:
        eval_tensor = model(helper.fen2vec(board.fen(), isWhite).unsqueeze(0))
        return None, eval_tensor.item()

    if isWhite:
        bestMove = None
        maxEval = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            _, curEval = base_minimax(board, depth - 1, False)
            board.pop()
            if curEval > maxEval:
                maxEval = curEval
                bestMove = move
        return bestMove, maxEval
    else:
        bestMove = None
        minEval = float("inf")
        for move in board.legal_moves:
            board.push(move)
            _, curEval = base_minimax(board, depth - 1, True)
            board.pop()
            if curEval < minEval:
                minEval = curEval
                bestMove = move
        return bestMove, minEval


# Minimax with alpha-beta pruning
def minimax_alphabeta(
    board, depth=1, alpha=-float("inf"), beta=float("inf"), isWhite=True
):
    if depth == 0:
        eval_tensor = model(helper.fen2vec(board.fen(), isWhite).unsqueeze(0))
        return None, eval_tensor.item()

    if isWhite:
        bestMove = None
        maxEval = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            _, curEval = minimax_alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            if curEval > maxEval:
                maxEval = curEval
                bestMove = move
            alpha = max(alpha, curEval)
            if beta <= alpha:
                break
        return bestMove, maxEval
    else:
        bestMove = None
        minEval = float("inf")
        for move in board.legal_moves:
            board.push(move)
            _, curEval = minimax_alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()
            if curEval < minEval:
                minEval = curEval
                bestMove = move
            beta = min(beta, curEval)
            if beta <= alpha:
                break
        return bestMove, minEval


# Directly calling minimax with a board string (FEN)
def minimax_from_fen(fen_str, depth=1, isWhite=True, alphabeta=False):
    board = chess.Board(fen_str)
    if alphabeta:
        return minimax_alphabeta(board, depth, isWhite=isWhite)
    else:
        return base_minimax(board, depth, isWhite=isWhite)


# Example usage:
fen = "rnbqkbnr/pppp1ppp/8/8/8/4Pp2/PPPP1KPP/RNBQ1BNR w KQkq - 0 1"
move, evaluation = minimax_from_fen(fen, depth=4, isWhite=True, alphabeta=True)

print("Best Move:", move)
print("Evaluation:", evaluation)
