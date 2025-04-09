



# # Example usage:
# fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# move, evaluation = minimax_from_fen(fen, depth=2, isWhite=True, alphabeta=True)

# print("Best Move:", move)
# print("Evaluation:", evaluation)


import nn as defs
import torch
import helper
import chess

# Load in trained model
model = defs.Model(32, 4)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Transposition Table
transposition_table = {}

def evaluate_board(board, isWhite):
    fen = board.fen()
    if fen in transposition_table:
        return transposition_table[fen]
    eval_tensor = model(helper.fen2vec(fen, isWhite).unsqueeze(0))
    eval_score = eval_tensor.item()
    transposition_table[fen] = eval_score
    return eval_score

# Simple move ordering heuristic (captures first)
def order_moves(board):
    captures = []
    non_captures = []
    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            non_captures.append(move)
    return captures + non_captures

def minimax_alphabeta(board, depth=1, alpha=-float('inf'), beta=float('inf'), isWhite=True):
    if depth == 0 or board.is_game_over():
        return None, evaluate_board(board, isWhite)

    bestMove = None
    if isWhite:
        maxEval = -float('inf')
        for move in order_moves(board):
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
        minEval = float('inf')
        for move in order_moves(board):
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

# Example usage
fen = "r3b2k/p5p1/4pq2/1p1p4/2n2P2/P2B4/1P2Q2P/1K1R2R1 w - - 4 26"
board = chess.Board(fen)
move, evaluation = minimax_alphabeta(board, depth=3, isWhite=True)

# fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print("Best Move:", move)
print("Evaluation:", evaluation)
