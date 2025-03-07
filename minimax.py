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

def base_minimax(board, depth, isWhite):
    if(depth == 0):
        return None, model(helper.fen2vec(board.fen(), isWhite).unsqueeze(0))
    if isWhite:
        bestMove = None
        maxEval = -9999
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
        maxEval = 9999
        for move in board.legal_moves:
            board.push(move)
            _, curEval = base_minimax(board, depth - 1, True)
            board.pop()
            maxEval = min(maxEval, eval)
            if curEval > maxEval:
                maxEval = curEval
                bestMove = move
        return bestMove, maxEval

def minimax_alphabeta(board, depth, alpha, beta, isWhite):
    if(depth == 0):
        return None, model(helper.fen2vec(board.fen(), isWhite).unsqueeze(0))
    if isWhite:
        bestMove = None
        maxEval = -9999
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
        maxEval = 9999
        for move in board.legal_moves:
            board.push(move)
            _, curEval = minimax_alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()
            maxEval = min(maxEval, eval)
            if curEval > maxEval:
                maxEval = curEval
                bestMove = move
            beta = min(beta, curEval)
            if beta <= alpha:
                break
        return bestMove, maxEval