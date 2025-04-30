import torch
import chess
from stockfish import Stockfish
import os
import numpy as np

# load stockfish model
stockfish = Stockfish(
    # path=os.path.abspath("stockfish/stockfish-macos-x86-64-bmi2")
    path=os.path.abspath(
        "stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
    )
)

# uppercase is white, lowercase is black
fen_map = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


# function takes in an input fen and outputs an 14x8x8 tensor
#
# The first 12 matricies of the tensor will represent the positions of the pieces,
# converted to an index by the dictionary (white, then black). The final two matricies
# represent the attacked squares from both sides.
# The function will now work for both white and black sides.
def fen2vec(fen, isWhite=None):  # isWhite parameter is now optional
    tens = torch.zeros((14, 8, 8), dtype=torch.float32)
    strs = fen.split(" ")
    rows = strs[0].split("/")
    side = True if strs[1] == "w" else False

    # Removed the condition that returned None when side != isWhite
    # This allows the function to work for both sides

    for row in range(8):
        currInd = 0
        stringInd = 0

        while currInd < 8:
            currChar = rows[row][stringInd]
            currAsc = ord(currChar)

            if currAsc > 48 and currAsc < 57:  # from 1-8
                currInd += currAsc - 48
            else:
                currPiece = fen_map[currChar]
                tens[currPiece][row][currInd] = 1
                currInd += 1

            stringInd += 1

    # computing attacked squares
    board = chess.Board(fen)
    whiteAttacked = chess.SquareSet()
    blackAttacked = chess.SquareSet()

    for attacker in chess.SquareSet(board.occupied_co[chess.WHITE]):
        whiteAttacked |= board.attacks(attacker)
    for attacker in chess.SquareSet(board.occupied_co[chess.BLACK]):
        blackAttacked |= board.attacks(attacker)

    for square in whiteAttacked:
        tens[12][(63 - square) // 8][7 - ((63 - square) % 8)] = 1
    for square in blackAttacked:
        tens[13][(63 - square) // 8][7 - ((63 - square) % 8)] = 1

    return tens


# create custom dataset from fen strings using stockfish at different depths
def fen2pair(fen, isWhite, depth):
    stockfish.set_depth(depth)
    stockfish.set_fen_position(fen)
    move = stockfish.get_best_move()
    return (fen2vec(fen, isWhite), move)


# encode uci moves from string into int
def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return (
        np.array([move_to_int[move] for move in moves], dtype=np.float32),
        move_to_int,
    )


# early stopping to training to prevent overfitting
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
