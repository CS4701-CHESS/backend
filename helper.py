import torch
import chess
from stockfish import Stockfish

# load stockfish model
stockfish = Stockfish(
    path="stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
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
# The function will return None if the given side does not match the data.
def fen2vec(fen, isWhite):
    tens = torch.zeros((14, 8, 8), dtype=torch.float32)
    strs = fen.split(" ")
    rows = strs[0].split("/")
    side = True if strs[1] == "w" else False

    if side != isWhite:
        return None

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


# function takes in input centipawn string and converts it into a usable integer.
#
# #'s are converted into +9999 or -9999
def cp2val(cp):
    op = cp[0]
    if op == "+":
        return int(cp[1:])

    if op == "-":
        return -(int(cp[1:]))

    if op == "0":
        return 0

    else:
        if cp[1] == "+":
            return 9999
        else:
            return -9999


# create custom dataset from fen strings using stockfish at different depths
def fen2pair(fen, isWhite, depth):
    stockfish.set_depth(depth)
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    if eval["type"] == "cp":
        return fen2vec(fen, isWhite), eval["value"]
    else:
        return fen2vec(fen, isWhite), eval["value"] * 1000
