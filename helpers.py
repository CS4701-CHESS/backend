import torch

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


# function takes in an input fen and outputs an 8x8x6 tensor, and whose move it is.
#
# Each matrix of the tensor will represent the positions of the pieces,
# for 6 unique chess pieces. 1 represents a white piece, and -1 for black.
# The function will also output whose turn it is, true for white and false for black.
def fen2vec(fen):
    tens = torch.zeros((8, 8, 6), dtype=torch.int32)
    strs = fen.split(" ")
    rows = strs[0].split("/")
    side = True if strs[1] == "w" else False

    for row in range(8):
        currInd = 0

        while currInd < 8:
            currChar = rows[row][currInd]
            currAsc = ord(currChar)

            if currAsc > 48 & currAsc < 57:  # from 1-8
                currInd += currAsc - 48
            else:
                currPiece = fen_map[currChar]
                setVal = 1 if currPiece < 6 else -1
                tens[row][currInd][currPiece % 6] = setVal
                currInd += 1

    return tens, side


# function takes in input centipawn string and converts it into a usable integer.
#
# #'s are converted into +9999 or -9999
def cp2val(cp):
    op = cp[0]
    if op == "+":
        return int(cp[1, :])

    if op == "-":
        return int(cp[1, :])

    else:
        if cp[1] == "+":
            return 9999
        else:
            return -9999
