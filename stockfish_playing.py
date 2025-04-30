from stockfish import Stockfish
import os

# load stockfish model
stockfish = Stockfish(
    # path=os.path.abspath("stockfish/stockfish-macos-x86-64-bmi2")
    path=os.path.abspath(
        "stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
    )
)


# Play one game against stockfish
def play_stockfish(side):
    # set stockfish to shitty
    # make new board
    # pass new board to one model depending on side
    # get move from first model
    # update board
    # pass to other model
    # add stockfish eval to list for graphing
    # while loop until someone loses
    pass


# play n games against stockfish
def play_multiple(n):
    # set accumulator
    # call above func n times
    pass
