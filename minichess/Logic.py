from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

import chess


# Interface to the chess package for Python
#
# WHITE is Player 0
# BLACK is Player 1

class ID(Enum):
    """ Player ID """


@dataclass
class Player:
    name: str  # Player name
    id_: ID  # Player id


@dataclass
class Action:
    chessmove: chess.Move  # Move as in chess/Python

    def __init__(self, move):
        self.chessmove = move

    def __str__(self):
        return self.chessmove.uci()

    def __eq__(self, other):
        return self.chessmove == other.chessmove


FIRSTROW = [chess.Piece(chess.KNIGHT, chess.WHITE),
            chess.Piece(chess.QUEEN, chess.WHITE),
            chess.Piece(chess.KING, chess.WHITE),
            chess.Piece(chess.BISHOP, chess.WHITE),
            chess.Piece(chess.KNIGHT, chess.WHITE)]
LASTROW = [chess.Piece(chess.KNIGHT, chess.BLACK),
           chess.Piece(chess.BISHOP, chess.BLACK),
           chess.Piece(chess.KING, chess.BLACK),
           chess.Piece(chess.QUEEN, chess.BLACK),
           chess.Piece(chess.KNIGHT, chess.BLACK)]


class State:

    # Initialize the Chess board
    #
    # maxX = index of last file (column), when first column is 0
    # maxY = index of last rank (row), when first row is 0

    def __init__(self, players, maxX=4, maxY=4):
        self.__players = players
        self.board = chess.Board()
        self.board.clear_board()

        for i, p in list(enumerate(FIRSTROW)):
            self.board.set_piece_at(chess.parse_square(chess.FILE_NAMES[i] + "1"), p)

        for i, p in list(enumerate(LASTROW)):
            self.board.set_piece_at(chess.parse_square(chess.FILE_NAMES[i] + str(maxY + 1)), p)

        self.maxX = maxX
        self.maxY = maxY

    @classmethod
    def from_board(cls, board: chess.Board, maxX=4, maxY=4, players=2):
        state = cls(players, maxX, maxY)
        state.board = board.copy()
        return state

    # Current player

    def current_player(self) -> int:
        if self.board.turn == chess.WHITE:
            return 0
        else:
            return 1

    def current_player_id(self) -> ID:
        return self.__players[self.current_player()].id_

    def __str__(self) -> str:
        # Get whole board in Unicode
        s = self.board.unicode()
        # Split it to rows/ranks
        ranks = s.split("\n")
        visibleRanks = list(enumerate(ranks[7 - self.maxY:]))
        # Return maxX prefixes of maxY rows/ranks
        return '\n'.join(
            ["  abcdefgh"[:(self.maxX) * 2]] + [str(self.maxX + 1 - i) + " " + s[:(self.maxX + 1) * 2] for i, s in
                                                visibleRanks])

    def __str2__(self) -> str:
        # Empty board
        board = [["." for x in range(0, self.maxX + 1)] for y in range(0, self.maxY + 1)]
        # Map Square -> Piece
        pieces = self.board.piece_map()
        # Set all pieces on board
        for s in pieces:
            x = chess.square_file(s)
            y = chess.square_rank(s)
            board[y][x] = pieces.get(s).symbol()
        visibleRanks = list(enumerate(["".join(l) for l in board]))
        return '\n'.join([" abcdefgh"[:(self.maxX) + 2]] + [str(self.maxX + 1 - i) + s[:(self.maxX + 1) * 2] for i, s in
                                                            visibleRanks])

    # Is somebody the winner? 1 for current player, -1 for opponent, 0 for draw,
    # None for game is still on.

    def is_winner(self) -> Optional[int]:
        # Try python-chess winner determination:
        # (Winning on 8 X 8 board implies win also on smaller boards!)
        outc = self.board.outcome()
        if outc != None and (outc.winner == chess.WHITE or outc.winner == chess.BLACK):
            if outc.winner == self.board.turn:
                return 1
            else:
                return -1
        # There may still be Stalemate or Checkmate on the smaller board. Check!
        # First check for Checkmate:
        if self.checkmate(): return -1
        # Then check for Stalemate:
        if self.stalemate(): return 0
        return None

    # Checkmate? King threatened, and no legal move out of the situation?

    def checkmate(self) -> bool:
        return self.board.is_check() and all(not self.withinBounds(m) for m in self.board.legal_moves)

    # Stalemate? All legal moves lead to being in check.

    def stalemate(self) -> bool:
        return (not self.board.is_check()) and all(not self.withinBounds(m) for m in self.board.legal_moves)

    # Test if move is within bounds 0..maxX, 0..maxY

    def withinBounds(self, m):
        src = m.from_square
        dest = m.to_square
        #        print("Move: " +str(self.board.piece_at(m.from_square)) + " " + chess.square_name(src) + " to " +chess.square_name(dest))
        return chess.square_file(dest) <= self.maxX and chess.square_rank(dest) <= self.maxY

    # Applicable moves in the current state

    def applicable_moves(self) -> List[Action]:
        return [Action(m) for m in list(self.board.legal_moves) if self.withinBounds(m)]

    # Perform action in the current state

    def execute_move(self, m):
        self.board.push(m.chessmove)

    # Undo changes caused by last action, to enable backtracking

    def undo_last_move(self):
        self.board.pop()

    def clone(self):
        newstate = State(self.__players)
        newstate.board = self.board.copy()
        newstate.maxX = self.maxX
        newstate.maxY = self.maxY
        return newstate
