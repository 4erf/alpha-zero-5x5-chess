from typing import Tuple, List
import chess
from .Logic import State, Action

class MinichessGame:
    actions: List[Tuple[str, int, int]] = [
        # Knight 1
        ('N1', 1, 2), ('N1', 2, 1), ('N1', 2, -1), ('N1', 1, -2),
        ('N1', -1, -2), ('N1', -2, -1), ('N1', -2, 1), ('N1', -1, 2),
        # Knight 2
        ('N2', 1, 2), ('N2', 2, 1), ('N2', 2, -1), ('N2', 1, -2),
        ('N2', -1, -2), ('N2', -2, -1), ('N2', -2, 1), ('N2', -1, 2),
        # Bishop
        ('B', 1, 1), ('B', 2, 2), ('B', 3, 3), ('B', 4, 4),
        ('B', 1, -1), ('B', 2, -2), ('B', 3, -3), ('B', 4, -4),
        ('B', -1, -1), ('B', -2, -2), ('B', -3, -3), ('B', -4, -4),
        ('B', -1, 1), ('B', -2, 2), ('B', -3, 3), ('B', -4, 4),
        # Queen
        ('Q', 0, 1), ('Q', 0, 2), ('Q', 0, 3), ('Q', 0, 4),
        ('Q', 1, 1), ('Q', 2, 2), ('Q', 3, 3), ('Q', 4, 4),
        ('Q', 1, 0), ('Q', 2, 0), ('Q', 3, 0), ('Q', 4, 0),
        ('Q', 1, -1), ('Q', 2, -2), ('Q', 3, -3), ('Q', 4, -4),
        ('Q', 0, -1), ('Q', 0, -2), ('Q', 0, -3), ('Q', 0, -4),
        ('Q', -1, -1), ('Q', -2, -2), ('Q', -3, -3), ('Q', -4, -4),
        ('Q', -1, 0), ('Q', -2, 0), ('Q', -3, 0), ('Q', -4, 0),
        ('Q', -1, 1), ('Q', -2, 2), ('Q', -3, 3), ('Q', -4, 4),
        # King
        ('K', 0, 1), ('K', 1, 1), ('K', 1, 0), ('K', 1, -1),
        ('K', 0, -1), ('K', -1, -1), ('K', -1, 0), ('K', -1, 1),
    ]

    def __init__(self, maxX=4, maxY=4):
        self.maxX = maxX
        self.maxY = maxY

    def getInitBoard(self):
        state = State(players=2, maxX=self.maxX, maxY=self.maxY)
        return state.board

    def getBoardSize(self):
        return self.maxX + 1, self.maxY + 1

    def getActionSize(self):
        return len(self.actions)

    def getKnightSquares(self, board: chess.Board, color=chess.WHITE):
        squares = board.pieces(chess.KNIGHT, color)
        coords = sorted(
            (chess.square_file(square), chess.square_rank(square), square)
            for square in squares
        )
        return [square for _, _, square in coords]

    def getNextState(self, board: chess.Board, player, action):
        assert board.turn == (player == 1)
        color = chess.WHITE if player == 1 else chess.BLACK

        state = State.from_board(board, self.maxX, self.maxY)
        (piece, dx, dy) = self.actions[action]

        knight_squares = self.getKnightSquares(board, color)
        if piece == 'N1' or piece == 'N2':
            square = knight_squares[int(piece[1]) - 1]
        else:
            piece_type = chess.Piece.from_symbol(piece).piece_type
            square = board.pieces(piece_type, color).pop()

        to_x, to_y = chess.square_file(square) + dx, chess.square_rank(square) + dy

        move = chess.Move.from_uci("0000")
        move.from_square = square
        move.to_square = chess.parse_square(chess.FILE_NAMES[to_x] + str(to_y + 1))

        action = Action(move)
        state.execute_move(action)

        return state.board, -player


    def getActionTuple(self, board: chess.Board, action: Action):
        move = action.chessmove
        piece = board.piece_at(move.from_square).symbol()

        knight_squares = self.getKnightSquares(board)
        if piece == 'N':
            piece = 'N1' if move.from_square == knight_squares[0] else 'N2'

        from_x, from_y = chess.square_file(move.from_square), chess.square_rank(move.from_square)
        to_x, to_y = chess.square_file(move.to_square), chess.square_rank(move.to_square)

        return piece, to_x - from_x, to_y - from_y


    def getValidMoves(self, board: chess.Board, player):
        assert player == 1
        state = State.from_board(board, self.maxX, self.maxY)
        moves = set(
            self.getActionTuple(board, action)
            for action in state.applicable_moves()
        )
        return [1 if action in moves else 0 for action in self.actions]

    def getGameEnded(self, board: chess.Board, player):
        # assert board.turn == (player == 1)
        draw = 1e-5

        state = State.from_board(board, self.maxX, self.maxY)
        if state.is_winner() is not None:
            return state.is_winner() if state.is_winner() != 0 else draw
        if len(board.move_stack) > 100:
            return draw

        return 0


    def getCanonicalForm(self, board: chess.Board, player):
        if player == 1:
            return board
        else:
            fen = board.fen()
            position, *rest = fen.split(" ")
            inverted_pos = position.swapcase()
            inverted_fen = ' '.join([inverted_pos, *rest])
            inverted_board = chess.Board(inverted_fen)
            inverted_board.turn = not inverted_board.turn
            inverted_board.move_stack = board.move_stack
            return inverted_board

    def getSymmetries(self, board: chess.Board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board: chess.Board):
        return str(board)
