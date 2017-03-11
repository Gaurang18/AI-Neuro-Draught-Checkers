"""
Boilerplate code adapted from Everest Witman. Implemented TD-Learning functionality for intelligent agent.

*checkers.py
*A simple checkers engine written in Python with the pygame 1.9.1 libraries.
*Here are the rules I am using: http://boardgames.about.com/cs/checkersdraughts/ht/play_checkers.htm
*I adapted some code from checkers.py found at 
*http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py starting on line 159 of my program.
*This is the final version of my checkers project for Programming Workshop at Marlboro College. The entire thing has been rafactored and made almost completely object oriented.
*Funcitonalities include:
*- Having the pieces and board drawn to the screen
*- The ability to move pieces by clicking on the piece you want to move, then clicking on the square you would
  like to move to. You can change you mind about the piece you would like to move, just click on a new piece of yours.
*- Knowledge of what moves are legal. When moving pieces, you'll be limited to legal moves.
*- Capturing
*- DOUBLE capturing etc.
*- Legal move and captive piece highlighting
*- Turn changes
*- Automatic kinging and the ability for them to move backwords
*- Automatic check for and end game. 
*- A silky smoooth 60 FPS!
*Everest Witman - May 2014 - Marlboro College - Programming Workshop 

Shivin Srivastava, Gaurang Bansal - November 2016 - BITS Pilani - AI Course Project
"""

import pygame
import copy
import pprint
import sys
import time
import numpy as np
from pygame.locals import *
from predictor import *

pygame.font.init()
pp = pprint.PrettyPrinter()
# COLORS #
#             R    G    B
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
HIGH = (160, 190, 255)

##DIRECTIONS##
NORTHWEST = "northwest"
NORTHEAST = "northeast"
SOUTHWEST = "southwest"
SOUTHEAST = "southeast"

params = {'input_dimension': 24,
          'h1_dimension': 64,
          'alpha': 0.1,
          'lmbda': 0.8}

##SUPPORT
class Game:
    """
    The main game control.
    """

    def __init__(self):
        self.graphics = Graphics()
        self.board = Board()

        self.turn = BLUE
        self.hoppos = None
        self.selected_piece = None  # a board location.
        self.hop = False
        self.selected_legal_moves = []

    def setup(self):
        """Draws the window and board at the beginning of the game"""
        self.graphics.setup_window()

    def event_loop(self, network, pclr=RED, AI=False):
        """
        The event loop. This is where events are triggered
        (like a mouse click) and then effect the game state.
        """
        self.mouse_pos = self.graphics.board_coords(
            pygame.mouse.get_pos())  # what square is the mouse in?
        if self.selected_piece != None:
            self.selected_legal_moves = self.board.legal_moves(
                self.selected_piece, self.hop)

        for event in pygame.event.get():
            #OWN
            if AI and self.turn == pclr:
                boards = self.board.getAllBoards(self.board, self.hop, self.hoppos, color=pclr)

                if len(boards) > 0:
                    _, nextBoard, self.hop, self.hoppos = self.board.getBest(boards, network, 0.1, pclr)
                else:
                    # no change
                    nextBoard = self.board
                    self.hop = False

                self.board = nextBoard
                if not self.hop:
                    self.hoppos = None
                    self.end_turn()
                time.sleep(0.3)

            else:
                if event.type == QUIT:
                    self.terminate_game()

                if event.type == MOUSEBUTTONDOWN:
                    if self.hop == False:
                        if self.board.location(self.mouse_pos).occupant != None and self.board.location(self.mouse_pos).occupant.color == self.turn:
                            self.selected_piece = self.mouse_pos

                        elif self.selected_piece != None and self.mouse_pos in self.board.legal_moves(self.selected_piece):

                            self.board.move_piece(
                                self.selected_piece, self.mouse_pos)

                            if self.mouse_pos not in self.board.adjacent(self.selected_piece):
                                self.board.remove_piece((self.selected_piece[0] + (self.mouse_pos[0] - self.selected_piece[0]) / 2, self.selected_piece[1] + (self.mouse_pos[1] - self.selected_piece[1]) / 2))

                                self.hop = True
                                self.selected_piece = self.mouse_pos

                            else:
                                self.end_turn()

                    if self.hop == True:
                        if self.selected_piece != None and self.mouse_pos in self.board.legal_moves(self.selected_piece, self.hop):
                            self.board.move_piece(
                                self.selected_piece, self.mouse_pos)
                            self.board.remove_piece((self.selected_piece[0] + (self.mouse_pos[0] - self.selected_piece[0]) / 2, self.selected_piece[1] + (self.mouse_pos[1] - self.selected_piece[1]) / 2))

                        if self.board.legal_moves(self.mouse_pos, self.hop) == []:
                            self.end_turn()

                        else:
                            self.selected_piece = self.mouse_pos

    def update(self):
        """Calls on the graphics class to update the game display."""
        self.graphics.update_display(
            self.board, self.selected_legal_moves, self.selected_piece)

    def terminate_game(self):
        """Quits the program and ends the game."""
        pygame.quit()
        sys.exit

    def main(self, network, color):
        """"This executes the game and controls its flow."""
        self.setup()

        while True:  # main game loop
            self.event_loop(network, pclr=color, AI=True)
            self.update()

    def end_turn(self):
        """
        End the turn. Switches the current player.
        end_turn() also checks for and game and resets a lot of class attributes.
        """
        if self.turn == BLUE:
            self.turn = RED
        else:
            self.turn = BLUE

        self.selected_piece = None
        self.selected_legal_moves = []
        self.hop = False

        if self.check_for_endgame():
            if self.turn == BLUE:
                self.graphics.draw_message("RED WINS!")
            else:
                self.graphics.draw_message("BLUE WINS!")

    def check_for_endgame(self):
        """
        Checks to see if a player has run out of moves or pieces. If so, then return True. Else return False.
        """
        for x in xrange(8):
            for y in xrange(8):
                if self.board.location((x, y)).color == BLACK and self.board.location((x, y)).occupant != None and self.board.location((x, y)).occupant.color == self.turn:
                    if self.board.legal_moves((x, y)) != []:
                        return False

        return True

##SUPPORT
class Graphics:
    def __init__(self):
        self.caption = "Checkers"

        self.fps = 60
        self.clock = pygame.time.Clock()

        self.window_size = 600
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size))
        self.background = pygame.image.load('board.gif')

        self.square_size = self.window_size / 8
        self.piece_size = self.square_size / 2

        self.message = False

    def setup_window(self):
        """
        This initializes the window and sets the caption at the top.
        """
        pygame.init()
        pygame.display.set_caption(self.caption)

    def update_display(self, board, legal_moves, selected_piece):
        """
        This updates the current display.
        """
        self.screen.blit(self.background, (0, 0))

        self.highlight_squares(legal_moves, selected_piece)
        self.draw_board_pieces(board)

        if self.message:
            self.screen.blit(self.text_surface_obj, self.text_rect_obj)

        pygame.display.update()
        self.clock.tick(self.fps)

    def draw_board_squares(self, board):
        """
        Takes a board object and draws all of its squares to the display
        """
        for x in xrange(8):
            for y in xrange(8):
                pygame.draw.rect(self.screen, board[x][
                                 y].color, (x * self.square_size, y * self.square_size, self.square_size, self.square_size), )

    def draw_board_pieces(self, board):
        """
        Takes a board object and draws all of its pieces to the display
        """
        for x in xrange(8):
            for y in xrange(8):
                if board.matrix[x][y].occupant != None:
                    pygame.draw.circle(self.screen, board.matrix[x][
                                       y].occupant.color, self.pixel_coords((x, y)), self.piece_size)

                    if board.location((x, y)).occupant.king == True:
                        pygame.draw.circle(self.screen, GOLD, self.pixel_coords(
                            (x, y)), int(self.piece_size / 1.7), self.piece_size / 4)

    def pixel_coords(self, board_coords):
        """
        Takes in a tuple of board coordinates (x,y)
        and returns the pixel coordinates of the center of the square at that location.
        """
        return (board_coords[0] * self.square_size + self.piece_size, board_coords[1] * self.square_size + self.piece_size)

    def board_coords(self, (pixel_x, pixel_y)):
        """
        Does the reverse of pixel_coords(). Takes in a tuple of of pixel coordinates and returns what square they are in.
        """
        return (pixel_x / self.square_size, pixel_y / self.square_size)

    def highlight_squares(self, squares, origin):
        """
        Squares is a list of board coordinates.
        highlight_squares highlights them.
        """
        for square in squares:
            pygame.draw.rect(self.screen, HIGH, (square[
                             0] * self.square_size, square[1] * self.square_size, self.square_size, self.square_size))

        if origin != None:
            pygame.draw.rect(self.screen, HIGH, (origin[
                             0] * self.square_size, origin[1] * self.square_size, self.square_size, self.square_size))

    def draw_message(self, message):
        """
        Draws message to the screen.
        """
        self.message = True
        self.font_obj = pygame.font.Font('freesansbold.ttf', 44)
        self.text_surface_obj = self.font_obj.render(
            message, True, HIGH, BLACK)
        self.text_rect_obj = self.text_surface_obj.get_rect()
        self.text_rect_obj.center = (
            self.window_size / 2, self.window_size / 2)

#OWN
class Board:
    def __init__(self):
        self.matrix = self.new_board()

    def new_board(self):
        """
        Create a new board matrix.
        """

        # initialize squares and place them in matrix

        matrix = [[None for i in xrange(8)] for i in xrange(8)]

        # The following code block has been adapted from
        # http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py
        for x in xrange(8):
            for y in xrange(8):
                if (x % 2 != 0) and (y % 2 == 0):
                    matrix[y][x] = Square(WHITE)
                elif (x % 2 != 0) and (y % 2 != 0):
                    matrix[y][x] = Square(BLACK)
                elif (x % 2 == 0) and (y % 2 != 0):
                    matrix[y][x] = Square(WHITE)
                elif (x % 2 == 0) and (y % 2 == 0):
                    matrix[y][x] = Square(BLACK)

        # initialize the pieces and put them in the appropriate squares

        for x in xrange(8):
            for y in xrange(3):
                if matrix[x][y].color == BLACK:
                    matrix[x][y].occupant = Piece(RED)
            for y in xrange(5, 8):
                if matrix[x][y].color == BLACK:
                    matrix[x][y].occupant = Piece(BLUE)
        return matrix

    def board_string(self):
        """
        #OWN
        Takes a board and returns a matrix of the board space colors. Used for testing new_board()
        """
        board = self.matrix

        board_string = [[None for x in range(8)] for x in range(8)]

        for x in xrange(8):
            for y in xrange(8):
                if board[x][y].occupant == None:
                    board_string[x][y] = " "
                elif board[x][y].occupant.color == RED:
                    if board[x][y].occupant.king:
                        board_string[x][y] = "R*"
                    else:
                        board_string[x][y] = "R"
                else:
                    if board[x][y].occupant.king:
                        board_string[x][y] = "B*"
                    else:
                        board_string[x][y] = "B"
        # for row in board_string:
        #     print "".join(row)
        # print "\n"
        return board_string

    def train(self, episodes=100, eps=0.1):
        ##OWN
        '''
        This function is used for training the agent to play the game
        '''
        epno = 0
        # initializes a neural network
        network = initialize_network(params)
        turn = RED
        agent = 0
        player = 0
        while epno < episodes:
            hasGameEnded = False
            curr_board = Board()
            hop = False
            hoppos = None
            # randomly choose a starting player
            if np.random.random_integers(0, 1):
                turn = RED
            timetowin = []
            temp = 0
            while not hasGameEnded:
                evalPresent = evaluateNN(network, self.features(curr_board))
                # print evalPresent
                boards = self.getAllBoards(curr_board, hop, hoppos, turn)
                # epsilon is for eps-greedy approach
                if len(boards) > 0:
                    evalNext, nextBoard, hop, hoppos = self.getBest(boards, network, eps, turn)
                    # print pp.pprint(nextBoard.board_string())
                    # print "\n"
                else:
                    # no change
                    evalNext, nextBoard = evalPresent, curr_board
                    hop = False

                curr_board = nextBoard
                stat1 = curr_board.check_for_endgame(RED)
                stat2 = curr_board.check_for_endgame(BLUE)

                if stat1 or stat2:
                    evalNext = stat1  # assuming that stat1 is the status of player1
                    agent += stat1
                    player += stat2
                    # print "GAME OVER!!!"
                    hasGameEnded = True
                    epno += 1

                network = backpropagate(params, network, evalNext,
                                        evalPresent, self.features(nextBoard))

                if hop == False:
                    if turn == RED:
                        turn = BLUE
                    else:
                        turn = RED
                temp+=1
            print("{},{},{}").format(agent, player, temp)
            save_Network(network)
        fp = open("wins", "w")
        print timetowin
        print >> fp, timetowin
        return network

    def getAllBoards(self, curr_board, hop, hoppos, color=RED):
        """
        hop: tells whether we can hop again or not
        hoppos: gives the previous position where we were
        and from where we have to hop
        """
        newboards = []
        mtrx = curr_board.matrix
        if hop:
            hop = False
            tempBoard = copy.deepcopy(curr_board)
            tempMoves = tempBoard.legal_moves(hoppos, hop=True)
            for move in tempMoves:
                tempBoard = copy.deepcopy(curr_board)
                x = hoppos[0]
                y = hoppos[1]
                adjacent = tempBoard.adjacent((x, y))
                tempBoard.move_piece((x, y), move)
                hoppos = None
                if move not in adjacent:
                    tempBoard.remove_piece((x + (move[0] - x) / 2, y + (move[1] - y) / 2))
                    hop = True
                    hoppos = move
                newboards.append((tempBoard, hop, hoppos))

        else:
            for x in xrange(8):
                for y in xrange(8):
                    if mtrx[x][y].occupant is not None and mtrx[x][y].occupant.color == color:
                        tempMoves = []
                        tempBoard = copy.deepcopy(curr_board)
                        tempMoves = tempBoard.legal_moves((x, y))
                        for move in tempMoves:
                            tempBoard = copy.deepcopy(curr_board)
                            adjacent = tempBoard.adjacent((x, y))
                            tempBoard.move_piece((x, y), move)

                            if move not in adjacent:
                                tempBoard.remove_piece((x + (move[0] - x) / 2, y + (move[1] - y) / 2))
                                hop = True
                                hoppos = move

                            newboards.append((tempBoard, hop, hoppos))

        return newboards

    def rel(self, dir, (x, y)):
        """
        Returns the coordinates one square in a different direction to (x,y).
        """
        if dir == NORTHWEST:
            return (x - 1, y - 1)
        elif dir == NORTHEAST:
            return (x + 1, y - 1)
        elif dir == SOUTHWEST:
            return (x - 1, y + 1)
        elif dir == SOUTHEAST:
            return (x + 1, y + 1)
        else:
            return 0

    def getBest(self, boards, network, eps, color):
        """
        This function evaluates all the boards and evaluates them according to their
        favourability for winning as evaluated by the neural network.
        """
        if color == RED:
            best = -np.inf
        else:
            best = np.inf

        bestBoard = None
        list = []
        hop = False
        hoppos = (0, 0)
        for board in boards:
            brd = board[0]
            feature = brd.features(brd, color)
            evalval = evaluateNN(network, feature)
            list.append([evalval, brd, board[1], board[2]])

            if color == RED and evalval > best:
                best = evalval
                bestBoard = brd
                hop = board[1]
                hoppos = board[2]

            elif color == BLUE and evalval < best:
                best = evalval
                bestBoard = brd
                hop = board[1]
                hoppos = board[2]

        # implementing eps-greedy method
        if np.random.rand(1)[0] < eps:
            idx = np.random.random_integers(0, len(list) - 1)
            selected = list[idx]
            return selected[0], selected[1], selected[2], selected[3]

        return best, bestBoard, hop, hoppos

    def adjacent(self, (x, y)):
        """
        Returns a list of squares locations that are adjacent (on a diagonal) to (x,y).
        """

        return [self.rel(NORTHWEST, (x, y)), self.rel(NORTHEAST, (x, y)),
                self.rel(SOUTHWEST, (x, y)), self.rel(SOUTHEAST, (x, y))]

    def location(self, (x, y)):
        """
        Takes a set of coordinates as arguments and returns self.matrix[x][y]
        This can be faster than writing something like self.matrix[coords[0]][coords[1]]
        """

        return self.matrix[x][y]

    def blind_legal_moves(self, (x, y)):
        """
        Returns a list of blind legal move locations from a set of coordinates (x,y) on the board.
        If that location is empty, then blind_legal_moves() return an empty list.
        Both color kings have same moves
        """

        if self.matrix[x][y].occupant != None:

            if self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == BLUE:
                blind_legal_moves = [
                    self.rel(NORTHWEST, (x, y)), self.rel(NORTHEAST, (x, y))]

            elif self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == RED:
                blind_legal_moves = [
                    self.rel(SOUTHWEST, (x, y)), self.rel(SOUTHEAST, (x, y))]

            else:
                blind_legal_moves = [self.rel(NORTHWEST, (x, y)), self.rel(
                    NORTHEAST, (x, y)), self.rel(SOUTHWEST, (x, y)), self.rel(SOUTHEAST, (x, y))]

        else:
            blind_legal_moves = []

        return blind_legal_moves

    def legal_moves(self, (x, y), hop=False):
        """
        Returns a list of legal move locations from a given set of coordinates (x,y) on the board.
        If that location is empty, then legal_moves() returns an empty list.
        """

        blind_legal_moves = self.blind_legal_moves((x, y))
        legal_moves = []

        if hop == False:
            for move in blind_legal_moves:
                if hop == False:
                    if self.on_board(move):
                        if self.location(move).occupant is None:
                            legal_moves.append(move)

                        # is this location filled by an enemy piece?
                        elif self.location(move).occupant.color != self.location((x, y)).occupant.color and self.on_board((move[0] + (move[0] - x), move[1] + (move[1] - y))) and self.location((move[0] + (move[0] - x), move[1] + (move[1] - y))).occupant is None:
                            legal_moves.append((move[0] + (move[0] - x), move[1] + (move[1] - y)))

        else:  # hop == True
            for move in blind_legal_moves:
                if self.on_board(move) and self.location(move).occupant is not None:
                    # is this location filled by an enemy piece?
                    if self.location(move).occupant.color != self.location((x, y)).occupant.color and self.on_board((move[0] + (move[0] - x), move[1] + (move[1] - y))) and self.location((move[0] + (move[0] - x), move[1] + (move[1] - y))).occupant is None:
                        legal_moves.append((move[0] + (move[0] - x), move[1] + (move[1] - y)))

        return legal_moves

    def remove_piece(self, (x, y)):
        """
        Removes a piece from the board at position (x,y).
        """
        self.matrix[x][y].occupant = None

    def move_piece(self, (start_x, start_y), (end_x, end_y)):
        """
        Move a piece from (start_x, start_y) to (end_x, end_y).
        """

        self.matrix[end_x][end_y].occupant = self.matrix[
            start_x][start_y].occupant
        self.remove_piece((start_x, start_y))

        self.king((end_x, end_y))

    def is_end_square(self, coords):
        """
        Is passed a coordinate tuple (x,y), and returns true or
        false depending on if that square on the board is an end square.
        """

        if coords[1] == 0 or coords[1] == 7:
            return True
        else:
            return False

    def on_board(self, (x, y)):
        """
        Checks to see if the given square (x,y) lies on the board.
        If it does, then on_board() return True. Otherwise it returns false.
        """

        if x < 0 or y < 0 or x > 7 or y > 7:
            return False
        else:
            return True

    def king(self, (x, y)):
        """
        Takes in (x,y), the coordinates of square to be considered for kinging.
        If it meets the criteria, then king() kings the piece in that square and kings it.
        """
        if self.location((x, y)).occupant != None:
            if (self.location((x, y)).occupant.color == BLUE and y == 0) or (self.location((x, y)).occupant.color == RED and y == 7):
                self.location((x, y)).occupant.king = True

    def features(self, board, color=RED):
        """
        This function converts the board to an 18 dimentional vector representation
        so that it can be fed into the neural network for training.
        """
        mtrx = board.matrix
        feats = np.zeros(params['input_dimension'])
        # number of keys
        for x in xrange(8):
            for y in xrange(8):
                key = mtrx[x][y].occupant
                if key is not None:
                    if key.color == RED:
                        fnum = 0
                    if key.color == BLUE:
                        fnum = 1

                    feats[fnum] += 2

                    # compare blind_legal_moves with legal_moves
                    # piece_take
                    blm = self.blind_legal_moves((x, y))
                    lm = self.legal_moves((x, y))
                    for move in lm:
                        if move not in blm:
                            feats[fnum + 2] += 1.5
                            if key.king is True:
                                feats[fnum + 2] += 1.6

                    # number of pieces under threat
                    feats[fnum + 4] = feats[fnum] / 2 - feats[fnum + 2]

                    # proximity of pieces to being knighted
                    if key.color == RED and key.king is False:
                        feats[fnum + 6] += x / 7

                    elif key.color == BLUE and key.king is False:
                        feats[fnum + 6] += (7 - x) / 7

                    # more features here
                    # Back Row Bridge
                    ad = self.adjacent((x, y))
                    bd = self.blind_legal_moves((x, y))
                    for p in ad:
                        if p not in bd:
                            if self.on_board(p) and mtrx[p[0]][p[1]].occupant is not None and mtrx[p[0]][p[1]].color == key.color:
                                feats[fnum + 8] += 2.5

                    # key is going to get cut
                    for p in ad:
                        if self.on_board(p) and mtrx[p[0]][p[1]].occupant is not None and mtrx[p[0]][p[1]].color != key.color:
                                lmo = self.legal_moves((p[0], p[1]))
                                for l in lmo:
                                    if l in ad:
                                        feats[fnum + 10] -= 3

                    if x - 3 >= 0 and x - 3 <= 3 and y - 3 >= 0 and y - 3 <= 3:
                        feats[fnum + 12] += 1
                        if key.king is True:
                            feats[fnum + 14] += 0.02

                    for move in lm:
                        x1 = move[0]
                        y1 = move[1]
                        if x1 - 3 >= 0 and x1 - 3 <= 3 and y1 - 3 >= 0 and y1 - 3 <= 3:
                            feats[fnum + 16] += 1
                            if key.king is True:
                                feats[fnum + 18] += 0.2

                    #Peice Taken Advantage
                    for move in lm:
                        if move not in ad:
                            feats[fnum + 20] += 5

                    #Check if peice can be cut
                    for dp in lm:
                        k = self.adjacent(dp)
                        for p in k: 
                            if self.on_board(p) and mtrx[p[0]][p[1]].occupant is not None and mtrx[p[0]][p[1]].color != key.color:
                                slm = self.legal_moves((p[0], p[1]))
                                for io in lm:
                                    if io in slm:
                                        feats[fnum + 22] -= 2.5


        feats = feats / np.sum(feats)
        return feats

    def check_for_endgame(self, color):
        """
        Checks to see if a player has run out of moves or pieces. If so, then return 1. Else return 0. These are the rewards at each step also.
        """
        cnt = 0
        flag = 0
        mtrx = self.matrix
        for x in xrange(8):
            for y in xrange(8):
                if mtrx[x][y].occupant is not None and mtrx[x][y].occupant.color == color:
                    cnt += 1
                    if self.legal_moves((x, y)) != []:
                        flag = 1
        if cnt <= 2:
            return 1
        if flag:
            return 0
        return 1

#SUPPORT
class Piece:
    def __init__(self, color, king=False):
        self.color = color
        self.king = king

#SUPPORT
class Square:
    def __init__(self, color, occupant=None):
        self.color = color  # color is either BLACK or WHITE
        self.occupant = occupant  # occupant is a Square object


def save_Network(network):
    """
    This function saves the network parameters
    """
    np.save('./parameters/VIH_nok5', network[0])
    np.save('./parameters/VHO_nok5', network[1])
    np.save('./parameters/EIH_nok5', network[2])
    np.save('./parameters/EHO_nok5', network[3])


def load_Network():
    """
    This function loades the previously saved network parameters
    """
    network = [0, 0, 0, 0]
    np.load('./parameters/VIH_nok.npy', network[0])
    np.load('./parameters/VHO_nok.npy', network[1])
    np.load('./parameters/EIH_nok.npy', network[2])
    np.load('./parameters/EHO_nok.npy', network[3])
    return network


def main():
    game = Game()
    board = Board()
    network = load_Network()
    #network = board.train(episodes=100)
    #save_Network(network)
    game.main(network, RED)

if __name__ == "__main__":
    main()
