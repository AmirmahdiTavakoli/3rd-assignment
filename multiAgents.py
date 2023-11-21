# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import queue
from collections import deque

from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = better
        self.depth = int(depth)
        self.BEST_ACTION = None


class MinimaxAgent(MultiAgentSearchAgent):
    def minimax(self, gameState, agent_num, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent_num)

        if agent_num == 0:
            best_move = Directions.STOP
            max_value = float('-inf')

            for action in actions:
                next_state = gameState.generateSuccessor(0, action)
                value = self.minimax(next_state, agent_num + 1, depth)
                if value > max_value:
                    max_value = value
                    best_move = action
            if depth == self.depth:
                return best_move
            else:
                return max_value

        else:
            min_value = float('inf')
            for action in actions:
                next_state = gameState.generateSuccessor(agent_num, action)
                if agent_num == next_state.getNumAgents() - 1:
                    value = self.minimax(next_state, 0, depth - 1)
                else:
                    value = self.minimax(next_state, agent_num + 1, depth)
                min_value = min(min_value, value)
            return min_value

    def getAction(self, gameState):
        return self.minimax(gameState, self.index, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pac_pos = currentGameState.getPacmanPosition()

    def find_nearest_food(gameState):
        start_point = gameState.getPacmanState().getPosition()
        Map = gameState.__str__().splitlines()
        num_rows = len(Map) - 1
        num_cols = len(Map[0]) if num_rows > 0 else 0
        Map = Map[:num_rows][::-1]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        visited = set()
        queue = deque([((start_point[1], start_point[0]), 0)])  # (point, distance)

        while queue:
            current_point, dis = queue.popleft()
            row, col = current_point

            if row >= num_rows or col >= num_cols:
                continue
            if Map[row][col] == '.':
                return dis

            # Explore the neighbors in all four directions
            for direction in directions:
                new_row = row + direction[0]
                new_col = col + direction[1]

                if 0 <= new_row < num_rows and 0 <= new_col < num_cols and Map[new_row][new_col] != '%' and \
                        Map[new_row][new_col] != 'G':
                    new_point = (new_row, new_col)
                    if new_point not in visited:
                        visited.add(new_point)
                        queue.append((new_point, dis + 1))
        return 50

    ghost_pos = currentGameState.getGhostPositions()
    nearest_ghost = 1000
    for pos in ghost_pos:
        nearest_ghost = min(nearest_ghost, util.manhattanDistance(pos, pac_pos))
    if nearest_ghost < 4:
        nearest_ghost = -20
    return currentGameState.getScore() + (1 / (10000 * find_nearest_food(currentGameState))) + nearest_ghost


# Abbreviation
better = betterEvaluationFunction
