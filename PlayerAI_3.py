import random
import math
import numpy as np
from BaseAI import BaseAI
import time
#em3373

class PlayerAI(BaseAI):
    log2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11, 4096: 12, 8192: 13,
            16384: 14}
    weights = np.array([1,1, 1, -1,1])
    def getMove(self, grid):
        return self.Expectminimax(state= grid, depth= 3, ourTurn=True, alpha=float("-inf"), beta=float("inf"))[0][0]

    def Expectminimax(self, state, depth, ourTurn, alpha, beta, tile = 2):

        if (depth == 0) or (not state.canMove()):
            return (None, self.eval(state))

        maxChild = None
        minChild = None
        maxUtility = float("-inf")
        minUtility = float("inf")

        if ourTurn: # MAXIMIZE
            for child in state.getAvailableMoves():
                utility = self.Expectminimax(state= child[1], depth=depth-1, ourTurn=False, alpha=alpha, beta=beta)[1]
                utility2 = self.Expectminimax(state=child[1], depth=depth - 1, ourTurn=False, alpha=alpha, beta=beta, tile=4)[1]

                utility = utility*0.9 + utility2*0.1

                if utility > maxUtility:
                    maxChild = child
                    maxUtility = utility

                if maxUtility >= beta:
                    break

                if maxUtility >= alpha:
                    alpha = maxUtility

            return (maxChild, maxUtility)
        else: #MINIMIZE
            for move in state.getAvailableCells():

                gridcopy = state.clone()
                gridcopy.setCellValue(move, tile)
                utility = self.Expectminimax(state= gridcopy, depth=depth-1, ourTurn=True, alpha=alpha, beta=beta)[1]

                if utility < minUtility:
                    minChild = gridcopy
                    minUtility = utility

                if minUtility <= alpha:
                    break

                if minUtility < beta:
                    beta = minUtility

            return (minChild, minUtility)

    def eval(self, grid):
        weight = [[4**15, 4**14, 4**13, 4**12], [4**8, 4**9, 4**10, 4**11], [4**7, 4**6, 4**5, 4**4], [1, 4, 4**2, 4**3]]
        score = 0
        for i in range(0,4):
            for j in range(0,4):
                score = score + (grid.map[i][j])*weight[i][j]

        grid_as_matrix = np.array(grid.map)
        avc = grid.getMaxTile()
        bonus = -100
        cells = grid.getAvailableCells()
        if (grid_as_matrix[0][0] == avc) or (grid_as_matrix[0][3] == avc) or (grid_as_matrix[3][0] == avc) or (grid_as_matrix[3][3] == avc):
            bonus = 0
        else:
            score = -100
        if not (len(cells) == 0):
            avc = len(cells)
        else:
           return -10
        smooth = self.smoother(grid_as_matrix)
        linea = [score,5*avc,self.monotonic_checker(grid_as_matrix),.3*smooth[0], bonus]
        heuristic = np.array(linea)

        utility = heuristic.dot(PlayerAI.weights)

        return  utility

    def abs_check(self, tile1, tile2):
        return [abs(tile1 - tile2), 0]

    def smoother(self, a):
        total = 0
        pairs = 0
        for i in range(4):
            for j in range(4):
                fijo = PlayerAI.log2[a[i, j]]
                if i > 0:
                    checks = self.abs_check(fijo, PlayerAI.log2[a[i - 1, j]])
                    total += checks[0]
                    pairs += checks[1]
                if i < 3:
                    checks = self.abs_check(fijo, PlayerAI.log2[a[i + 1, j]])
                    total += checks[0]
                    pairs += checks[1]
                if j > 0:
                    checks = self.abs_check(fijo, PlayerAI.log2[a[i, j - 1]])
                    total += checks[0]
                    pairs += checks[1]
                if j < 3:
                    checks = self.abs_check(fijo, PlayerAI.log2[a[i, j + 1]])
                    total += checks[0]
                    pairs += checks[1]
        return [total, pairs]

    def mono(self, vector):
        if all(x >= y for x, y in zip(vector, vector[1:])):
            return True
        else:

            if all(x <= y for x, y in zip(vector, vector[1:])):
                return True
        return False

    def monotonic_checker(self, array):

        total = 0

        inverse = np.fliplr(array)

        for i in range(4):
            if self.mono(tuple(array[i])): total += 1
            if self.mono(tuple(array[:, i])): total += 1


        for i in range(-1, 2):
            if self.mono(tuple(array.diagonal(i))): total += 1
            if self.mono(tuple(inverse.diagonal(i))): total += 1

        return total







