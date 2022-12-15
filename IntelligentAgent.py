import random
from BaseAI import BaseAI
import numpy as np


class IntelligentAgent(BaseAI):
	log2_dict = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9,
				 1024: 10, 2048: 11, 4096: 12, 8192: 13, 16384: 14, 32768: 15}
	structure = np.array([1, 1, 1, -1, 1])

	def getMove(self, grid):
		return self.minimax(state=grid, depth=3, turn=True, alpha=float("-inf"), beta=float("inf"))[0][0]

	def minimax(self, state, depth, turn, alpha, beta, tile=2):

		if (depth == 0) or (not state.canMove()):
			return (None, self.evaluation(state))

		max_child = None
		min_child = None
		max_utility = float("-inf")
		min_utility = float("inf")

		if turn:  #max
			for child in state.getAvailableMoves():
				two_utility = self.minimax(state=child[1], depth=depth - 1, turn=False, alpha=alpha, beta=beta)[1]
				four_utility = self.minimax(state=child[1], depth=depth - 1, turn=False, alpha=alpha, beta=beta, tile=4)[1]


				total_utility = (two_utility*0.9) + (four_utility*0.1)

				if total_utility > max_utility:
					max_child = child
					max_utility = total_utility

				if max_utility >= beta:
					break

				if max_utility >= alpha:
					alpha = max_utility

			return max_child, max_utility

		else:  #min
			for move in state.getAvailableCells():

				copy = state.clone()
				copy.setCellValue(move, tile)
				utility = self.minimax(state=copy, depth=depth - 1, turn=True, alpha=alpha, beta=beta)[1]

				if utility < min_utility:
					min_child = copy
					min_utility = utility

				if min_utility <= alpha:
					break

				if min_utility < beta:
					beta = min_utility

			return min_child, min_utility

	def evaluation(self, grid):
		score_snake = [[4**15,4**14,4**13,4**12],
					   [4**8,4**9,4**10,4**11],
					   [4**7,4**6,4**5,4**4],
					   [4**0,4**1,4**2,4**3]]

		score_counter = 0

		for i in range(0,4):
			for j in range(0,4):
				score_counter += (grid.map[i][j])*score_snake[i][j]

		matrix = np.array(grid.map)
		max_tile = grid.getMaxTile()
		reward = -100
		available_cells = grid.getAvailableCells()

		if matrix[0][0] == max_tile:
			reward = 0
		else:
			score_counter = -100

		if not (len(available_cells) == 0):
			max_tile = len(available_cells)
		else:
			return -10

		change = self.modifier(matrix)
		heuristic = [score_counter, 5*max_tile, self.monotonic_counter(matrix), 0.3*change[0], reward]
		heuristic_as_array = np.array(heuristic)

		reward_structure = heuristic_as_array.dot(IntelligentAgent.structure)

		return reward_structure

	def absolute_value(self, first_tile, second_tile):
		return [abs(first_tile - second_tile), 0]

	def modifier(self, modify):
		pair_count = 0
		total_count = 0

		for i in range(4):
			for j in range(4):
				filler = IntelligentAgent.log2_dict[modify[i,j]]

				if i > 0:
					check = self.absolute_value(filler,IntelligentAgent.log2_dict[modify[i-1,j]])
					total_count = total_count + check[0]
					pair_count = pair_count + check[1]
				if i < 3:
					check = self.absolute_value(filler, IntelligentAgent.log2_dict[modify[i+1, j]])
					total_count = total_count + check[0]
					pair_count = pair_count + check[1]
				if j > 0:
					check = self.absolute_value(filler,IntelligentAgent.log2_dict[modify[i,j-1]])
					total_count = total_count + check[0]
					pair_count = pair_count + check[1]
				if j < 3:
					check = self.absolute_value(filler, IntelligentAgent.log2_dict[modify[i, j+1]])
					total_count = total_count + check[0]
					pair_count = pair_count + check[1]

		return[total_count, pair_count]

	def is_monotonic(self, v):
		if all(x >= y for x,y in zip(v,v[1:])):
			return True
		elif all(x <= y for x,y in zip(v,v[1:])):
			return True
		else:
			return False

	def monotonic_counter(self, a):
		monotonic_count = 0
		flip = np.fliplr(a)

		for i in range(4):
			if self.is_monotonic(tuple(a[i])):
				monotonic_count += 1
			if self.is_monotonic(tuple(a[:, i])):
				monotonic_count += 1
		for i in range(-1,2):
			if self.is_monotonic(tuple(a.diagonal(i))):
				monotonic_count += 1
			if self.is_monotonic(tuple(flip.diagonal(i))):
				monotonic_count += 1

		return monotonic_count












