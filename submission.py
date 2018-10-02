from util import *
from game import Directions
import random, util, math

from game import Agent

"""
General information
			Pac-Man is always agent 0, and the agents move in order of increasing agent index. Use self.index in your minimax 
			implementation, but only Pac-Man will actually be running your MinimaxAgent.
			
			Functions are provided to get legal moves for Pac-Man or the ghosts and to execute a move by any agent. 
			See GameState in pacman.py for details.

			All states in minimax should be GameStates, either passed in to getAction or generated via GameState.generateSuccessor. 
			In this project, you will not be abstracting to simplified states.

			Use self.evaluationFunction in your definition of Vmax,min wherever you used Eval(s) in your on paper description

			The minimax values of the initial state in the minimaxClassic layout are 9, 8, 7, -492 for depths 1, 2, 3 and 4 respectively. 
			You can use these numbers to verify if your implementation is correct. Note that your minimax agent will often win 
			(just under 50% of the time for us--be sure to test on a large number of games using the -n and -q flags) despite the 
			dire prediction of depth 4 minimax.

			Here is an example call to run the program: 
					python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
			Note that this is designed for python 2.7 based on print statements so you may need to replace "python" with "python2.7"

			You can assume that you will always have at least one action from which to choose in getAction.
			You are welcome to, but not required to, remove Directions.STOP as a valid action 
			(this is true for any part of the assignment).
			If there is a tie between multiple actions for the best move, you may break the tie however you see fit.

			THIS IS THE ONLY FILE YOU NEED TO EDIT. 
			pacman.py runs the pacman game and describes the GameState type
			game.py contains logic about the world - agent, direction, grid.
			util.py: data structures for implementing search algirthms
			graphicsDisplay.py does what the title says
			graphicsUtils.py is just more support for graphics
			textDisplay is just for ASCII displays
			ghostAgents are the agents that control the ghosts
			keyboardAgents is what allows you to control pacman
			layout.py reads files and stores their contents.
"""


class ReflexAgent(Agent):
	"""
		A reflex agent chooses an action at each choice point by examining
		its alternatives via a state evaluation function.

		The code below is provided as a guide.    You are welcome to change
		it in any way you see fit, so long as you don't touch our method
		headers.
	"""
	def __init__(self):
				
		self.lastPositions = []
		self.dc = None


	def getAction(self, gameState):
		"""
		getAction chooses among the best options according to the evaluation function.

		getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
		------------------------------------------------------------------------------
		Description of GameState and helper functions:

		A GameState specifies the full game state, including the food, capsules,
		agent configurations and score changes. In this function, the |gameState| argument 
		is an object of GameState class. Following are a few of the helper methods that you 
		can use to query a GameState object to gather information about the present state 
		of Pac-Man, the ghosts and the maze.
		
		gameState.getLegalActions(): 
				Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

		gameState.generateSuccessor(agentIndex, action): 
				Returns the successor state after the specified agent takes the action. 
				Pac-Man is always agent 0.

		gameState.getPacmanState():
				Returns an AgentState object for pacman (in game.py)
				state.configuration.pos gives the current position
				state.direction gives the travel vector

		gameState.getGhostStates():
				Returns list of AgentState objects for the ghosts

		gameState.getNumAgents():
				Returns the total number of agents in the game

		gameState.getScore():
				Returns the score corresponding to the current state of the game

		
		The GameState class is defined in pacman.py and you might want to look into that for 
		other helper methods, though you don't need to.
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best


		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (oldFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		oldFood = currentGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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
		multi-agent searchers.    Any methods defined here will be available
		to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

		You *do not* need to make any changes here, but you can if you want to
		add functionality to all your adversarial search agents.    Please do not
		remove anything, however.

		Note: this is an abstract class: one that should not be instantiated.    It's
		only partially specified, and designed to be extended.    Agent (game.py)
		is another abstract class.
	"""

	def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)
		self.numRec = 0

######################################################################################
# Problem 1: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
	"""
		Your minimax agent (problem 1)
	"""

	def getAction(self, gameState):
		"""
			Returns the minimax action from the current gameState using self.depth
			and self.evaluationFunction. Terminal states can be found by one of the following: 
			pacman won, pacman lost or there are no legal moves. 

			Here are some method calls that might be useful when implementing minimax.

			gameState.getLegalActions(agentIndex):
				Returns a list of legal actions for an agent
				agentIndex=0 means Pacman, ghosts are >= 1

			Directions.STOP:
				The stop direction, which is always legal

			gameState.generateSuccessor(agentIndex, action):
				Returns the successor game state after an agent takes an action

			gameState.getNumAgents():
				Returns the total number of agents in the game

			gameState.getScore():
				Returns the score corresponding to the current state of the game
		
			gameState.isWin():
				Returns True if it's a winning state
		
			gameState.isLose():
				Returns True if it's a losing state

			self.depth:
				The depth to which search should continue

	
			This function should use Vmax,min to determine the best action for Pac-Man. Consider
			defining another function within this function that you just call to get your result. Hint:
			That function should be recursive, and the initial call should include self.depth as a parameter.

			One thing to consider is when the "depth" should decrement by one. Why are you decrementing?
			If you scroll up in init you can see that the default is "2" which means that you should go to depths
			0, 1, and 2. It's easiest to do so by starting at depth 2, then going to depth 1, then depth 0, and on
			depth 0 doing "something special" (think about what is reasonable). 
			Another thing to consider is when you should "stop."
		"""


		self.numRec = 0
		val, act = self.valueMinimax(gameState, self.depth, self.index)
		print(self.numRec)
		return act


	def valueMinimax(self, state, d, currAgent):
		
		self.numRec += 1

		nextAgent = currAgent + 1
		nextAgent %= state.getNumAgents()

		if state.isWin() or state.isLose():
			return state.getScore(), None
		
		elif d == 0:
			return self.evaluationFunction(state), None
		
		elif currAgent == self.index:
			legalActions = state.getLegalActions(currAgent)
			maxVal, maxAction = None, None

			isFirstIter = True
			
			for a in legalActions:
				val, _ = self.valueMinimax(state.generateSuccessor(currAgent, a), d, nextAgent)
				if isFirstIter or val > maxVal:
					maxVal = val
					maxAction = a
					isFirstIter = False

			return maxVal, maxAction

		elif currAgent > self.index:
			legalActions = state.getLegalActions(currAgent)
			minVal, minAction = None, None

			isFirstIter = True

			for a in legalActions:
				val, _ = self.valueMinimax(state.generateSuccessor(currAgent, a), d - 1 if nextAgent == self.index else d, nextAgent)
				if isFirstIter or val < minVal:
					minVal = val
					minAction = a
					isFirstIter = False

			return minVal, minAction
		
		else:
			raise Exception("Literally impossible")

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
		Your expectimax agent (problem 3)
	"""

	def getAction(self, gameState):
		"""
			Returns the expectimax action using self.depth and self.evaluationFunction

			All ghosts should be modeled as choosing uniformly at random from their
			legal moves.
		"""

		val, act = self.valueExpectimax(gameState, self.depth, self.index)
		return act


		# BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
		#no i will not. raise Exception("Not implemented yet")
		# END_YOUR_CODE

	def valueExpectimax(self, state, d, currAgent):
		
		nextAgent = currAgent + 1
		nextAgent %= state.getNumAgents()

		if state.isWin() or state.isLose():
			return state.getScore(), None
		
		elif d == 0:
			return self.evaluationFunction(state), None
		
		elif currAgent == self.index:
			legalActions = state.getLegalActions(currAgent)
			maxVal, maxAction = None, None

			isFirstIter = True
			
			for a in legalActions:
				val, _ = self.valueExpectimax(state.generateSuccessor(currAgent, a), d, nextAgent)
				if isFirstIter or val > maxVal:
					maxVal = val
					maxAction = a
					isFirstIter = False

			return maxVal, maxAction

		elif currAgent > self.index:
			legalActions = state.getLegalActions(currAgent)
			totalVal = 0.0

			for a in legalActions:
				val, _ = self.valueExpectimax(state.generateSuccessor(currAgent, a), d - 1 if nextAgent == self.index else d, nextAgent)
				totalVal += val

			avVal = totalVal / len(legalActions)

			randAct = legalActions[random.randint(0, len(legalActions) - 1)]

			return avVal, randAct
		
		else:
			raise Exception("Literally impossible")

######################################################################################
# BONUS PROBLEM: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
		Your minimax agent with alpha-beta pruning 
		Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in AlphaBetaAgent. 
		Again, your algorithm will be slightly more general than the pseudo-code in the slides, so part of the challenge 
		is to extend the alpha-beta pruning logic appropriately to multiple minimizer agents.
		You should see a speed-up (perhaps depth 3 alpha-beta will run as fast as depth 2 minimax). Ideally, depth 3
		on mediumClassic should run in just a few seconds per move or faster.
		Here is an example of how to call the program (again, you may need to sub in "python2.7" instead of "python")
				python pacman.py -p AlphaBetaAgent -a depth=3
		The AlphaBetaAgent minimax values should be identical to the MinimaxAgent minimax values, although the actions 
		it selects can vary because of different tie-breaking behavior. Again, the minimax values of the initial state in 
		the minimaxClassic layout are 9, 8, 7, and -492 for depths 1, 2, 3, and 4, respectively. Running the command given 
		above this paragraph, the minimax values of the initial state should be 9, 18, 27, and 36 for depths 1, 2, 3, and 4, respectively.


	"""

	def getAction(self, gameState):
		"""
			Returns the minimax action using self.depth and self.evaluationFunction
		"""

		self.numRec = 0

		val, act = self.valueMinimax(gameState, self.depth, self.index, float("-inf"), float("inf"))

		print(self.numRec)

		return act


	def valueMinimax(self, state, d, currAgent, a, b):
		
		self.numRec += 1

		nextAgent = currAgent + 1
		nextAgent %= state.getNumAgents()

		if state.isWin():
			# return float("inf"), None
			return state.getScore() + 10e6, None

		elif state.isLose():
			# return float("-inf"), None
			return -state.getScore() - 10e6, None
		
		elif d == 0:
			return self.evaluationFunction(state), None
		
		elif currAgent == self.index:
			legalActions = state.getLegalActions(currAgent)
			legalActions.remove(Directions.STOP)
			maxVal, maxAction = float("-inf"), Directions.STOP

			for act in legalActions:

				succState = state.generateSuccessor(currAgent, act)
				val, _ = self.valueMinimax(succState, d, nextAgent, a, b)

				if val > maxVal:
					maxVal = val
					maxAction = act

				a = max(a, maxVal)

				if b <= a:
					# print "| " * (self.depth - d), "PRUNE", currAgent, "val:", val, "action", act, "maxVal:", maxVal, "maxAction:", maxAction, "a:", a, "b:", b
					break

				# print "| " * (self.depth - d), currAgent, "val:", val, "action", act, "maxVal:", maxVal, "maxAction:", maxAction, "a:", a, "b:", b

			
			return maxVal, maxAction

		elif currAgent > self.index:

			pacPos = state.getPacmanPosition()
			ghostPos = state.getGhostPosition(currAgent)

			if manhattanDistance(pacPos, ghostPos) > d * 2:
				val, _ = self.valueMinimax(state, d - 1 if nextAgent == self.index else d, nextAgent, a, b)
				return val, None

			legalActions = state.getLegalActions(currAgent)
			minVal, minAction = float("inf"), None

			for act in legalActions:
				val, _ = self.valueMinimax(state.generateSuccessor(currAgent, act), d - 1 if nextAgent == self.index else d, nextAgent, a, b)
				if val < minVal:
					minVal = val
					minAction = act

				b = min(b, minVal)

				if b <= a:
					# print "| " * (self.depth - d), currAgent, "val:", val, "action", act, "minVal:", minVal, "minAction:", minAction, "a:", a, "b:", b
					break

				# print "| " * (self.depth - d), currAgent, "val:", val, "action", act, "minVal:", minVal, "minAction:", minAction, "a:", a, "b:", b

			return minVal, minAction
		
		else:
			raise Exception("Literally impossible")


######################################################################################
# BONUS PROBLEM: creating a better evaluation function (hint: consider generic search algorithms)

def betterEvaluationFunction(currentGameState):
	"""
		Your extreme, unstoppable evaluation function 
		 Write a better evaluation function for Pac-Man in the provided function betterEvaluationFunction. 
		 The evaluation function should evaluate states (rather than actions). You may use any tools at your disposal 
		 for evaluation, including any util.py code from the previous assignments. With depth 2 search, your 
		 evaluation function should clear the smallClassic layout with two random ghosts more than half the time 
		 for full credit and still run at a reasonable rate.
		 Here's how to call it:
		python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n 10
	
		
		Rule 1:
		say double when card change not change color and knock
		Face card twice = 4
		other card twice = value + 2

		Rule 2:
		Difference of two card values

		prev, 	this, 	...
		6 		3: 		3 5 8
		5 		8: 		3 10 13
		6 		7: 		1 9 10
		7 		7: 		9
		3 		7: 		4 5
		6 		A: 		5 8
		6 		7: 		1 9
		K(13) 	9: 		4 11  
		9 		2: 		7 4 3?
		9 		1: 		8 3
		A(1) 	2: 		3 1 4
		2 		10: 	8 12
		10 		8: 		2 10
		8 		8: 		10
		4 		K(13): 	9 4
		K(13) 	5: 		7 7
		5 		9: 		4 11
		J(11) 	10: 	1 12
		10 		4: 		6 6
		4 		5: 		1 2
		5 		2: 		3 4 
		2 		2: 		4
		2 		Q(12): 	10 4
		4 		4: 		6 6
		4 		Q(12): 	8 4 12
		Q(12) 	Q(12):	4
		Q(12) 	J(11):	1 4 5 
		7 		A(1): 	3 6 9
		A(1) 	K(13): 	4 12 16
		A(1) 	3: 		2 5 7
		Q(12) 	K(13):	1 4
		K(13) 	2: 		4 11
		J(11) 	J(11): 	4
		J(11) 	10: 	1 12 13 
		3		7:		4 9 13
		7 		8: 		1 10
		8 		8: 		10 10
		8 		10: 	2 12 14 
		10 		10: 	12
		10		7:		3 9
		7 		4: 		3 6
		4 		J(11): 	4 7
		5 		10:		5 7
		5		J(11):	4 6
		9 		8: 		1 10
		8 		8: 		10
		8 		5: 		3 7
		5 		6: 		1 8
		6 		Q(12): 	4 6
		7 		A(1): 	6 3
		A(1)	A(1):	3
		A(1) 	2: 		4 1 5
		3 		2:		4 1 5
		2 		2: 		4
		10 		2: 		4 8

		DESCRIPTION: <write something here so we know what you did>
	"""

	score = currentGameState.getScore()
	
	pacPos = currentGameState.getPacmanPosition()
	foods = currentGameState.getFood()

	maxDist = foods.height + foods.width
	minDist = maxDist	
	minFoodPos = None	

	for y in range(foods.height):
		for x in range(foods.width):
			if foods[x][y]:
				minDist = min(eucDist(pacPos, (x, y)), minDist)
				minFoodPos = (x, y)

	
	numFood = currentGameState.getNumFood()

	# ghostStates = currentGameState.getGhostStates()
	# scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

	
	return score + (maxDist - minDist) - numFood * 30
	# return score - numFood * 30




def eucDist(xy1, xy2):
	return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

class AStarNode:

	def __init__(self, pos):
		self.pos = pos
		self.parent = None
		self.hCost = 0
		self.gCost = float("inf")
		self.moveCost = 1

	def fCost(self):
		return self.hCost + self.gCost

	def __str__(self):
		return str(pos)

class AStarGrid:

	def __init__(self, walls):

		self.grid = [[None] * walls.height for _ in range(walls.width)]

		self.width = walls.width
		self.height = walls.height

		for x in range(self.width):
			for y in range(self.height):
				if not walls[x][y]:
					self.grid[x][y] = AStarNode((x, y))

	def getNeighbors(self, node):
		x, y = node.pos
		ns = []

		if x + 1 < self.width:
			ns.append(self.grid[x+1][y])
		if x - 1 >= 0:
			ns.append(self.grid[x-1][y])
		if y + 1 < self.height:
			ns.append(self.grid[x][y+1])
		if y - 1 >= 0:
			ns.append(self.grid[x][y-1])

		return (n for n in ns if n != None)

	def getNode(self, pos):
		return self.grid[pos[0]][pos[1]]

	def setMoveCost(self, pos, newMCost):
		n = self.getNode(pos)
		if n:
			n.moveCost = newMCost

	def setHCost(self, pos, newHCost):
		n = self.getNode(pos)
		if n:
			n.hCost = newHCost

	def printMoveCosts(self):
		s = ""
		for y in range(self.height - 1, -1, -1):
			for x in range(self.width):
				n = self.getNode((x,y))
				if n:
					if n.moveCost >= 10:
						s += str(n.moveCost) + " "
					else:
						s += " " + str(n.moveCost) + " "
				else:
					s += "## "
			s += "\n"
		print s



def generateHCost(pos, goal):
	return manhattanDistance(pos, goal)

def retracePath(goal):
	current = goal
	path = [current]
	while current.parent:
		path.append(current.parent)
		current = current.parent

	path.reverse()

	return path

def aStar(grid, start, goal):

	openQueue = PriorityQueue()
	openSet = set()
	closedSet = set()

	current = grid.getNode(start)

	if current == None:
		print("CURRENT IS NONE")
		return []

	current.gCost = 0

	openSet.add(current)
	openQueue.push(current, current.fCost())

	while openSet:
		current = openQueue.pop()

		# print current.pos

		if current.pos == goal:
			return retracePath(current)

		openSet.remove(current)
		closedSet.add(current)

		neighbors = grid.getNeighbors(current)

		for n in neighbors:
			if n in closedSet:
				continue

			newGCost = current.gCost + n.moveCost

			if newGCost >= n.gCost:
				continue

			n.parent = current
			n.gCost = newGCost

			if n not in openSet:
				openSet.add(n)
				openQueue.push(n, n.fCost())

	return []


# A STAR AGENT
class Team6Agent(MultiAgentSearchAgent):

	DIRS = {

		(-1,0) 	: Directions.WEST,
		( 1,0)	: Directions.EAST,
		( 0,1)	: Directions.NORTH,
		( 0,-1) : Directions.SOUTH

	}

	DXYS = ((-1,0),(1,0),(0,1),(0,-1))

	GHOST_NORMAL_L_1 = 99
	GHOST_NORMAL_L_2 = 50
	EMPTY_COST = 5
	FOOD_COST = 1

	currentTargetPos = None
	prevHasScaredGhost = False

	MAX_SCARED_TIMER = 1

	def getAction(self, state):

		# -------------
		# Define Variables
		# -------------

		pacState = state.getPacmanState()
		ghostStates = state.getGhostStates()
		
		pacPos = pacState.getPosition()
		ghostPositions = [(int(s.getPosition()[0]), int(s.getPosition()[1])) for s in ghostStates]

		ghostNeighborsLv1 = set()
		for i in range(len(ghostStates)):
			if ghostStates[i].scaredTimer > self.MAX_SCARED_TIMER:
				continue
			ghostPos = ghostPositions[i]
			ghostNeighborsLv1.add(ghostPos)
			for xy in self.DXYS:
				ghostNeighborsLv1.add((ghostPos[0] + xy[0], ghostPos[1] + xy[1]))

		ghostNeighborsLv2 = set()
		for i in range(len(ghostStates)):
			if ghostStates[i].scaredTimer > self.MAX_SCARED_TIMER:
				continue
			ghostPos = ghostPositions[i]
			for i in range(-1, 2):
				for j in range(-1, 2):
					ghostNeighborsLv2.add((ghostPos[0] + i, ghostPos[1] + j))


		# -------------
		# Init Grid
		# -------------

		walls = state.getWalls()
		foods = state.getFood()
		capsules = state.getCapsules()

		hasScaredGhost = False
		for gs in ghostStates:
			if gs.scaredTimer > self.MAX_SCARED_TIMER and gs.getPosition() not in ghostNeighborsLv1:
				hasScaredGhost = True
				break

		if hasScaredGhost:

			maxDist = walls.height + walls.width
			minDist = maxDist
			minGhostPos = None

			for i in range(len(ghostStates)):
				if ghostStates[i].scaredTimer > self.MAX_SCARED_TIMER:
					if manhattanDistance(pacPos, ghostPositions[i]) < minDist:
						minDist = manhattanDistance(pacPos, ghostPositions[i])
						minGhostPos = ghostPositions[i]

			self.currentTargetPos = minGhostPos

			self.prevHasScaredGhost = True

		else:

			if self.prevHasScaredGhost or self.currentTargetPos == None or pacPos == self.currentTargetPos or self.currentTargetPos in ghostNeighborsLv1 or self.currentTargetPos in ghostNeighborsLv2:

				self.prevHasScaredGhost = False

				# Random Pos
				# foodPositions = []
				# for x in range(foods.width):
				# 	for y in range(foods.height):
				# 		if foods[x][y] and (x,y) not in ghostNeighborsLv1:
				# 			foodPositions.append((x,y))
				# self.currentTargetPos = random.choice(foodPositions)

				# Min Pos
				maxDist = foods.height + foods.width
				minDist = maxDist
				minFoodPos = None

				for y in range(foods.height):
					for x in range(foods.width):
						if foods[x][y] and (x,y) not in ghostNeighborsLv1:
							if manhattanDistance(pacPos, (x, y)) < minDist:
								minDist = manhattanDistance(pacPos, (x, y))
								minFoodPos = (x, y)

				if minFoodPos == None:
					for i in range(-1, 2):
						for j in range(-1, 2):
							pos = (pacPos[0] + i, pacPos[1] + j)
							if walls[pos[0]][pos[1]] or pos in ghostNeighborsLv1 or pos in ghostNeighborsLv2 or pos == pacPos:
								continue
							self.currentTargetPos = pos
				else:
					self.currentTargetPos = minFoodPos

		grid = AStarGrid(walls)

		# H Cost
		for x in range(grid.width):
			for y in range(grid.height):
				grid.setHCost((x,y), generateHCost((x,y), self.currentTargetPos))

		# Move Cost
		for x in range(grid.width):
			for y in range(grid.height):
				if (x,y) in ghostNeighborsLv1:
					grid.setMoveCost((x,y), self.GHOST_NORMAL_L_1)
				elif (x,y) in ghostNeighborsLv2:
					grid.setMoveCost((x,y), self.GHOST_NORMAL_L_2)
				elif foods[x][y]:
					grid.setMoveCost((x,y), self.FOOD_COST)
				else:
					grid.setMoveCost((x,y), self.EMPTY_COST)


		print(hasScaredGhost, self.prevHasScaredGhost, self.currentTargetPos)
		
		# -------------
		# A* Search
		# -------------

		path = aStar(grid, pacPos, self.currentTargetPos)

		# print pacPos, "->", minFoodPos, "[" + ",".join(str(n.pos) for n in path) + "]"


		# -------------
		# Get Direction
		# -------------

		nx, ny = path[1].pos
		px, py = path[0].pos

		dxy = nx - px, ny - py

		act = self.DIRS[dxy]

		return act










