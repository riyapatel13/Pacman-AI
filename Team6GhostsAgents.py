from game import Agent
from game import Actions
from game import Directions
import random, util, math
from util import *

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist


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
		#print("CURRENT IS NONE")
		return []

	current.gCost = 0

	openSet.add(current)
	openQueue.push(current, current.fCost())

	while openSet:
		current = openQueue.pop()

		# #print current.pos

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

def eucDist(xy1, xy2):
	return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)


class Team6Ghosts(GhostAgent):

	DIRS = {

		(-1,0) 	: Directions.WEST,
		( 1,0)	: Directions.EAST,
		( 0,1)	: Directions.NORTH,
		( 0,-1) : Directions.SOUTH

	}

	DXYS = {

		Directions.WEST : (-1,0),
		Directions.EAST : ( 1,0),
		Directions.NORTH : ( 0,1),
		Directions.SOUTH : ( 0,-1),
		Directions.STOP : (0, 0)

	}

	PAC_WALL_COST = 10
	FOOD_COST = 1

	DEPTH = 4

	def getAction(self, state):

		actions = state.getLegalActions(self.index)
		# #print (actions)

		# -------------
		# Define Variables
		# -------------


		ghostState = state.getGhostState(self.index)
		pacState = state.getPacmanState()

		pacPos = pacState.getPosition()
		pacDir = pacState.getDirection()

		ghostPosDec = ghostState.getPosition()

		ghostPosInt = int(ghostPosDec[0]), int(ghostPosDec[1])

		# #print(ghostPos)


		# -------------
		# Init Grid
		# -------------

		walls = state.getWalls()
		foods = state.getFood()
		capsules = state.getCapsules()

		grid = AStarGrid(walls)

		for x in range(grid.width):
			for y in range(grid.height):
				grid.setHCost((x,y), generateHCost((x,y), pacPos))


		# -------------
		# Ghost Specific Grid
		# -------------

		# behind pacman
		if self.index == 1:
			fpos = (pacPos[0] + self.DXYS[pacDir][0], pacPos[1] + self.DXYS[pacDir][1])
			grid.setMoveCost(fpos, self.PAC_WALL_COST)

		# in front pacman
		elif self.index == 2:
			bpos = (pacPos[0] - self.DXYS[pacDir][0], pacPos[1] - self.DXYS[pacDir][1])
			grid.setMoveCost(bpos, self.PAC_WALL_COST)

		# All directions
		elif self.index == 4:
			pass

		# random location
		elif self.index == 3:
			emptyLocations = []
			for x in range(walls.width):
				for y in range(walls.height):
					if not walls[x][y]:
						emptyLocations.append((x,y))

			self.currentTargetPos = random.choice(emptyLocations)

			for x in range(walls.width):
				for y in range(walls.height):
					if not walls[x][y]:
						grid.setMoveCost((x,y), (pacPos[0] - x) ** 2 + (pacPos[1] - y) ** 2)


		# -------------
		# A* Search
		# -------------

		path = aStar(grid, ghostPosInt, pacPos)

		if len(path) == 0:
			#print("NO PATH")
			return random.choice(actions)

		# -------------
		# Get Direction
		# -------------

		nx, ny = path[1].pos
		px, py = path[0].pos

		dxy = nx - px, ny - py

		act = self.DIRS[dxy]

		# -------------
		# RUN AWAY
		# -------------

		if len(path) < ghostState.scaredTimer * 1.5:
			maxDistFromPac = -1
			maxDistFromPacAct = None

			for a in actions:

				walls = state.getWalls()
				grid = AStarGrid(walls)
				for x in range(grid.width):
					for y in range(grid.height):
						grid.setHCost((x,y), generateHCost((x,y), ghostPosInt))

				newGhostPos = (ghostPosInt[0] + self.DXYS[a][0], ghostPosInt[1] + self.DXYS[a][1])

				newPath = aStar(grid, newGhostPos, pacPos)

				if len(newPath) > maxDistFromPac:
					maxDistFromPac = len(newPath)
					maxDistFromPacAct = a

			return maxDistFromPacAct



		# -------------
		# Last Check and Return
		# -------------

		if act not in actions:
			act = random.choice(actions)

		return act
