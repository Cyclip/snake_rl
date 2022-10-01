import numpy as np

GRID_SIZE = 2 ** 4

class CellType:
    EMPTY = 0
    FOOD = 1
    BODY = 2
    HEAD = 3

class EnvReturnCode:
    SUCCESS = 0
    GAME_OVER = 1


class BodyCell:
    def __init__(self, pos, life):
        self.pos = np.copy(pos)
        self.life = life
    
    def step(self):
        """Returns True if the cell is still alive"""
        self.life -= 1
        return self.life > 0


class Game:
    """Snake game
    Cell IDs:
    0: Empty
    1: Snake head
    2: Snake body
    3: Food

    Position is in format [y, x]
    """
    DIRECTION_UP = np.array([0, -1])
    DIRECTION_DOWN = np.array([0, 1])
    DIRECTION_LEFT = np.array([-1, 0])
    DIRECTION_RIGHT = np.array([1, 0])

    def __init__(self):
        self.headPos = np.array([GRID_SIZE / 2, GRID_SIZE / 2], dtype=int)
        self.bodyPos = []
        self.foodPos = self.__generate_food_pos()
        self.life = 3
        self.direction = np.array([0, 1], dtype=int)
    
    def step(self, direction):
        """Step game"""
        # Update direction
        self.direction = direction

        # Update bodys
        self.__update_body()

        # Update head position
        self.headPos += self.direction

        # Return game over if head is out of bounds
        if self.headPos[0] < 0 or self.headPos[0] >= GRID_SIZE or self.headPos[1] < 0 or self.headPos[1] >= GRID_SIZE:
            return EnvReturnCode.GAME_OVER

        # Check if head is in food cell
        if (self.headPos == self.foodPos).all():
            self.life += 1
            self.foodPos = self.__generate_food_pos()
        
        # Check if head is in body cell
        for bodyPos in self.bodyPos:
            if (bodyPos.pos == self.headPos).all():
                return EnvReturnCode.GAME_OVER
        
        return EnvReturnCode.SUCCESS
    
    def display(self):
        """Display game"""
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid[self.headPos[0], self.headPos[1]] = CellType.HEAD
        grid[self.foodPos[0], self.foodPos[1]] = CellType.FOOD
        for bodyPos in self.bodyPos:
            grid[bodyPos.pos[0], bodyPos.pos[1]] = CellType.BODY
        
        grid_str = '\n'.join(["|" + ' '.join([self.__display_val(i) for i in row]) + "|" for row in grid])
        return " " + "_" * (2*GRID_SIZE - 1) + "\n" + grid_str + "\n" + " " + "‾" * (2*GRID_SIZE - 1)
    
    def __display_val(self, val):
        if val == CellType.HEAD:
            return '■'
        elif val == CellType.BODY:
            return '□'
        elif val == CellType.FOOD:
            return '◆'
        else:
            return ' '
    
    def __update_body(self):
        # Step all of the body cells and remove dead ones
        for bodyCell in self.bodyPos:
            if not bodyCell.step():
                self.bodyPos.remove(bodyCell)
        
        # Place new body at headPos (as it will be updated)
        self.bodyPos.append(BodyCell(self.headPos, self.life))

    def __get_cell(self, pos):
        """Get cell"""
        # Check if in head cell
        if pos[0] == self.headPos[0] and pos[1] == self.headPos[1]:
            return CellType.HEAD
        
        # Check if in food cell
        if pos[0] == self.foodPos[0] and pos[1] == self.foodPos[1]:
            return CellType.FOOD

        # Check if in body cell
        for bodyPos in self.bodyPos:
            if pos[0] == bodyPos[0] and pos[1] == bodyPos[1]:
                return CellType.BODY
        
        return 0

    def __generate_food_pos(self):
        """Generate food position"""
        while True:
            pos = np.random.randint(0, GRID_SIZE, 2)
            if self.__is_cell_empty(pos):
                return pos
    
    def __is_cell_empty(self, pos):
        """Check if cell is empty"""
        # Check if in head cell
        if pos[0] == self.headPos[0] and pos[1] == self.headPos[1]:
            return False
        
        # Check if in body cell
        for bodyPos in self.bodyPos:
            if (bodyPos == pos).all():
                return False
        return True