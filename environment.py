import numpy as np
from game import Game, EnvReturnCode, GRID_SIZE

class Environment(Game):
    def __init__(self):
        super().__init__()
        self.iteration = 0
    
    def reset(self):
        """Reset the game state"""
        self.__init__()
    
    def step(self, action):
        """Play one step in the environment"""
        # Action formatted as turning [straight, left, right]
        # [0 0 0] -> straight
        # [0 1 0] -> left
        # [0 0 1] -> right

        # Get direction
        if action[1] == 1:
            self.direction = self.__rotate_left(self.direction)
        elif action[2] == 1:
            self.direction = self.__rotate_right(self.direction)

        if self.iteration < 100 * self.life:
            self.iteration += 1
            result, reward, score = self.play()
            return result, reward, score
        
        return EnvReturnCode.GAME_OVER, -10, self.score
        
    def get_state(self):
        """Get state of the environment
        State:
            - Distance to wall (straight)
            - Distance to wall (left)
            - Distance to wall (right)
            - Distance to body (straight)
            - Distance to body (left)
            - Distance to body (right)
            - Direction horizontal (-1 left or 1 right)
            - Direction vertical (-1 up or 1 down)
            - Angle to food (cos θ)
        """

        # Get direction
        if self.direction[0] == 0:
            direction_horizontal = 0
            direction_vertical = self.direction[1]
        else:
            direction_horizontal = self.direction[0]
            direction_vertical = 0

        # Get distance to wall
        distance_to_wall_straight = self.__get_distance_to_wall(self.direction)
        distance_to_wall_left = self.__get_distance_to_wall(self.__rotate_left(self.direction))
        distance_to_wall_right = self.__get_distance_to_wall(self.__rotate_right(self.direction))

        # Get distance to body
        distance_to_body_straight = self.__get_distance_to_body(self.direction)
        distance_to_body_left = self.__get_distance_to_body(self.__rotate_left(self.direction))
        distance_to_body_right = self.__get_distance_to_body(self.__rotate_right(self.direction))

        # Get angle to food
        angle_to_food = self.__get_angle_to_food()

        # Return state
        return np.array([
            distance_to_wall_straight,
            distance_to_wall_left,
            distance_to_wall_right,
            distance_to_body_straight,
            distance_to_body_left,
            distance_to_body_right,
            direction_horizontal,
            direction_vertical,
            angle_to_food
        ])
    
    def __rotate_left(self, dir):
        """Rotate direction left"""
        return np.array([-dir[1], dir[0]])
    
    def __rotate_right(self, dir):
        """Rotate direction right"""
        return np.array([dir[1], -dir[0]])
    
    def __get_angle_to_food(self):
        """Get cos of angle from head to food"""
        θ = np.arctan2(self.foodPos[0] - self.headPos[0], self.foodPos[1] - self.headPos[1])
        return np.cos(θ)
    
    def __get_distance_to_wall(self, direction):
        """Get distance to wall in direction (normalized)"""
        distance = 0
        pos = self.headPos + direction
        while self.__is_valid_position(pos):
            distance += 1
            pos += direction
        return distance / GRID_SIZE
    
    def __get_distance_to_body(self, direction):
        """Get distance to body in direction (normalized)"""
        distance = 0
        pos = self.headPos + direction
        while self.__is_valid_position(pos):
            for bodyPos in self.bodyPos:
                if (bodyPos.pos == pos).all():
                    return distance / GRID_SIZE
            distance += 1
            pos += direction
        return distance / GRID_SIZE