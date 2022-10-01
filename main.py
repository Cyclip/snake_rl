import time
import numpy as np
import keyboard
from game import Game, EnvReturnCode
import os

clear = lambda: os.system('cls')

FPS = 8

def get_direction():
    """Get arrow key direction"""
    if keyboard.is_pressed('up'):
        return Game.DIRECTION_LEFT
    elif keyboard.is_pressed('down'):
        return Game.DIRECTION_RIGHT
    elif keyboard.is_pressed('left'):
        return Game.DIRECTION_UP
    elif keyboard.is_pressed('right'):
        return Game.DIRECTION_DOWN
    else:
        return None

game = Game()
direction = np.array([0, 1])

prev = time.time()

while True:
    # Get direction
    now = time.time()
    while now - prev < 1/FPS:
        new_direction = get_direction()
        if new_direction is not None:
            direction = new_direction
        now = time.time()

    result = game.step(direction)
    # print(f"{game.headPos} {game.foodPos} {game.life} \n" + game.display())
    clear()
    print(game.display())
    prev = now

    if result == EnvReturnCode.GAME_OVER:
        break