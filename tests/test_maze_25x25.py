'''
Daniel Vogler

test_maze_25x25.py
Create, visualize and train on 25x25 maze

'''
import logging
from rlmaze.rlmaze import RLMaze
from rlmaze.utils import Utils


logging.basicConfig(level=logging.INFO)

Utils().visualize_maze('maze_25x25')
RLMaze('maze_25x25').escape_maze()