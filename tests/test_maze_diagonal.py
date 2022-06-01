'''
Daniel Vogler

test_maze_diagonal.py
Create, visualize and train on diagonal maze

'''
import logging
from rlmaze.rlmaze import RLMaze
from rlmaze.utils import Utils

logging.basicConfig(level=logging.INFO)

Utils().visualize_maze('maze_diagonal')
RLMaze().escape_maze('maze_diagonal')