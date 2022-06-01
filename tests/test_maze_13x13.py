'''
Daniel Vogler

test_maze_13x13.py
Create, visualize and train on 13x13 maze

'''
import logging
from rlmaze.rlmaze import RLMaze
from rlmaze.utils import Utils

logging.basicConfig(level=logging.INFO)

Utils().visualize_maze('maze_13x13')
RLMaze().escape_maze('maze_13x13')