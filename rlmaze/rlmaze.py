'''
Daniel Vogler

rlmaze.py

TODO
- initialize Q table: states (cells) times actions (up, down, left, right)
'''

import numpy as np
from .settings import Settings

class RLMaze:

    def __init__(self):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT
        return

    def learning_setup(self,
                        maze_grid):
        """ initialize q-learning setup 
        
        args:
        - maze_grid: np.array = maze grid to traverse 

        return:
        - 
        """

        maze_dim = maze_grid.shape
        self.actions = [[0,0], [0,1], [0,-1], [1,0], [-1,0]]
        self.states = np.array([ [x, y] for x in range(0,maze_dim[0]) for y in range(0,maze_dim[1]) ])
        self.q_table_init = np.array( [ np.append(s, np.zeros(len(self.actions))) for s in self.states] )

        return


