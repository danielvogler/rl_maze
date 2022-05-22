'''
Daniel Vogler

rl maze utils.py
- construct maze

'''

import numpy as np
import matplotlib.pyplot as plt
from .settings import Settings
from pathlib import Path

class Utils:

    def __init__(self):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT

        ### initialize folders
        self.fig_dir = self.PROJECT_ROOT + "/figures/"
        Path(self.fig_dir).mkdir(parents=True, exist_ok=True)
        return


    def sample_maze(self):
        """
        sample maze (11 x 8)
        
        return:
            maze (array)
        """

        maze_grid = np.array(
                [[0, 0, 1, 0, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 1, 1, 0],
                [0, 1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 1]], dtype = int )

        maze_start = np.array( [maze_grid.shape[0]-1, 0] )
        maze_finish = np.array( [0, maze_grid.shape[1]-1] )

        maze_grid[maze_start[0]][maze_start[1]] = -1
        maze_grid[maze_finish[0]][maze_finish[1]] = -1

        print('\nmaze_grid:')
        print(maze_grid)
        print('\nmaze_start:')
        print(maze_start)
        print('\nmaze_finish:')
        print(maze_finish)

        return maze_grid, maze_start, maze_finish
        

    def visualize_maze(
                        self,
                        maze_grid = None,
                        maze_start = None,
                        maze_finish = None):
        """ visualize maze 

        args:

        maze_grid: np.array
        maze_start: np.array
        maze_finish: np.array

        return:
            
        """

        if not maze_grid:
            print('Using sample maze')
            maze_grid, maze_start, maze_finish = self.sample_maze()

        plt.imshow(maze_grid)
        plt.annotate('S', maze_start[::-1], color='white', fontsize=14)
        plt.annotate('F', maze_finish[::-1], color='white', fontsize=14)
        plt.savefig(self.fig_dir + 'maze.png', bbox_inches='tight')
        return