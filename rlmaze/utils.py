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
        sample maze (13 x 13)
        
        return:
            maze (array)
        """
        maze_name = 'maze_13x13'

        maze_grid = np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype = int )

        maze_start = np.array( [1, 1] )
        maze_finish = np.array( [maze_grid.shape[0]-2, maze_grid.shape[1]-2] )

        maze_grid[maze_start[0]][maze_start[1]] = 0
        maze_grid[maze_finish[0]][maze_finish[1]] = 0

        print('\nmaze_grid:')
        print(maze_grid)
        print('\nmaze_start:')
        print(maze_start)
        print('\nmaze_finish:')
        print(maze_finish)

        return maze_grid, maze_start, maze_finish, maze_name
        

    def sample_diagonal_maze(self):
        """
        
        return:
            maze (array)
        """
        maze_name = 'maze_diagonal'

        maze_grid = np.array(
                [[1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1]], dtype = int )

        maze_start = np.array( [1, 1] )
        maze_finish = np.array( [maze_grid.shape[0]-2, maze_grid.shape[1]-2] )

        maze_grid[maze_start[0]][maze_start[1]] = 0
        maze_grid[maze_finish[0]][maze_finish[1]] = 0

        print('\nmaze_grid:')
        print(maze_grid)
        print('\nmaze_start:')
        print(maze_start)
        print('\nmaze_finish:')
        print(maze_finish)

        return maze_grid, maze_start, maze_finish, maze_name


    def visualize_maze(
                        self,
                        maze_grid = None,
                        maze_start = None,
                        maze_finish = None,
                        maze_name:str = 'maze_example'):
        """ visualize maze 

        args:

        maze_grid: np.array
        maze_start: np.array
        maze_finish: np.array

        return:
            
        """
        print(maze_grid)
        maze_img = plt.imshow(maze_grid, cmap='Greys')
        plt.annotate('S', maze_start[::-1], color='red', fontsize=14)
        plt.annotate('F', maze_finish[::-1], color='red', fontsize=14)
        plt.savefig(self.fig_dir + maze_name + '.png', bbox_inches='tight')
        return maze_img

    
    def plot_epochs(self,
                    epoch_steps):
        """ plot required steps to escape maze with training 
        
        args:
        - epoch_steps (list): steps required in each epoch
        
        return:
        - 
        """
        plt.figure()
        plt.plot(epoch_steps)
        plt.title('Steps required to escape maze')
        plt.xlabel('Epoch [-]')
        plt.ylabel('Required steps [-]')
        plt.savefig(self.fig_dir + 'training.png', bbox_inches='tight')

        return