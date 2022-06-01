'''
Daniel Vogler

rl maze utils.py
- construct maze

'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from .settings import Settings

class Utils:

    def __init__(self):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT

        ### initialize folders
        self.fig_dir = self.PROJECT_ROOT + "/figures/"
        Path(self.fig_dir).mkdir(parents=True, exist_ok=True)
        return


    def visualize_maze(
                        self,
                        cfg_file:str):
        """ visualize maze 

        args:

        cfg_file (str)

        return:
            
        """
        cfg = Settings().config( cfg_file )

        plt.imshow(cfg.maze_grid, cmap='Greys')
        plt.annotate('S', cfg.maze_start[::-1], color='red', fontsize=14)
        plt.annotate('F', cfg.maze_finish[::-1], color='red', fontsize=14)
        plt.savefig(self.fig_dir + cfg.maze_name + '.png', bbox_inches='tight')
        return

    
    def plot_epochs(self,
                    epoch_steps,
                    maze_name):
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
        plt.savefig(self.fig_dir + maze_name + '_training.png', bbox_inches='tight')

        return