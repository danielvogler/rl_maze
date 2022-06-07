import os
import configparser
import numpy as np

class Settings:

    def __init__(self):
        self.PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = os.path.dirname(self.PACKAGE_ROOT)
        return

    def config(self,
                cfg_file: str = None):
        """ open config file 
        
        args:
        - cfg file (str): config file name
        
        return:
        -
        """
        cfg = configparser.ConfigParser()

        cfg_file_path = str(self.PROJECT_ROOT + '/config/')
        cfg_file = str(cfg_file_path + cfg_file + '.ini')
        print(f'Reading config file: {cfg_file}')

        cfg.read( cfg_file )

        ### parameters
        ### learning rate (0 < alpha < 1)
        ###     -> high alpha, fast learning
        cfg.alpha = float( cfg['parameters']['alpha'] )
        ### exploration vs exploitation 
        ###     -> high epsilon, favor exploration
        cfg.epsilon = float( cfg['parameters']['epsilon'] )
        cfg.epsilon_decay = float( cfg['parameters']['epsilon_decay'] )
        ### discount factor (0 < alpha < 1) -
        ###     -> importance of long-term reward
        ###     -> high gamma, high long-term effective reward        ->
        cfg.gamma = float( cfg['parameters']['gamma'] )
        cfg.reward_wall = float( cfg['parameters']['reward_wall'] )
        cfg.reward_finish = float( cfg['parameters']['reward_finish'] )
        cfg.reward_active = float( cfg['parameters']['reward_active'] )
        cfg.reward_inactive = float( cfg['parameters']['reward_inactive'] )

        ### training
        cfg.epochs = int( cfg['training']['epochs'] )

        ### maze
        cfg.maze_name = str( cfg['maze']['name'] )
        cfg.maze_start = np.array( cfg['maze']['start'].split() ).astype(int)
        cfg.maze_finish = np.array( cfg['maze']['finish'].split() ).astype(int)

        maze_file = str(cfg_file_path + cfg.maze_name + '_grid.txt')
        cfg.maze_grid = np.loadtxt( maze_file )

        return cfg