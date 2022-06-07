'''
Daniel Vogler

rlmaze.py
RL learning setup and algo

'''

import numpy as np
import logging
from .settings import Settings
from .utils import Utils


class RLMaze: 

    def __init__(self):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT
        return

   
    def escape_maze(self,
                cfg_file):
        """ initialize q-learning setup 
        
        args:

        return:
        - 
        """
        self.cfg = Settings().config( cfg_file )

        self.cfg.maze_dim = self.cfg.maze_grid.shape

        actions = { 0: [0,0], 1: [0,1], 2: [0,-1], 3: [1,0], 4: [-1,0] }
        states = np.array([ [x, y] for x in range(0, self.cfg.maze_dim[0]) for y in range(0, self.cfg.maze_dim[1]) ])
        q_table = np.zeros( [len(states), len(actions)] )

        epoch_steps = []

        for i in range( self.cfg.epochs ):

            logging.debug(f'Epoch ({i}/{self.cfg.epochs})')

            ### initialize
            state = self.cfg.maze_start

            self.maze_completed = False
            action_counter = 0

            # epsilon = i / epochs

            while self.maze_completed == False and action_counter < 100000:
                action_counter += 1

                action_arg = self.perform_action(state, actions, q_table)

                new_action = actions[ action_arg ]

                ### get new state
                old_state = state
                new_state = old_state + new_action

                new_state, reward = self.compute_reward(
                                            old_state,
                                            new_state)

                logging.debug(f'\nnew_action ({action_counter}): {new_action} state: {old_state} -> {new_state}')

                q_table[ old_state[0]*self.cfg.maze_dim[0] + old_state[1] ][ action_arg ] = \
                    (1- self.cfg.alpha) * q_table[ old_state[0]*self.cfg.maze_dim[0] + old_state[1] ][ action_arg ] \
                    + self.cfg.alpha * (reward + self.cfg.gamma * np.max(q_table[ new_state[0]*self.cfg.maze_dim[0] + new_state[1] ]))

                ### if finished - print status
                if self.maze_completed == True:
                    logging.debug('Q:', q_table)

                state = new_state

            logging.info(f'Epoch ({i}/{self.cfg.epochs}): actions ({action_counter})')

            ### keep track of required steps
            epoch_steps.append(action_counter)

        self.agent_location(state)
        logging.info(f'Q: {q_table}')
        logging.info(f'Epoch learning: \n {epoch_steps}')
        Utils().plot_epochs(epoch_steps, self.cfg.maze_name)

        return


    def compute_reward(self,
                        old_state,
                        new_state):
        """ evaluate performed action
        
        args:
        - state (tuple): state before action was performed
        - new_state (tuple): state after action, has to be checked first
        
        return:
        - state (tuple): return (new) state of agent
        """

        ### check for maze finish
        if new_state[0] == self.cfg.maze_finish[0] and new_state[1] == self.cfg.maze_finish[1]:
            new_state = new_state
            reward = self.cfg.reward_finish
            logging.debug('Maze finish')
            self.maze_completed = True

        ### perform action
        elif self.cfg.maze_grid[new_state[0]][new_state[1]] == 0:
            ### punish inaction
            if new_state[0] == old_state[0] and new_state[1] == old_state[1]:
                reward = self.cfg.reward_inactive
                logging.debug('Inactive - new_state')

            ### reward action
            else:
                reward = self.cfg.reward_active
                logging.debug('Move forward - new_state')

            new_state = new_state

        ### check for wall
        elif self.cfg.maze_grid[new_state[0]][new_state[1]] == 1:
            new_state = old_state
            reward = self.cfg.reward_wall
            logging.debug('Maze wall - use old_state')

        else:
            logging.debug('Maze entry not valid')

        return new_state, reward


    def perform_action(self,
                    state,
                    actions,
                    q_table):
        """ perform action """

        ### explore/exploit
        if np.random.uniform(low=0, high=1) < self.cfg.epsilon:
            ### explore -> randomly choose action from discrete action space
            action_arg = np.random.randint(low=0, high=len(actions))

        else:
            ### exploit -> choose highest scoring action
            action_arg = np.argmax(q_table[ state[0]*self.cfg.maze_dim[0] + state[1] ])

        return action_arg


    def agent_location(self,
                    state):
        """ agent location
        
        args:
        - state (tuple): current state of agent
        
        return:

        """
        ### agent location
        agent_loc = np.copy( self.cfg.maze_grid[:] )
        agent_loc[state[0]][state[1]] = 5

        logging.info(f'\n Agent location (5) in maze:\n{agent_loc}')

        return