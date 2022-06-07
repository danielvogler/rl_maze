'''
Daniel Vogler

rlmaze.py
RL learning setup and algo

'''

import numpy as np
import logging
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from .settings import Settings
from .utils import Utils



class RLMaze: 

    def __init__(self,
                cfg_file):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT

        self.cfg = Settings().config( cfg_file )

        ### environment
        self.actions = { 0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0] }
        self.states = tf.convert_to_tensor( self.cfg.maze_grid.flatten(), np.int64)

        ### maze properties
        self.cfg.maze_dim = self.cfg.maze_grid.shape
        self.maze_start = tf.convert_to_tensor( [[self.cfg.maze_start[0] * self.cfg.maze_dim[0] + self.cfg.maze_start[1]]], np.int64)
        self.maze_finish = tf.convert_to_tensor( [[self.cfg.maze_finish[0] * self.cfg.maze_dim[0] + self.cfg.maze_finish[1]]], np.int64)
        self.state = self.maze_start

        ### dimensions
        self.actions_dim = len(self.actions)
        self.states_dim = len(self.states)

        return

   
    def escape_maze(self):
        """ initialize q-learning setup 
        
        args:

        return:
        - 
        """

        ### construct Q-DNN
        qdnn = self.initialize_qdnn(self.states_dim, self.actions_dim)

        ### states identity to feed states into QDNN
        states_identity = np.identity(len(self.states))

        epoch_steps = []
        epsilon = self.cfg.epsilon

        ### train maze escaping
        for i in range( self.cfg.epochs ):

            logging.debug(f'Epoch ({i}/{self.cfg.epochs})')

            ### initialize system
            state = self.maze_start
            self.done = False
            action_counter = 0

            epsilon *= self.cfg.epsilon_decay

            while self.done == False:
                action_counter += 1

                ### q-values for current state
                old_state = state
                q_values_old = qdnn.predict( tf.convert_to_tensor( states_identity[state], np.int64), verbose=0 )

                ### explore/exploit
                if np.random.uniform(low=0, high=1) < epsilon:
                    ### explore -> randomly choose action from discrete action space
                    action_arg = np.random.randint(low=0, high=len(self.actions))

                else:
                    ### exploit -> choose highest scoring action
                    action_arg = np.argmax(q_values_old)

                ### determine action, get new state + reward
                action = self.actions[ action_arg ]
                state = old_state + action[0] * self.cfg.maze_dim[0] + action[1]
                state, reward = self.compute_reward( old_state, state)

                ### q values for new state
                q_values_state = qdnn.predict( tf.convert_to_tensor( states_identity[state], np.int64), verbose=0 )

                ### constrain reward if maze is completed
                if self.done == True:
                    target = reward

                elif self.done == False:
                    target = (reward + self.cfg.gamma * np.max( q_values_state ) )

                q_values_old[0, action_arg] = target

                ### fit model with updated targets
                qdnn.fit(   tf.convert_to_tensor( states_identity[old_state], np.int64), q_values_old, epochs=1, verbose=0)

            logging.info(f'Epoch ({i}/{self.cfg.epochs}): epsilon ({epsilon}) - actions ({action_counter})')

            ### keep track of required steps in epoch
            epoch_steps.append(action_counter)

        # self.agent_location(state)
        logging.info(f'Epoch learning: \n {epoch_steps}')
        Utils().plot_epochs(epoch_steps, self.cfg.maze_name)

        return


    def initialize_qdnn(self, state_dim: int, action_dim: int) -> keras.Model:
        """ Initialize Q-DNN 
        
        args: 
        - state_dim (int): number of states
        - action_dim (int): number of actions
        
        return:
        - model (keras.Model): QDNN model to train
        """

        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, state_dim)))
        model.add(Dense( self.cfg.layer_nodes, activation='relu'))
        model.add(Dense( self.cfg.layer_nodes, activation='relu'))
        model.add(Dense(action_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model


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
        if new_state == self.maze_finish:
            new_state = new_state
            reward = self.cfg.reward_finish
            logging.debug('Maze finish')
            self.done = True

        ### perform action
        elif tf.gather(self.states, new_state) == 0:
            new_state = new_state
            reward = self.cfg.reward_active
            logging.debug('Move forward - new_state')

        ### check for wall
        elif tf.gather(self.states, new_state) == 1:
            new_state = old_state
            reward = self.cfg.reward_wall
            logging.debug('Maze wall - use old_state')

        else:
            logging.debug('Maze entry not valid')

        return new_state, reward


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