'''
Daniel Vogler

rlmaze.py

TODO
- initialize Q table: states (cells) times actions (up, down, left, right)
'''

import numpy as np
from .settings import Settings
from .utils import Utils

class RLMaze:

    def __init__(self):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT
        return

    def initialize_learning_setup(self,
                        maze_grid,
                        maze_start,
                        maze_finish):
        """ initialize q-learning setup 
        
        args:
        - maze_grid: np.array = maze grid to traverse 

        return:
        - 
        """
        self.maze_grid = maze_grid
        self.maze_start = maze_start
        self.maze_finish = maze_finish

        ### exploration vs exploitation 
        ###     -> high epsilon, favor exploration
        self.epsilon = 0.1

        ### learning rate (0 < alpha < 1)
        ###     -> high alpha, fast learning
        self.alpha = 0.15

        ### discount factor (0 < alpha < 1) -
        ###     -> importance of long-term reward
        ###     -> high gamma, high long-term effective reward        ->
        self.gamma = 0.5

        self.maze_dim = maze_grid.shape

        ### discrete action space (up, down, stay, left, right)
        self.actions = { 0: [0,0], 1: [0,1], 2: [0,-1], 3: [1,0], 4: [-1,0] }
        self.states = np.array([ [x, y] for x in range(0,self.maze_dim[0]) for y in range(0,self.maze_dim[1]) ])
        self.q_table = np.zeros( [len(self.states), len(self.actions)] )

        return

    
    def learn(self,
                maze_grid,
                maze_start,
                maze_finish):
        """ initialize q-learning setup 
        
        args:
        - maze_grid: np.array = maze grid to traverse 

        return:
        - 
        """
        self.initialize_learning_setup(maze_grid, maze_start, maze_finish)

        epochs = 200
        epoch_steps = []
        epoch_explore_ratio = []

        for i in range(epochs):

            print(f'\n\n\nEpoch ({i})')

            ### initialize
            state = self.maze_start

            self.maze_completed = False
            action_counter = 0
            explore_counter = 0

            # epsilon = i / epochs

            while self.maze_completed == False:
                action_counter += 1

                ### explore/exploit
                if np.random.uniform(low=0, high=1) < self.epsilon:
                    ### explore -> randomly choose action from discrete action space
                    action_number = np.random.randint(low=0, high=len(self.actions))
                    explore_counter += 1

                else:
                    ### exploit -> choose highest scoring action
                    action_number = np.argmax(self.q_table[ state[0]*self.maze_dim[0] + state[1] ])

                new_action = self.actions[ action_number ]

                ### get new state
                old_state = state
                new_state = state + new_action
                state = self.evaluate_action(state, new_state, action_number)

                print(f'\nnew_action ({action_counter}): {new_action} state: {old_state} -> {state}')

            ### keep track of required steps
            epoch_steps.append(action_counter)
            epoch_explore_ratio.append(explore_counter / action_counter )

        print(f'\nEpoch learning:\n{epoch_steps}')
        # print(f'\nExplore/exploit ratio:\n{epoch_explore_ratio}')
        Utils().plot_epochs(epoch_steps)

        return


    def evaluate_action(self,
                        state,
                        new_state,
                        action_number: int):
        """ evaluate performed action
        
        args:
        - state (tuple): state before action was performed
        - new_state (tuple): state after action, has to be checked first
        - action_number (int): number of action to perform
        
        return:
        - state (tuple): return (new) state of agent
        """
        old_state = state

        ### check for maze finish
        if new_state[0] == self.maze_finish[0] and new_state[1] == self.maze_finish[1]:
            state = new_state
            reward = 100
            print('Maze finish')
            self.maze_completed = True

        ### perform action
        elif self.maze_grid[new_state[0]][new_state[1]] == 0:
            ### punish inaction
            if action_number == 0:
                reward = -0.5
                print('Inactive - new_state')

            ### reward action
            else:
                reward = 0.5
                print('Move forward - new_state')

            state = new_state

        ### check for wall
        elif self.maze_grid[new_state[0]][new_state[1]] == 1:
            state = state
            reward = -1
            print('Maze wall - use old_state')

        else:
            print('Maze entry not valid')

        self.q_table[ old_state[0]*self.maze_dim[0] + old_state[1] ][ action_number ] = \
            (1- self.alpha) * self.q_table[ old_state[0]*self.maze_dim[0] + old_state[1] ][ action_number ] \
            + self.alpha * (reward + self.gamma * np.max(self.q_table[ state[0]*self.maze_dim[0] + state[1] ]))

        ### if finished - print status
        if self.maze_completed == True:
            print('\nQ:', self.q_table)
            self.agent_location(state)

        return state


    def agent_location(self,
                    state):
        """ agent location
        
        args:
        - state (tuple): current state of agent
        
        return:

        """
        ### agent location
        agent_loc = np.copy( self.maze_grid[:] )
        agent_loc[state[0]][state[1]] = 5

        print(f'\n Agent location (5) in maze:\n{agent_loc}')

        return