# RL_maze
Escape a maze using reinforcement learning

Generates a grid maze, with start, finish and walls, which an agent has to navigate. The agent can move through the maze by performing actions and learning from them.

## How-to

### Pre-requisites
- Install [poetry](https://python-poetry.org/) for dependency management and install venv:
  `poetry install`
- Open poetry shell:
  `poetry shell`
Then run the desired Python commands.

### Configuration
A complete configuration should be in `root/config/` and consists of:
- Config file: Maze settings, parameters, training
- Maze file: Text file containing a maze with walls (1) and hallways (0) and other features.

Choose settings in `root/config/<config_file>`:
- epochs
- learning rate $\alpha$
- discount rate $\gamma$
- exploration vs. exploitation $\epsilon$
- rewards

Custom mazes and configurations can easily be added.

### Run
RLMaze can easily be used as follows:

```
from rlmaze.rlmaze import RLMaze
RLMaze().escape_maze(<config_file>)
```

to train on a maze. The maze can be visualized with:

`Utils().visualize_maze(<config_file>)`

Some test cases can be found in `root/tests/' and run with:

`python tests/test_maze_diagonal.py`

or

`python tests/test_maze_13x13.py`


## Figures

### Diagonal maze
![Example image](/images/maze_diagonal.png "Diagonal maze layout")  
Fig 1: Diagonal maze layout with start (S) and finish (F).

![Example image](/images/maze_diagonal_training.png "Diagonal maze training")  
Fig 2: Diagonal maze training. Convergence to optimal number of steps (8). 

## 13x13 maze
![Example image](/images/maze_13x13.png "13x13 maze layout")  
Fig 3: Diagonal maze layout with start (S) and finish (F).

![Example image](/images/maze_13x13_training.png "13x13 maze training")  
Fig 4: 13x13 maze training. Convergence to optimal number of steps (20). 