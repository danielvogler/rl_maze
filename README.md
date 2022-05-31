# RL_maze
Escape a maze using reinforcement learning

## How-to

### Pre-requisites
- Install [poetry](https://python-poetry.org/) for dependency management and install venv:
  `poetry install`
- Open poetry shell:
  `poetry shell`
Then run the desired Python commands.

### Configuration
Choose settings in `root/rlmaze/rlmaze.py`:
- epochs
- learning rate $\alpha$
- discount rate $\gamma$
- exploration vs. exploitation $\epsilon$

### Run
`python tests.py`

### Figures

## Diagonal maze
![Example image](/images/maze_diagonal.png "Diagonal maze layout")  
Fig 1: Diagonal maze layout with start (S) and finish (F).

![Example image](/images/maze_diagonal_training.png "Diagonal maze training")  
Fig 2: Diagonal maze training. Convergence to optimal number of steps (8). 

## 13x13 maze
![Example image](/images/maze_13x13.png "13x13 maze layout")  
Fig 3: Diagonal maze layout with start (S) and finish (F).

![Example image](/images/maze_13x13_training.png "13x13 maze training")  
Fig 4: 13x13 maze training. Convergence to optimal number of steps (20). 