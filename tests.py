from rlmaze.utils import Utils
from rlmaze.rlmaze import RLMaze

maze_grid, maze_start, maze_finish, maze_name = Utils().sample_maze()
maze_img = Utils().visualize_maze(maze_grid, maze_start, maze_finish, maze_name)

maze_grid, maze_start, maze_finish, maze_name = Utils().sample_diagonal_maze()
maze_img = Utils().visualize_maze(maze_grid, maze_start, maze_finish, maze_name)

RLMaze().learn(maze_grid, maze_start, maze_finish)