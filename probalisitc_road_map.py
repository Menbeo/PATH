from gridmap import create_grid_map,grid_map,default_goal, default_start
import numpy as np 
import random 

#Check the node is free or not 
def is_free(x,y,grid):
    x = int(x)
    y = int(y)
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x,y] == 0 
