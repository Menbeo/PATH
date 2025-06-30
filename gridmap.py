import numpy as np
import math
import matplotlib.pyplot as plt
#define map size 

area = 288_409.51
size = int(math.sqrt(area))

#default goal and start 

default_goal = (480,480)
default_start = (30,30)
def grid_map( size=size):
    grid = np.zeros((size,size))
    grid[50:80, 50:200] = 1
    grid[220:300, 220:400] = 1
    grid[100:200, 200:537] = 1  
    grid[100:200, 100:200] = 1
    grid[300:400, 400:500] = 1
    grid[200:300, 0:60] = 1


    return grid
def create_grid_map(grid: np.ndarray, path: list[tuple[int, int]] = None):
    plt.imshow(grid, cmap='binary')
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path")
    plt.imshow(grid, cmap='gray_r', origin='upper')
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', linewidth=2, label='Path') 
    
    #draw default goal and start
    plt.plot(default_start[1], default_start[0], 'go', markersize=10, label = 'Start')
    plt.plot(default_goal[1],default_goal[0],'bo',markersize=10,label='Goal')
    plt.grid(True)
    plt.legend()
    plt.show()
