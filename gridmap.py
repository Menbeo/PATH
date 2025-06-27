import numpy as np
import math
import matplotlib.pyplot as plt
#define map size 
area = 288_409.51
size = int(math.sqrt(area))

def grid_map( size=size):
    grid = np.zeros((size,size))
    grid[50:80, 50:200] = 1
    grid[220:300, 220:400] = 1
    grid[100:200, 200:537] = 1  
    grid[100:200, 100:200] = 1
    grid[300:400, 400:500] = 1
    grid[200:300, 0:60] = 1
    # grid[300:450, 300:500] = 1     
    # grid[220:300, 300:350] = 1  

    return grid
def create_grid_map(grid: np.ndarray, path: list[tuple[int, int]] = None):
    plt.imshow(grid, cmap='binary')
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path")
    plt.imshow(grid, cmap='gray_r', origin='upper')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='red', linewidth=2, label='Path') 
    plt.grid(True)
    if path:
        plt.legend()
    plt.show()

  
