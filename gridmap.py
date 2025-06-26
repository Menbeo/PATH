import numpy as np
import matplotlib.pyplot as plt

def grid_map(grid: np.ndarray, path: list[tuple[int, int]] = None):
    plt.imshow(grid, cmap='binary')

    if grid is None:
        grid = np.zeros((50,50)) #default map

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

  
