import numpy as np
import math
import matplotlib.pyplot as plt

# ========== MAP CONFIGURATION ==========
original_latitude = 10.9288775106096
original_longtitude = 106.796958446503
meters_per_grid = 1
METERS_PER_DEGREE_LATITUDE = 111_139
METERS_PER_DEGREE_LONGITUDE = 111_320 * math.cos(math.radians(original_latitude))

# ========== START & GOAL ==========
default_start = (5, 5)
default_goal = (48, 5)

# ========== SHAPE HELPERS ==========
def plot_circle(grid: np.ndarray, center: tuple[int, int], radius: int):
    x_c, y_c = center
    for x in range(max(0, x_c - radius), min(grid.shape[0], x_c + radius + 1)):
        for y in range(max(0, y_c - radius), min(grid.shape[1], y_c + radius + 1)):
            if (x - x_c)**2 + (y - y_c)**2 <= radius**2:
                grid[x, y] = 1

def plot_diamond(grid: np.ndarray, center: tuple[int, int], size: int):
    x_c, y_c = center
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if abs(x - x_c) + abs(y - y_c) <= size:
                grid[x, y] = 1

def plot_rhombus(grid: np.ndarray, center: tuple[int, int], height: int, width: int):
    x_c, y_c = center
    for dx in range(-height, height + 1):
        dy_limit = int((width / height) * (height - abs(dx)))
        for dy in range(-dy_limit, dy_limit + 1):
            x = x_c + dx
            y = y_c + dy
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = 1

# ========== GRID MAP GENERATOR ==========
def grid_map(map_id=1, size=50):
    grid = np.zeros((size, size))

    if map_id == 1:
        # Map 1: Basic rectangles
        grid[10:20, 30:50] = 1  
        grid[30:40, 15:35] = 1
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1
        grid[20:30, 0:6] = 1

    elif map_id == 2:
        # Map 2: Add some circular obstacles
        grid[8:10, 5:20] = 1
        grid[22:30, 22:30] = 1
        grid[10:20, 40:48] = 1  
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1

        # Circles
        plot_circle(grid, center=(10, 30), radius=4)
        plot_circle(grid, center=(45, 20), radius=4)
        plot_circle(grid, center=(25, 15), radius=4)
        plot_circle(grid, center=(40, 5), radius=4)

    elif map_id == 3:
        # Map 3: Combo of shapes
        grid[10:20, 30:48] = 1  
        grid[10:20, 0:20] = 1
        grid[30:40, 40:50] = 1
        grid[20:30, 0:6] = 1
        grid[38:45, 1:12] = 1

        # Circles
        plot_circle(grid, center=(10, 30), radius=4)
        plot_circle(grid, center=(45, 20), radius=4)
        plot_circle(grid, center=(25, 15), radius=4)
        plot_circle(grid, center=(30, 30), radius=6)

    elif map_id == 4:
        # Map 4: New complex shapes
        plot_circle(grid, center=(15, 10), radius=4)       # Circle
        plot_circle(grid, center=(25, 40), radius=6)       # Larger circle
        plot_circle(grid, center=(40, 10), radius=2)       # Smaller circle
        plot_diamond(grid, center=(30, 30), size=6)        # Diamond
        plot_rhombus(grid, center=(10, 40), height=5, width=14)  # Rhombus
        grid[35:40, 35:45] = 1                             # Rectangle

    return grid

# ========== VISUALIZATION ==========
def create_grid_map(grid: np.ndarray, path: list[tuple[int, int]] = None):
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path")
    plt.imshow(grid, cmap='gray_r', origin='upper', extent=[0, 50, 50, 0])

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', linewidth=2, label='Path')

    plt.plot(default_start[1], default_start[0], 'go', markersize=10, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'ro', markersize=10, label='Goal')

    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0, 55, 5))
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    for i in range(1, 5):
        print(f"Displaying Map {i}")
        grid = grid_map(map_id=i)
        create_grid_map(grid)
