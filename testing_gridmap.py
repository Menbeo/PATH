import numpy as np
import math
import matplotlib.pyplot as plt

# Map configuration
area = 288_409.51
original_latitude = 10.9300468100767
original_longtitude = 106.796904802322
meters_per_grid = 1
METERS_PER_DEGREE_LATITUDE = 111_139
METERS_PER_DEGREE_LONGITUDE = 111_320 * math.cos(math.radians(original_latitude))
size = int(math.sqrt(area))

def convert_grid_to_lat_lon(x_grid: int, y_grid: int) -> tuple[float, float]:
    delta_x_meters = x_grid * meters_per_grid
    delta_y_meters = y_grid * meters_per_grid
    latitude = original_latitude - (delta_y_meters / METERS_PER_DEGREE_LATITUDE)
    longitude = original_longtitude + (delta_x_meters / METERS_PER_DEGREE_LONGITUDE)
    return latitude, longitude

# Original start and goal positions (not flipped)
default_start = (500, 500)  # Bottom-right
default_goal = (48, 48)     # Top-left

def grid_map(size=size):
    grid = np.zeros((size, size))
    # Original obstacle positions (not flipped)
    grid[50:80, 50:200] = 1
    grid[220:300, 220:400] = 1
    grid[100:200, 200:537] = 1  
    grid[100:200, 100:200] = 1
    grid[300:400, 400:500] = 1
    grid[200:300, 0:60] = 1
    
    # Removed the flip operations to keep original orientation
    return grid

def create_grid_map(grid: np.ndarray, path: list[tuple[int, int]] = None):
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path (Original Orientation)")
    plt.imshow(grid, cmap='gray_r', origin='upper')
    
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', linewidth=2, label='Path') 
    
    # Original start (green) and goal (blue) positions
    plt.plot(default_start[1], default_start[0], 'go', markersize=10, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'bo', markersize=10, label='Goal')
    
    plt.grid(True)
    plt.legend()
    plt.show()

# Generate and display the map
grid = grid_map()
create_grid_map(grid)