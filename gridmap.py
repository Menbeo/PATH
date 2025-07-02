import numpy as np
import math
import matplotlib.pyplot as plt
#define map size 
area = 288_409.51
original_latitude = 10.9303312
original_longtitude = 106.7925060
meters_per_grid = 1
METERS_PER_DEGREE_LATITUDE = 111_139
METERS_PER_DEGREE_LONGITUDE = 111_320 * math.cos(math.radians(original_latitude))
size = int(math.sqrt(area))

def convert_grid_to_lat_lon(x_grid: int, y_grid: int) -> tuple[float, float]:
    # Calculate displacement in meters
    delta_x_meters = x_grid * meters_per_grid
    delta_y_meters = y_grid * meters_per_grid

    # Convert displacement in meters to degrees
    # Latitude: Y increases downwards, so we subtract from origin_latitude
    latitude = original_latitude - (delta_y_meters / METERS_PER_DEGREE_LATITUDE)
    # Longitude: X increases rightwards, so we add to origin_longitude
    longitude = original_longtitude + (delta_x_meters / METERS_PER_DEGREE_LONGITUDE)

    return latitude, longitude
#default goal and start 

default_start = (30,500)
default_goal = (500,200)
def grid_map( size=size):
    grid = np.zeros((size,size))
    # grid[50:80, 50:200] = 1
    # grid[220:400, 220:300] = 1
    grid[100:200, 400:537] = 1  
    grid[100:200, 100:200] = 1
    grid[300:400, 400:500] = 1
    grid[200:300, 0:60] = 1

    # grid = np.flipud(np.fliplr(grid))

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


# Like this
grid = grid_map()
create_grid_map(grid)


  
