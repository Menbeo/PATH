import numpy as np
import math
import matplotlib.pyplot as plt

#define maximum longitude and latitude
original_latitude = 10.9287722 
original_longtitude = 106.7978704
meters_per_grid = 1
METERS_PER_DEGREE_LATITUDE = 111_139
METERS_PER_DEGREE_LONGITUDE = 111_320 * math.cos(math.radians(original_latitude))


#Convert to circle 
def plot_circle(grid: np.ndarray, center: tuple[int,int], radius: int):
    x_c,y_c = center
    for x in range(max(0,x_c - radius), min(grid.shape[0], x_c + radius + 1)):
        for y in range(max(0,y_c - radius), min(grid.shape[0],y_c + radius + 1)):
            if(x - x_c)**2 + (y - y_c)**2 <= radius**2:
                grid[x,y] = 1

#Convert to latitude and longitude
def convert_grid_to_lat_lon(x_grid: int, y_grid: int) -> tuple[float, float]:
    delta_x_meters = x_grid * meters_per_grid
    delta_y_meters = y_grid * meters_per_grid
    latitude = original_latitude - (delta_y_meters / METERS_PER_DEGREE_LATITUDE)
    longitude = original_longtitude + (delta_x_meters / METERS_PER_DEGREE_LONGITUDE)

    return latitude, longitude
#default goal and start 
default_start = (5,5)
default_goal = (48,5)

#Three map with different obstacles
def grid_map (map_id = 1, size=50):
    grid = np.zeros((size,size))
    
    if map_id == 1:
        #Grid map1
        grid[10:20, 30:53] = 1  
        grid[30:40, 15:35] = 1
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1
        grid[20:30, 0:6] = 1
    elif map_id == 2:
        # Grid map 2
        grid[8:10, 5:20] = 1
        grid[22:40, 22:30] = 1
        grid[10:20, 40:53] = 1  
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1
    
        plot_circle(grid, center=(10,30), radius=4)
        plot_circle(grid, center=(45,20), radius=4)
        plot_circle(grid, center=(25,15), radius=4)
        plot_circle(grid, center=(40,5), radius=4)
    
    elif map_id == 3:

        #Grid map3
        grid[10:20, 30:53] = 1  
        grid[10:20, 0:20] = 1
        grid[30:40, 40:50] = 1
        grid[20:30, 0:6] = 1
        grid[38:45, 1:12] = 1

        plot_circle(grid, center=(10,30), radius=4)
        plot_circle(grid, center=(45,20), radius=4)
        plot_circle(grid, center=(25,15), radius=4)
        plot_circle(grid, center=(30,30), radius=6)
    return grid

def create_grid_map(grid: np.ndarray, path: list[tuple[int, int]] = None):
    plt.imshow(grid, cmap='binary')
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path")
    plt.imshow(grid, cmap='gray_r', origin='upper',extent=[0,50,50,0])
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', linewidth=2, label='Path') 
    
    #draw default goal and start
    plt.plot(default_start[1], default_start[0], 'go', markersize=10, label = 'Start')
    plt.plot(default_goal[1],default_goal[0],'bo',markersize=10,label='Goal')
    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0, 55, 5))
    plt.xlim(0,50)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


# # # Like this
# for i in range(1,4):
#     grid = grid_map(map_id=i)
#     print(f"display grid map {i}")
#     create_grid_map(grid)


  
