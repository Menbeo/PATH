import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ========== CONFIGURATION ==========
original_latitude = 10.9288327400429
original_longitude = 106.796797513962
meters_per_grid = 1
METERS_PER_DEGREE_LATITUDE = 111_139
METERS_PER_DEGREE_LONGITUDE = 111_320 * math.cos(math.radians(original_latitude))

# Start and goal
default_start = (2, 2)
default_goal = (47, 47)

# ========== SHAPE HELPERS ==========
def plot_circle(grid, center, radius):
    x_c, y_c = center
    for x in range(max(0, x_c - radius), min(grid.shape[0], x_c + radius + 1)):
        for y in range(max(0, y_c - radius), min(grid.shape[1], y_c + radius + 1)):
            if (x - x_c)**2 + (y - y_c)**2 <= radius**2:
                grid[x, y] = 1

def plot_diamond(grid, center, size):
    x_c, y_c = center
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if abs(x - x_c) + abs(y - y_c) <= size:
                grid[x, y] = 1

def plot_rhombus(grid, center, height, width):
    x_c, y_c = center
    for dx in range(-height, height + 1):
        dy_limit = int((width / height) * (height - abs(dx)))
        for dy in range(-dy_limit, dy_limit + 1):
            x = x_c + dx
            y = y_c + dy
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = 1

#=== Convert grid ====

def convert_grid_to_lat_lon(x_grid: int, y_grid: int) -> tuple[float, float]:
    delta_x_meters = x_grid * meters_per_grid
    delta_y_meters = y_grid * meters_per_grid
    latitude = original_latitude - (delta_y_meters / METERS_PER_DEGREE_LATITUDE)
    longitude = original_longitude + (delta_x_meters / METERS_PER_DEGREE_LONGITUDE)

    return latitude, longitude
#===== SCENARIOS 2: RANDOM OBSTACLES ======

def random_obstacles(grid,size = 50, obstacle = 11):

    for _ in range(obstacle):
        x,y = random.randint(0, size - 1), random.randint(0, size-1)

        #Random select size 
        shape = random.choice(
            ["circle", "diamond", "rhombus", "grid"])
        size = random.randint(1,6)

        #Shape: Simple shape & complex shape  + size random 
        if shape == "circle":
            plot_circle(grid, (x,y), size)

        elif shape == "diamond":
            plot_diamond(grid, (x,y), size)
        
        elif shape == "rhombus":
            h = random.randint(2, 6)
            w = random.randint(3, 6)
            plot_rhombus(grid, (x,y), h , w)

        elif shape == "grid":
            h = random.randint(3,6)
            w = random.randint(3,6)
            if x + h < size and y + w < size:
                grid[x:x+h, y:y+w] = 1
        
    return True

# ========== MAP GENERATOR ==========
def grid_map(map_id=1, size=50):
    grid = np.zeros((size, size))

    if map_id == 1:
        # Scenario 1: Uniform arrangement
        centers = [
            (8,8), (8,25), (8,42),
            (25,8), (25,25), (25,42),
            (42,8), (42,25), (42,42),
            (15,35)
        ]
        shapes = [
            ("circle",2), ("diamond",3), ("rhombus",(4,3)),
            ("circle",4), ("diamond",2), ("rhombus",(3,5)),
            ("circle",5), ("diamond",4), ("rhombus",(2,2)),
            ("circle",3)
        ]
        for (cx,cy),(shape,param) in zip(centers, shapes):
            if shape == "circle":
                plot_circle(grid, (cx,cy), param)
            elif shape == "diamond":
                plot_diamond(grid, (cx,cy), param)
            elif shape == "rhombus":
                h,w = param
                plot_rhombus(grid, (cx,cy), h, w)
    
    elif map_id == 2:
        #Scenario 2 - Random Gridmap
        obstacle = 10
        placed = 0 
        while placed <= obstacle:
            if random_obstacles(grid):
                placed += 1

    elif map_id == 3:
        grid[8:10, 5:20] = 1
        grid[22:40, 22:30] = 1
        grid[10:20, 40:53] = 1  
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1
    
        plot_circle(grid, center=(10,30), radius=4)
        plot_circle(grid, center=(45,20), radius=4)
        plot_circle(grid, center=(25,15), radius=4)
        plot_circle(grid, center=(40,5), radius=4)

    elif map_id == 4:
       #Scenarios 4 - Clustering Degree - grouped
       #Simple shaped 
        grid[5:20, 35:53] = 1  
        grid[5:20, 0:24] = 1
        grid[30:45, 30:50] = 1
        grid[20:23, 15:25] = 1
        #Complex shaped
        plot_circle(grid, center=(10,30), radius=3)
        plot_circle(grid, center=(45,20), radius=4)
        plot_circle(grid, center=(40,5), radius=7)
        plot_diamond(grid, (25,35), 6)
        plot_rhombus(grid, (25,5), 4, 5)
        plot_rhombus(grid,(30,16), 9, 10)

    return grid


def create_grid_map(grid: np.ndarray, path=None):

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
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print(f"Displaying Map {2}")
    grid = grid_map(map_id=2)
    create_grid_map(grid)
