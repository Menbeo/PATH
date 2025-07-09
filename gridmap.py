import numpy as np
import math
import matplotlib.pyplot as plt
import heapq

# ========== CONFIGURATION ==========
original_latitude = 10.9288775106096
original_longitude = 106.796958446503
meters_per_grid = 1
METERS_PER_DEGREE_LATITUDE = 111_139
METERS_PER_DEGREE_LONGITUDE = 111_320 * math.cos(math.radians(original_latitude))

# Start and goal
default_start = (5, 5)
default_goal = (48, 5)

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

# ========== MAP GENERATOR ==========
def grid_map(map_id=1, size=50):
    grid = np.zeros((size, size))

    if map_id == 1:
        grid[10:20, 30:50] = 1  
        grid[30:40, 15:35] = 1
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1
        grid[20:30, 0:6] = 1

    elif map_id == 2:
        grid[8:10, 5:20] = 1
        grid[22:30, 22:30] = 1
        grid[10:20, 40:48] = 1  
        grid[10:20, 10:20] = 1
        grid[30:40, 40:50] = 1
        plot_circle(grid, center=(10, 30), radius=4)
        plot_circle(grid, center=(45, 20), radius=4)
        plot_circle(grid, center=(25, 15), radius=4)
        plot_circle(grid, center=(40, 5), radius=4)

    elif map_id == 3:
        grid[10:20, 30:48] = 1  
        grid[10:20, 0:20] = 1
        grid[30:40, 40:50] = 1
        grid[20:30, 0:6] = 1
        grid[38:45, 1:12] = 1
        plot_circle(grid, center=(10, 30), radius=4)
        plot_circle(grid, center=(45, 20), radius=4)
        plot_circle(grid, center=(25, 15), radius=4)
        plot_circle(grid, center=(30, 30), radius=6)

    elif map_id == 4:
        plot_circle(grid, center=(15, 5), radius=3)   # In path
        plot_circle(grid, center=(25, 5), radius=4)   # In path
        plot_circle(grid, center=(35, 5), radius=2)   # In path
        grid[20:30, 3:8] = 1                          # Rectangle in path
        plot_diamond(grid, center=(30, 30), size=6)
        plot_rhombus(grid, center=(10, 40), height=5, width=14)

    return grid

# ========== A* PATHFINDING ==========
def heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def astar(grid, start, goal):
    neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_heap = [(fscore[start], start)]
    open_set = {start}

    while open_heap:
        current = heapq.heappop(open_heap)[1]
        open_set.discard(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            tentative_gscore = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set and tentative_gscore >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_gscore < gscore.get(neighbor, float('inf')) or neighbor not in open_set:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))
                    open_set.add(neighbor)

    return None

# ========== VISUALIZATION ==========
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
        path = astar(grid, default_start, default_goal)
        create_grid_map(grid, path)
