import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from gridmap import 

# Define map size and resolution
area = 288_409.51
size = int(math.sqrt(area))

# Grid map with obstacles
def grid_map(size=size):
    grid = np.zeros((size, size))
    grid[50:80, 50:200] = 1
    grid[220:300, 220:400] = 1
    grid[100:200, 200:537] = 1
    grid[100:200, 100:200] = 1
    grid[300:400, 400:500] = 1
    grid[200:300, 0:60] = 1
    return grid

# Heuristic function: Euclidean distance
def heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

# A* algorithm with 8-directional movement (optimized)
def astar_8_directions(grid, start, goal):
    neighbors = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_heap = []
    open_set = {start}  # Efficient membership check

    heapq.heappush(open_heap, (fscore[start], start))

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

            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in open_set:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))
                    open_set.add(neighbor)

    return False

# Visualization function
def create_grid_map_with_clear_path(grid, path=None):
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path")
    plt.imshow(grid, cmap='gray_r', origin='upper')

    if path and len(path) > 1:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='red', linewidth=3, label='Path')
        plt.scatter(path_y[0], path_x[0], c='green', s=100, label='Start')
        plt.scatter(path_y[-1], path_x[-1], c='blue', s=100, label='Goal')
    else:
        print("⚠️ No path found!")

    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.show()


# Run the A* algorithm
grid = grid_map()
start = (30, 30)
goal = (480, 480)
path_8dir = astar_8_directions(grid, start, goal)
create_grid_map_with_clear_path(grid, path_8dir)
