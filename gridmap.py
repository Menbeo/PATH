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

  
if __name__ == "__main__":
    from queue import PriorityQueue

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(grid, start, goal):
        rows, cols = grid.shape
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    if grid[neighbor] == 1:
                        continue  # obstacle
                    new_cost = cost_so_far[current] + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(goal, neighbor)
                        open_set.put((priority, neighbor))
                        came_from[neighbor] = current

        # Reconstruct path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return []  # No path
        path.append(start)
        path.reverse()
        return path

    # Create map and run test
    grid = grid_map()
    start = (0, 0)
    goal = (490, 490)
    path = a_star(grid, start, goal)

    # Visualize
    create_grid_map(grid, path)
