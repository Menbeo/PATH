# a_star_test.py

import heapq
import numpy as np
from gridmap import grid_map

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = [(0 + heuristic(start, goal), 0, start, [])]
    visited = set()

    while open_set:
        est_total_cost, cost_so_far, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        path = path + [current]

        if current == goal:
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0 and neighbor not in visited:
                    new_cost = cost_so_far + 1
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path))

    return None  # No path found

# ---- MAIN ----
if __name__ == "__main__":
    # Generate grid and define start/goal
    grid = np.zeros((50, 50))

    start = (10,10)
    goal = (46, 47)

    path = a_star(grid, start, goal)

    if path:
        print(f"Path found! Length: {len(path)}")
        grid_map(grid, path)
    else:
        print("No path found.")
        grid_map(grid)
