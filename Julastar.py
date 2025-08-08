import numpy as np
import math
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path
import heapq
from convert_to_waypoints import export_waypoints
from gridmap import convert_grid_to_lat_lon
from gridmap import create_grid_map, grid_map, default_goal,default_start
from new_apply import smooth_path, angle_between 


# ========== A* PATHFINDING ==========
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
    
    came_from = {}
    g_score = {start: 0}
    visited = set()
    

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tentative_g = cost + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))

    return None



# ========== MAIN ==========
if __name__ == "__main__":
    for map_id in range(1,5):
        print(f"Displaying Map {map_id}")
        grid = grid_map(map_id=map_id)
        path = astar(grid, default_start, default_goal)
        if path is not None and len(path) > 0:
            print(f"Original path length: {len(path)}")
            smoothed_path = smooth_path(path)
            print(f"Simplified path length: {len(smoothed_path)}")

            create_grid_map(grid, smoothed_path)
            lat_lon_path = [convert_grid_to_lat_lon(x,y) for (x,y) in smoothed_path]
            filename = f"A_star{map_id}.waypoints"
            export_waypoints(lat_lon_path, filename=filename)
        else:
            print("No path found.")
            create_grid_map(grid)
