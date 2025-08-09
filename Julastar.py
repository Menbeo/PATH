import numpy as np
import math
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path
import heapq
from convert_to_waypoints import export_waypoints
from gridmap import convert_grid_to_lat_lon
from gridmap import create_grid_map, grid_map, default_goal,default_start
from new_apply import turn_constraint, bspline_smooth

def bresenham_line(x0, y0, x1, y1):
    """Generate points along a straight line from (x0, y0) to (x1, y1) using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def simplify_path(grid, path):
    """Simplify the path by removing redundant waypoints using line-of-sight checks."""
    if not path or len(path) < 2:
        return path
    
    simplified = [path[0]]
    rows, cols = grid.shape
    
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            x0, y0 = path[i]
            x1, y1 = path[j]
            # Check if the line between points i and j is clear
            clear = True
            for x, y in bresenham_line(x0, y0, x1, y1):
                if not (0 <= x < rows and 0 <= y < cols) or grid[x, y] == 1:
                    clear = False
                    break
            if clear:
                simplified.append(path[j])
                i = j
                break
        else:
            i += 1
            if i < len(path):
                simplified.append(path[i])
    
    return simplified

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
        if path:
            # print(f"Original path length: {len(path)}")
            simplified_path = simplify_path(grid, path)
            simplified_path_corner = turn_constraint(simplified_path)
            round_path = bspline_smooth(simplified_path_corner)
            # print(f"Simplified path length: {len(simplified_path)}")
       
            round_path_int = [(int(x), int(y)) for x, y in round_path]
            create_grid_map(grid, round_path_int)
            # lat_lon_path = [convert_grid_to_lat_lon(x,y) for (x,y) in simplified_path_corner]
            # filename = f"A_star{map_id}.waypoints"
            # export_waypoints(lat_lon_path, filename=filename)
        else:
            print("No path found.")
            create_grid_map(grid)