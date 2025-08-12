import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib.path as Path 
from gridmap import create_grid_map, grid_map, default_goal,default_start
from gridmap import convert_grid_to_lat_lon,compute_neighborhood_layers
from convert_to_waypoints import export_waypoints
from new_apply import bspline_smooth
# ========== PATH SIMPLIFICATION ==========
def bresenham_line(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
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
    if len(path) < 2:
        return path
    simplified = [path[0]]
    rows, cols = grid.shape
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            x0, y0 = path[i]
            x1, y1 = path[j]
            if all(0 <= x < rows and 0 <= y < cols and grid[x, y] == 0 
                   for x, y in bresenham_line(int(x0), int(y0), int(x1), int(y1))):
                simplified.append(path[j])
                i = j
                break
        else:
            i += 1
            if i < len(path):
                simplified.append(path[i])
    return simplified
# ========== RRT PATHFINDING ==========
def nearest_node(nodes, point):
    return min(nodes, key=lambda node: math.hypot(node[0] - point[0], node[1] - point[1]))

def steer(from_node, to_point, step_size):
    dx, dy = to_point[0] - from_node[0], to_point[1] - from_node[1]
    dist = math.hypot(dx, dy)
    if dist <= step_size:
        return to_point
    return (from_node[0] + dx * step_size / dist, from_node[1] + dy * step_size / dist)

def is_collision_free(inflation, point1, point2):
    for x, y in bresenham_line(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])):
        # if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]) or grid[x, y] == 1:
        if inflation[x, y] == 9 or inflation[x, y] == 1:
                return False
    return True

def rrt(grid, inflation, start, goal, max_iter=3000, step_size=2.0, goal_sample_rate=0.1):
    nodes = [start]
    node_expand = 0
    parents = {start: None}
    for _ in range(max_iter):
        node_expand += 1
        rand_point = goal if random.random() < goal_sample_rate else (
            random.uniform(0, 50), random.uniform(0, 50))
        nearest = nearest_node(nodes, rand_point)
        new_point = steer(nearest, rand_point, step_size)
        if is_collision_free(inflation, nearest, new_point):
            nodes.append(new_point)
            parents[new_point] = nearest
            if math.hypot(new_point[0] - goal[0], new_point[1] - goal[1]) <= step_size:
                if is_collision_free(grid, new_point, goal):
                    nodes.append(goal)
                    parents[goal] = new_point
                    path = []
                    current = goal
                    while current is not None:
                        path.append(current)
                        current = parents[current]
                    return path[::-1]
    return None



# ========== MAIN ==========
if __name__ == "__main__":
        grid = grid_map(map_id=1)
        inflation = compute_neighborhood_layers(grid)
        path = rrt(grid, inflation, default_start, default_goal)
        smooth = bspline_smooth(path,grid, inflation)
        if path:
            print(f"Original path length: {len(smooth)}")
            # simplify_pathhe = simplify_path(grid,path)
            create_grid_map(grid, smooth)
            lat_lon_path = [convert_grid_to_lat_lon(x,y) for (x,y) in path]
            filename = f"RRT{1}.waypoints"
            export_waypoints(lat_lon_path, filename=filename)
        else:
            print("No path found.")
            create_grid_map(grid)
