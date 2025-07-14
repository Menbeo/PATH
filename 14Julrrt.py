import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ========== CONFIGURATION ==========
grid_size = 50
default_start = (2, 2)
default_goal = (47, 47)

# ========== OBSTACLE GENERATION ==========
def plot_circle(grid, center, radius):
    x_c, y_c = center
    for x in range(max(0, x_c - radius), min(grid.shape[0], x_c + radius + 1)):
        for y in range(max(0, y_c - radius), min(grid.shape[1], y_c + radius + 1)):
            if (x - x_c)**2 + (y - y_c)**2 <= radius**2:
                grid[x, y] = 1

def random_obstacles(grid, start, goal):
    centers = set()
    for _ in range(10):
        while True:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (math.hypot(x - start[0], y - start[1]) > 8 and 
                math.hypot(x - goal[0], y - goal[1]) > 8 and
                all(math.hypot(x - xc, y - yc) > 8 for xc, yc in centers)):
                plot_circle(grid, (x, y), random.randint(2, 4))
                centers.add((x, y))
                break

def grid_map():
    grid = np.zeros((grid_size, grid_size))
    random_obstacles(grid, default_start, default_goal)
    return grid

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

def is_collision_free(grid, point1, point2):
    for x, y in bresenham_line(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])):
        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]) or grid[x, y] == 1:
            return False
    return True

def rrt(grid, start, goal, max_iter=3000, step_size=2.0, goal_sample_rate=0.1):
    nodes = [start]
    parents = {start: None}
    for _ in range(max_iter):
        rand_point = goal if random.random() < goal_sample_rate else (
            random.uniform(0, grid_size), random.uniform(0, grid_size))
        nearest = nearest_node(nodes, rand_point)
        new_point = steer(nearest, rand_point, step_size)
        if is_collision_free(grid, nearest, new_point):
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

# ========== VISUALIZATION ==========
def create_grid_map(grid, path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='gray_r', origin='upper', extent=[0, grid_size, grid_size, 0])
    if path:
        px, py = zip(*path)
        plt.plot(py, px, 'b-', lw=2, label='Path')
    plt.plot(default_start[1], default_start[0], 'go', ms=8, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'ro', ms=8, label='Goal')
    plt.grid(True)
    plt.legend()
    plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    grid = grid_map()
    path = rrt(grid, default_start, default_goal)
    if path:
        print(f"Original path length: {len(path)}")
        simplified_path = simplify_path(grid, path)
        print(f"Simplified path length: {len(simplified_path)}")
        create_grid_map(grid, simplified_path)
    else:
        print("No path found.")
        create_grid_map(grid)
