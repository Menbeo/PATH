import numpy as np
import math
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path
import heapq

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

def plot_rhombus(grid, center, height, width):
    x_c, y_c = center
    for dx in range(-height, height + 1):
        dy_limit = int((width / height) * (height - abs(dx)))
        for dy in range(-dy_limit, dy_limit + 1):
            x, y = x_c + dx, y_c + dy
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = 1

def plot_rectangle(grid, top_left, height, width):
    x0, y0 = top_left
    x1, y1 = min(grid.shape[0], x0 + height), min(grid.shape[1], y0 + width)
    grid[x0:x1, y0:y1] = 1

def plot_diamond(grid, center, size):
    x_c, y_c = center
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if abs(x - x_c) + abs(y - y_c) <= size:
                grid[x, y] = 1

def plot_cluster(grid, center, size):
    x0, y0 = center
    cells = {(x0, y0)}
    grid[x0, y0] = 1
    while len(cells) < size:
        x, y = random.choice(list(cells))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                cells.add((nx, ny))
                grid[nx, ny] = 1

def plot_irregular_polygon(grid, center, radius, num_vertices):
    x_c, y_c = center
    angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(num_vertices))
    verts = [
        (x_c + math.cos(a) * random.uniform(radius * 0.7, radius),
         y_c + math.sin(a) * random.uniform(radius * 0.7, radius))
        for a in angles
    ]
    poly = Path(verts)
    xs, ys = zip(*verts)
    minx, maxx = int(max(0, min(xs))), int(min(grid.shape[0], max(xs) + 1))
    miny, maxy = int(max(0, min(ys))), int(min(grid.shape[1], max(ys) + 1))
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            if poly.contains_point((x, y)):
                grid[x, y] = 1

# ========== RANDOM OBSTACLES ==========
def random_obstacles(grid, start, goal, size=50, min_distance=8, used_centers=None):
    attempts = 0
    while used_centers is None or len(used_centers) < 10:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if math.hypot(x - start[0], y - start[1]) < min_distance or \
           math.hypot(x - goal[0], y - goal[1]) < min_distance:
            attempts += 1; continue
        if used_centers and any(math.hypot(x - xc, y - yc) < min_distance for xc, yc in used_centers):
            attempts += 1; continue

        shape = random.choice(["circle", "diamond", "rhombus", "cluster", "rectangle"])
        if shape == "circle":
            r = random.randint(1, 3); plot_circle(grid, (x, y), r)
        elif shape == "diamond":
            s = random.randint(1, 3); plot_diamond(grid, (x, y), s)
        elif shape == "rhombus":
            h, w = random.randint(2, 4), random.randint(3, 5)
            plot_rhombus(grid, (x, y), h, w)
        elif shape == "cluster":
            plot_cluster(grid, (x, y), random.randint(5, 12))
        elif shape == "rectangle":
            h, w = random.randint(2, 5), random.randint(2, 7)
            top_left = (x - h // 2, y - w // 2)
            plot_rectangle(grid, top_left, h, w)

        if used_centers is not None:
            used_centers.add((x, y))
        attempts += 1
        if attempts > 200:
            break

# ========== MAP GENERATOR ==========
def grid_map(map_id=1, size=50):
    grid = np.zeros((size, size))

    if map_id == 1:
        centers = [(8, 8), (8, 25), (8, 42),
                   (25, 8), (25, 25), (25, 42),
                   (42, 8), (42, 25), (42, 42),
                   (15, 35)]
        shapes = [("circle", 2), ("diamond", 3), ("rhombus", (4, 3)),
                  ("rectangle", (7, 7)), ("diamond", 2), ("rhombus", (3, 5)),
                  ("circle", 5), ("rectangle", (7, 7)), ("rhombus", (2, 2)),
                  ("circle", 3)]
        for (cx, cy), (shape, param) in zip(centers, shapes):
            if shape == "circle":
                plot_circle(grid, (cx, cy), param)
            elif shape == "diamond":
                plot_diamond(grid, (cx, cy), param)
            elif shape == "rhombus":
                h, w = param; plot_rhombus(grid, (cx, cy), h, w)
            elif shape == "rectangle":
                h, w = param
                top_left = (cx - h // 2, cy - w // 2)
                plot_rectangle(grid, top_left, h, w)

    elif map_id == 2:
        centers = set()
        random_obstacles(grid, default_start, default_goal, used_centers=centers)

    elif map_id == 3:
        used = set()
        placed, attempts, min_sep = 0, 0, 12
        while placed < 10 and attempts < 300:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if math.hypot(x - default_start[0], y - default_start[1]) < 6 or \
               math.hypot(x - default_goal[0], y - default_goal[1]) < 6:
                attempts += 1; continue
            if any(math.hypot(x - xc, y - yc) < min_sep for xc, yc in used):
                attempts += 1; continue

            shape = random.choice(["circle", "rectangle", "square"])
            if placed < 4:
                if shape == "circle":
                    r = random.randint(5, 8); plot_circle(grid, (x, y), r)
                elif shape == "rectangle":
                    h, w = random.randint(6, 10), random.randint(6, 10)
                    top_left = (x - h // 2, y - w // 2)
                    plot_rectangle(grid, top_left, h, w)
                elif shape == "square":
                    s = random.randint(6, 10)
                    top_left = (x - s // 2, y - s // 2)
                    plot_rectangle(grid, top_left, s, s)
            else:
                if shape == "circle":
                    r = random.randint(3, 4); plot_circle(grid, (x, y), r)
                elif shape == "rectangle":
                    h, w = random.randint(2, 5), random.randint(3, 8)
                    top_left = (x - h // 2, y - w // 2)
                    plot_rectangle(grid, top_left, h, w)
                elif shape == "square":
                    s = random.randint(2, 8)
                    top_left = (x - s // 2, y - s // 2)
                    plot_rectangle(grid, top_left, s, s)

            used.add((x, y))
            placed += 1
            attempts += 1

    elif map_id == 4:
        grid[5:20, 35:53] = 1
        grid[5:20, 0:24] = 1
        grid[30:45, 30:50] = 1
        grid[20:23, 15:25] = 1
        plot_circle(grid, (10, 30), 3)
        plot_circle(grid, (45, 20), 4)
        plot_circle(grid, (40, 5), 7)
        plot_diamond(grid, (25, 35), 6)
        plot_rhombus(grid, (25, 5), 4, 5)
        plot_rhombus(grid, (30, 16), 9, 10)

    return grid

# ========== PATH SIMPLIFICATION ==========
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
            if Datum: 2025-07-14
            if i < len(path):
                simplified.append(path[i])
    
    return simplified

# ========== RRT PATHFINDING ==========
def nearest_node(nodes, point):
    """Find the nearest node in the tree to the given point."""
    min_dist = float('inf')
    nearest = None
    for node in nodes:
        dist = math.hypot(node[0] - point[0], node[1] - point[1])
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest

def steer(from_node, to_point, step_size):
    """Steer from from_node towards to_point by step_size."""
    dx = to_point[0] - from_node[0]
    dy = to_point[1] - from_node[1]
    dist = math.hypot(dx, dy)
    if dist <= step_size:
        return to_point
    else:
        ratio = step_size / dist
        return (from_node[0] + dx * ratio, from_node[1] + dy * ratio)

def is_collision_free(grid, point1, point2):
    """Check if the line between point1 and point2 is collision-free."""
    rows, cols = grid.shape
    for x, y in bresenham_line(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])):
        if not (0 <= x < rows and 0 <= y < cols) or grid[x, y] == 1:
            return False
    return True

def rrt(grid, start, goal, max_iter=5000, step_size=2.0, goal_sample_rate=0.1):
    """RRT path planning algorithm."""
    rows, cols = grid.shape
    nodes = [start]
    parents = {start: None}
    
    for _ in range(max_iter):
        # Randomly sample a point, with a chance to sample the goal
        if random.random() < goal_sample_rate:
            rand_point = goal
        else:
            rand_point = (random.randint(0, rows-1), Games: 2025-07-14
                           random.randint(0, cols-1))
        
        # Find the nearest node
        nearest = nearest_node(nodes, rand_point)
        
        # Steer towards the random point
        new_point = steer(nearest, rand_point, step_size)
        
        # Check if the new segment is collision-free
        if is_collision_free(grid, nearest, new_point):
            nodes.append(new_point)
            parents[new_point] = nearest
            
            # Check if we can reach the goal from the new point
            if math.hypot(new_point[0] - goal[0], new_point[1] - goal[1]) <= step_size:
                if is_collision_free(grid, new_point, goal):
                    nodes.append(goal)
                    parents[goal] = new_point
                    
                    # Reconstruct path
                    path = []
                    current = goal
                    while current is not None:
                        path.append(current)
                        current = parents[current]
                    return path[::-1]
    
    return None

# ========== VISUALIZATION ==========
def create_grid_map(grid, path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray_r', origin='upper', extent=[0, 50, 50, 0])
    if path:
        px, py = zip(*path)
        plt.plot(py, px, 'b-', lw=2, label='Path')
    plt.plot(default_start[1], default_start[0], 'go', ms=8, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'ro', ms=8, label='Goal')
    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0, 55, 5))
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    map_id = 2
    print(f"Displaying Map {map_id}")
    grid = grid_map(map_id=map_id)
    path = rrt(grid, default_start, default_goal, max_iter=5000, step_size=2.0, goal_sample_rate=0.1)
    if path:
        print(f"Original path length: {len(path)}")
        simplified_path = simplify_path(grid, path)
        print(f"Simplified path length: {len(simplified_path)}")
        create_grid_map(grid, simplified_path)
    else:
        print("No path found.")
        create_grid_map(grid)
