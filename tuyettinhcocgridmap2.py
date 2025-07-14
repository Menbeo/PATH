import numpy as np
import math
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path

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

# ========== RANDOM OBSTACLES FOR SCENARIO 2 ==========
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
        # Scenario 1: Uniform arrangement with larger rectangles
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
        # Scenario 2: Random obstacles
        centers = set()
        random_obstacles(grid, default_start, default_goal, used_centers=centers)

    elif map_id == 3:
        # Scenario 3: Random placement with 4 larger obstacles and increased spacing
        used = set()
        placed = 0
        attempts = 0
        min_sep = 12  # minimum separation between obstacle centers
        while placed < 10 and attempts < 300:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            # avoid start/goal
            if math.hypot(x - default_start[0], y - default_start[1]) < 6 or \
               math.hypot(x - default_goal[0], y - default_goal[1]) < 6:
                attempts += 1; continue
            # enforce separation
            if any(math.hypot(x - xc, y - yc) < min_sep for xc, yc in used):
                attempts += 1; continue

            # larger first four
            if placed < 4:
                shape = random.choice(["circle", "rectangle", "square"])
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
                shape = random.choice(["circle", "rectangle", "square"])
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
        # Scenario 4: Grouped clusters
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

# ========== VISUALIZATION ==========
def create_grid_map(grid, path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray_r', origin='upper', extent=[0, 50, 50, 0])
    if path:
        px, py = zip(*path)
        plt.plot(py, px, 'b-', lw=2, label='Path')
    plt.plot(default_start[1], default_start[0], 'go', ms=8, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'ro', ms=8, label='Goal')
    plt.xticks(np.arange(0, 55, 5)); plt.yticks(np.arange(0, 55, 5))
    plt.grid(True); plt.gca().set_aspect('equal'); plt.legend(); plt.show()

# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    map_id = 3
    print(f"Displaying Map {map_id}")
    grid = grid_map(map_id=map_id)
    create_grid_map(grid)
