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

# ======= START AND GOAL =====
default_start = (2, 2)
default_goal = (47, 47)

# ========== SIMPLE SHAPE ==========
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

def plot_rectangle(grid, top_left, height, width):
    x0, y0 = top_left
    x1, y1 = min(grid.shape[0], x0 + height), min(grid.shape[1], y0 + width)
    grid[x0:x1, y0:y1] = 1

# ====== COMPLEX SHAPE =====
def plot_irregular_polygon(grid, center, radius, num_vertices):
    x_c, y_c = center
    angles = sorted(random.uniform(0, 2*math.pi) for _ in range(num_vertices))
    verts = [
        (x_c + math.cos(a)*random.uniform(radius*0.7, radius),
         y_c + math.sin(a)*random.uniform(radius*0.7, radius))
        for a in angles
    ]
    poly = Path(verts)
    xs, ys = zip(*verts)
    minx, maxx = int(max(0, min(xs))), int(min(grid.shape[0], max(xs)+1))
    miny, maxy = int(max(0, min(ys))), int(min(grid.shape[1], max(ys)+1))
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            if poly.contains_point((x, y)):
                grid[x, y] = 1

def plot_cluster(grid, center, size):
    x0, y0 = center
    cells = {(x0, y0)}
    grid[x0, y0] = 1
    while len(cells) < size:
        x, y = random.choice(list(cells))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                cells.add((nx, ny))
                grid[nx, ny] = 1

#=== CONVERT TO LATITUDE AND LONGITUDE ====
def convert_grid_to_lat_lon(x_grid: int, y_grid: int) -> tuple[float, float]:
    delta_x_meters = x_grid * meters_per_grid
    delta_y_meters = y_grid * meters_per_grid
    latitude = original_latitude - (delta_y_meters / METERS_PER_DEGREE_LATITUDE)
    longitude = original_longitude + (delta_x_meters / METERS_PER_DEGREE_LONGITUDE)

    return latitude, longitude

#===== SCENARIOS 2: RANDOM OBSTACLES ======

def random_obstacles(grid, start, goal, size = 50, used_centers = None):
    #from start and goal - may overlap - 8 cells 
    for _ in range(300): #loop attempts - 300 times 
        x,y = random.randint(0, size - 1), random.randint(0, size-1)
    
        if (math.hypot(x - start[0], y - start[1]) < 8) or \
           (math.hypot(x - goal[0], y - goal[1]) < 8):
            continue

        #Check distance 
        if used_centers:
            too_close = any(math.hypot(x - xc, y - yc) < 8 for xc, yc in used_centers)
            if too_close:
                continue

        #Random select size 
        shape = random.choice(
            ["circle", "diamond", "rhombus", "rectangle", "cluster"])
        size = random.randint(5, 8) #The bigger the size, the harder the map get

        #Shape: Simple shape & complex shape  + size random 
        if shape == "circle":
            plot_circle(grid, (x,y), size)

        elif shape == "diamond":
            plot_diamond(grid, (x,y), size)
        
        elif shape == "rhombus":
            h = random.randint(4, 7)
            w = random.randint(5, 8)
            plot_rhombus(grid, (x,y), h , w)

        elif shape == "rectangle":
            h,w = random.randint(3, 7), random.randint(4, 6)
            top_left = (x - h // 2, y - w // 2)
            plot_rectangle(grid, top_left, h, w)
    
        elif shape == "cluster":
            plot_cluster(grid, (x,y), random.randint(8,12))
        
        if used_centers is not None: 
            used_centers.add((x,y))
    return True


# ========== MAP GENERATOR ==========
def grid_map(map_id=1, size=50):
    grid = np.zeros((size, size))

    # ===== Scenario 1: Uniform arrangement ====
    if map_id == 1:
        centers = [(8, 8), (8, 25), (8, 42),
                   (25, 8), (25, 25), (25, 42),
                   (42, 8), (42, 25), (42, 42),
                   (15, 35)]
        shapes = [("circle", 2), ("diamond", 3), ("rhombus", (4, 3)),
                  ("rectangle", (7, 7)), ("diamond", 2), ("rhombus", (3, 3)),
                  ("circle", 5), ("rectangle", (6, 6)), ("rhombus", (2, 2)),
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

    # ======  Scenario 2 - Random Gridmap =====
    elif map_id == 2:
        obstacle = 11 #Since the second Scenario is random 
                      # the many the obstacles are, the more harder it get 
        placed = 0
        while placed <= obstacle:
            if random_obstacles(grid,default_start, default_goal):
                placed += 1

    #===== Scenario 3: Random placement with 4 larger obstacles and increased spacing ====
    elif map_id == 3:
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

    # ==== Scenarios 4 - Clustering Degree - grouped ==== 
    elif map_id == 4:
       #Simple shaped 
        grid[5:20, 0:10] = 1 #combine two different rectangle
        grid[5:20, 10:24] = 1
        grid[30:45, 30:50] = 1
        grid[5:20, 35:53] = 1
        #Complex shaped
        plot_circle(grid, center=(10,30), radius=2) #Different degree
        plot_circle(grid, center=(45,20), radius=6)
        plot_cluster(grid, (40,5), size = 100)
        plot_diamond(grid, (25,35), 6)
        plot_rhombus(grid, (25,5), 4, 5)
        plot_rhombus(grid,(30,16), 9, 9)

    return grid

# ===== COMPUTE BOUNDARIES FOR OBSTACLE ===
def compute_neighborhood_layers(grid, inflation_radius=1.8, meters_per_cell=1.0):
    rows, cols = grid.shape
    inflated_grid = np.zeros_like(grid)
    
    inflation_cells = int(np.ceil(inflation_radius / meters_per_cell))

    for x in range(rows):
        for y in range(cols):
            if grid[x, y] == 1:  # Obstacle cell
                for dx in range(-inflation_cells, inflation_cells + 1):
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            distance = np.sqrt(dx**2 + dy**2)
                            if distance <= inflation_radius:
                                inflated_grid[nx, ny] = 1  # Mark as inflated (dangerous) zone
    return inflated_grid

# ===== CREATE GRID MAP ==== 
def create_grid_map(grid: np.ndarray, path=None):
    neighborhood_layers = compute_neighborhood_layers(grid)
    # RGB map initialization
    color_grid = np.ones((*grid.shape, 3))  # default free space: white (1,1,1)

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                color_grid[x, y] = [1.0, 1.0, 0.0]  # obstacle core: Yellow
            elif neighborhood_layers[x, y] >= 1:
                color_grid[x, y] = [1.0, 0.0, 0.0]  # r2: red (dangerous inflation)
            else:
                color_grid[x, y] = [1.0, 1.0, 1.0]
    
    plt.figure(figsize=(10, 10))
    plt.title("Grid Map with Path")
    plt.imshow(color_grid, cmap='gray', origin='upper', extent=[0, 50, 50, 0])
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', linewidth=2, label='Path')
    plt.plot(default_start[1], default_start[0], 'go', markersize=10, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'ro', markersize=10, label='Goal')
    
    from matplotlib.patches import Patch

    # Custom legend for zones
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='Obstacle'),
        Patch(facecolor='red', edgecolor='black', label='Dangerous Zone'),
        Patch(facecolor='green', label='Start'),
        Patch(facecolor='red', label='Goal')
    ]

    # Draw custom legend box
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, borderpad=1)

    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0, 55, 5))
    plt.xlim(0, 50)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    plt.show()


if __name__ == "__main__":
    for i in range(1,5):
        print(f"Displaying Map {i}")
        grid = grid_map(map_id=i)
        create_grid_map(grid)

