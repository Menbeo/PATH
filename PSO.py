# pso_path_planning.py
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from gridmap import grid_map, create_grid_map, default_start, default_goal

# PSO Parameters
POP_SIZE = 50
MAX_ITER = 200
NUM_WAYPOINTS = 5
INERTIA = 0.5
C1 = 1.5
C2 = 1.5
PENALTY = 1e6

grid = grid_map()
rows, cols = grid.shape

def is_collision_free(grid, path, margin=2):
    """Check if path avoids obstacles with safety margin"""
    for i in range(1, len(path)):
        x0, y0 = int(round(path[i-1][0])), int(round(path[i-1][1]))
        x1, y1 = int(round(path[i][0])), int(round(path[i][1]))
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        while True:
            # Check current cell and surrounding margin
            for di in range(-margin, margin+1):
                for dj in range(-margin, margin+1):
                    xi, yj = x0 + di, y0 + dj
                    if (0 <= xi < grid.shape[0] and 0 <= yj < grid.shape[1] 
                        and grid[xi, yj] == 1):
                        return False
            
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
    return True

def fitness_function(particle, grid):
    waypoints = [default_start] + list(particle.reshape(-1, 2)) + [default_goal]
    
    # 1. Distance cost (shorter paths are better)
    path_length = sum(np.linalg.norm(np.array(waypoints[i])-np.array(waypoints[i-1])) 
                   for i in range(1, len(waypoints)))
    
    # 2. Collision penalty (HIGH cost for obstacles)
    if not is_collision_free(grid, waypoints):
        return float('inf')
    
    # 3. Proximity penalty (stay away from obstacles)
    proximity_cost = 0
    for wp in waypoints:
        x, y = int(round(wp[0])), int(round(wp[1]))
        for di in range(-3, 4):  # Check 3-cell radius
            for dj in range(-3, 4):
                xi, yj = x+di, y+dj
                if (0 <= xi < grid.shape[0] and 0 <= yj < grid.shape[1] 
                    and grid[xi, yj] == 1):
                    proximity_cost += 1/(di**2 + dj**2 + 1e-6)  # Inverse square law
    
    return path_length + 10*proximity_cost  # Weighted sum
# Initialize particles
particles = np.random.randint(0, rows, (POP_SIZE, NUM_WAYPOINTS * 2))
velocities = np.zeros_like(particles)
personal_best = particles.copy()
personal_best_scores = np.array([fitness_function(p,grid) for p in particles])
global_best = personal_best[np.argmin(personal_best_scores)]

for iter in range(MAX_ITER):
    for i in range(POP_SIZE):
        r1 = np.random.rand(NUM_WAYPOINTS * 2)
        r2 = np.random.rand(NUM_WAYPOINTS * 2)

        velocities[i] = (
            INERTIA * velocities[i]
            + C1 * r1 * (personal_best[i] - particles[i])
            + C2 * r2 * (global_best - particles[i])
        )
        particles[i] = np.clip(particles[i] + velocities[i], 0, rows - 1).astype(int)

        current_score = fitness_function(particles[i])
        if current_score < personal_best_scores[i]:
            personal_best[i] = particles[i].copy()
            personal_best_scores[i] = current_score

            if current_score < fitness_function(global_best):
                global_best = particles[i].copy()

# Generate final path
waypoints = [default_start] + list(global_best.reshape(-1, 2)) + [default_goal]
path = [(int(p[0]), int(p[1])) for p in waypoints]

# Show results
create_grid_map(grid, path)
