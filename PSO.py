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

def is_collision_free(grid, path):
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        x_vals = np.linspace(x0, x1, 100)
        y_vals = np.linspace(y0, y1, 100)
        for x, y in zip(x_vals.astype(int), y_vals.astype(int)):
            if x < 0 or y < 0 or x >= rows or y >= cols or grid[x, y] == 1:
                return False
    return True

def fitness_function(particle):
    waypoints = [default_start] + list(particle.reshape(-1, 2)) + [default_goal]
    total_dist = 0
    for i in range(1, len(waypoints)):
        total_dist += np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i - 1]))
    if not is_collision_free(grid, waypoints):
        total_dist += PENALTY
    return total_dist

# Initialize particles
particles = np.random.randint(0, rows, (POP_SIZE, NUM_WAYPOINTS * 2))
velocities = np.zeros_like(particles)
personal_best = particles.copy()
personal_best_scores = np.array([fitness_function(p) for p in particles])
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
