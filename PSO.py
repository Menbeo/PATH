import numpy as np
import random
import math
from gridmap import compute_neighborhood_layers, grid_map, default_goal, default_start

# Define objective: distance to goal + penalty for being in obstacle/inflation
def grid_objective(pos, grid, inflated_grid, goal):
    x, y = pos
    if grid[x, y] == 1:
        return 1e6  # hard penalty for obstacle
    elif inflated_grid[x, y] == 1:
        return 1000  # soft penalty for dangerous zone
    else:
        # Euclidean distance to goal
        gx, gy = goal
        return math.hypot(x - gx, y - gy)

# Clamp position inside grid
def clamp_position(pos, grid_shape):
    x, y = pos
    x = max(0, min(grid_shape[0]-1, x))
    y = max(0, min(grid_shape[1]-1, y))
    return (x, y)

# Run PSO in grid
def run_pso(grid, start, goal, n_particles=50, n_iterations=100):
    inflated_grid = compute_neighborhood_layers(grid)

    # Initialize particles
    particles = [start]
    while len(particles) < n_particles:
        x = random.randint(0, grid.shape[0]-1)
        y = random.randint(0, grid.shape[1]-1)
        if grid[x, y] == 0 and inflated_grid[x, y] == 0:
            particles.append((x, y))

    velocities = [(0, 0)] * n_particles
    personal_best = list(particles)
    personal_best_scores = [grid_objective(p, grid, inflated_grid, goal) for p in particles]

    # Global best
    global_best = personal_best[np.argmin(personal_best_scores)]

    # PSO constants
    w = 0.5
    c1 = 1.5
    c2 = 2.0

    # Start PSO
    for it in range(n_iterations):
        for i in range(n_particles):
            vx, vy = velocities[i]
            px, py = particles[i]
            pbest_x, pbest_y = personal_best[i]
            gbest_x, gbest_y = global_best

            # Update velocity (discrete rounding)
            new_vx = w * vx + c1 * random.random() * (pbest_x - px) + c2 * random.random() * (gbest_x - px)
            new_vy = w * vy + c1 * random.random() * (pbest_y - py) + c2 * random.random() * (gbest_y - py)

            new_px = round(px + new_vx)
            new_py = round(py + new_vy)
            new_pos = clamp_position((new_px, new_py), grid.shape)

            velocities[i] = (new_vx, new_vy)
            particles[i] = new_pos

            score = grid_objective(new_pos, grid, inflated_grid, goal)
            if score < personal_best_scores[i]:
                personal_best[i] = new_pos
                personal_best_scores[i] = score

        global_best = personal_best[np.argmin(personal_best_scores)]
        print(f"Iteration {it+1}/{n_iterations}, Best score: {grid_objective(global_best, grid, inflated_grid, goal):.2f}")

    return global_best

if __name__ == "__main__":
    for map_id in range(1,5):
        grid = grid_map(map_id==map_id)   
        start = default_start
        goal = default_goal
        best_path = run_pso(grid, start, goal)
        print(f"Best path found: {best_path}")
