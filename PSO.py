import math
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.path import Path
from matplotlib.patches import Patch
from gridmap import create_grid_map, grid_map, default_goal, default_start
from gridmap import convert_grid_to_lat_lon, compute_neighborhood_layers
from convert_to_waypoints import export_waypoints
from new_apply import bspline_smooth

# --- PSO Parameters ---
DIMENSIONS = 2
GRID_SIZE = 50
B_LO = 0
B_HI = GRID_SIZE - 1
POPULATION = 150
V_MAX = 1.0
PERSONAL_C = 2.0
SOCIAL_C = 2.0
CONVERGENCE_DISTANCE = 0.5
MAX_ITER = 500

def cost_function_grid(x, y, grid_data, neighborhood_layers_data, goal_pos):
    rows, cols = grid_data.shape
    x_int, y_int = int(round(x)), int(round(y))
    if not (0 <= x_int < rows and 0 <= y_int < cols):
        return float('inf')

    if grid_data[x_int, y_int] == 1:
        return float('inf')  # Block obstacles
    elif neighborhood_layers_data[x_int, y_int] == 1:
        return 50_000  # Heavy penalty for danger zone

    distance_to_goal = np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)
    if distance_to_goal < 0.5:
        return 0.0  # Perfect match
    return distance_to_goal * 500  # Stronger pull to goal

class Particle:
    def __init__(self, x_init, y_init, velocity_init):
        self.pos = np.array([x_init, y_init], dtype=float)
        self.velocity = np.array(velocity_init, dtype=float)
        self.best_pos = self.pos.copy()
        self.best_pos_cost = float('inf')
        self.personal_path = [self.pos.copy()]

    def update_position(self, new_pos):
        self.pos = new_pos
        self.personal_path.append(self.pos.copy())

    def update_best_pos(self, current_cost):
        if current_cost < self.best_pos_cost:
            self.best_pos = self.pos.copy()
            self.best_pos_cost = current_cost

class Swarm:
    def __init__(self, pop_size, v_max_limit, grid_data, neighborhood_layers_data, goal_pos, start_pos):
        self.particles = []
        self.global_best_pos = None
        self.global_best_pos_cost = float('inf')
        self.global_best_path = []

        for _ in range(pop_size):
            x = start_pos[0] + np.random.uniform(-1, 1)
            y = start_pos[1] + np.random.uniform(-1, 1)
            x_int, y_int = int(round(x)), int(round(y))

            while (not (0 <= x_int < GRID_SIZE and 0 <= y_int < GRID_SIZE)) or grid_data[x_int, y_int] == 1:
                x = start_pos[0] + np.random.uniform(-1, 1)
                y = start_pos[1] + np.random.uniform(-1, 1)
                x_int, y_int = int(round(x)), int(round(y))

            velocity = np.random.uniform(-v_max_limit, v_max_limit, DIMENSIONS)
            particle = Particle(x, y, velocity)
            current_cost = cost_function_grid(x, y, grid_data, neighborhood_layers_data, goal_pos)
            particle.update_best_pos(current_cost)
            self.particles.append(particle)

            if particle.best_pos_cost < self.global_best_pos_cost:
                self.global_best_pos = particle.best_pos.copy()
                self.global_best_pos_cost = particle.best_pos_cost
                self.global_best_path = particle.personal_path.copy()

def particle_swarm_optimization(map_id):
    grid = grid_map(map_id=map_id, size=GRID_SIZE)
    neighborhood_layers = compute_neighborhood_layers(grid)

    # Create inflated grid (danger zones treated as obstacles)
    inflated_grid = grid.copy()
    inflated_grid[neighborhood_layers >= 1] = 1

    swarm = Swarm(POPULATION, V_MAX, inflated_grid, neighborhood_layers, default_goal, default_start)

    for curr_iter in range(MAX_ITER):
        for particle in swarm.particles:
            r1 = np.random.uniform(0, 1, DIMENSIONS)
            r2 = np.random.uniform(0, 1, DIMENSIONS)
            inertia_weight = 0.9 - ((0.7 / MAX_ITER) * curr_iter)

            personal_term = PERSONAL_C * r1 * (particle.best_pos - particle.pos)
            social_term = SOCIAL_C * r2 * (swarm.global_best_pos - particle.pos)

            new_velocity = inertia_weight * particle.velocity + personal_term + social_term
            new_velocity = np.clip(new_velocity, -V_MAX, V_MAX)
            particle.velocity = new_velocity

            new_pos = particle.pos + particle.velocity
            new_pos = np.clip(new_pos, B_LO, B_HI)
            particle.update_position(new_pos)

            current_cost = cost_function_grid(
                particle.pos[0], particle.pos[1], inflated_grid, neighborhood_layers, default_goal
            )
            particle.update_best_pos(current_cost)

            if particle.best_pos_cost < swarm.global_best_pos_cost:
                swarm.global_best_pos = particle.best_pos.copy()
                swarm.global_best_pos_cost = particle.best_pos_cost
                swarm.global_best_path = particle.personal_path.copy()

        dist = np.linalg.norm(swarm.global_best_pos - np.array(default_goal))
        if dist < CONVERGENCE_DISTANCE:
            print(f"PSO converged on map {map_id} at iteration {curr_iter + 1}")
            break

    # Convert to integer path & remove duplicates
    int_path = [(int(round(p[0])), int(round(p[1]))) for p in swarm.global_best_path]
    filtered_path = [int_path[i] for i in range(len(int_path)) if i == 0 or int_path[i] != int_path[i-1]]

    # Force goal into path if missing
    if filtered_path[-1] != tuple(default_goal):
        filtered_path.append(tuple(default_goal))

    # Smooth path
    if len(filtered_path) > 3:
        smoothed_path = bspline_smooth(filtered_path, inflated_grid, degree=3, num_points=100)
    else:
        smoothed_path = filtered_path

    # # Export to waypoints
    # latlon_path = [convert_grid_to_lat_lon(x, y) for x, y in smoothed_path]
    # export_waypoints(latlon_path, filename=f"PSO_map{map_id}.waypoints")

    # # Display on map without interactive plot
    # create_grid_map(grid, smoothed_path)
    return smoothed_path

# --- Main Execution ---
if __name__ == "__main__":
    for i in range(1, 5):
        particle_swarm_optimization(map_id=i)
