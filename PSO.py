import math
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.path import Path
from matplotlib.patches import Patch # For custom legend elements
from gridmap import create_grid_map, grid_map, default_goal,default_start
from gridmap import convert_grid_to_lat_lon,compute_neighborhood_layers
from convert_to_waypoints import export_waypoints

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

    cost = 0.0
    if grid_data[x_int, y_int] == 1:
        cost += 1_000_000
    elif neighborhood_layers_data[x_int, y_int] == 1:
        cost += 50_000

    distance_to_goal = np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)
    cost += distance_to_goal * 100
    return cost

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

def particle_swarm_optimization(map_id, show_plot=False):
    node_expand = 0 
    grid = grid_map(map_id=map_id, size=GRID_SIZE)
    neighborhood_layers = compute_neighborhood_layers(grid)
    swarm = Swarm(POPULATION, V_MAX, grid, neighborhood_layers, default_goal, default_start)

    for curr_iter in range(MAX_ITER):
        for particle in swarm.particles:
            node_expand += 1
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
                particle.pos[0], particle.pos[1], grid, neighborhood_layers, default_goal
            )
            particle.update_best_pos(current_cost)

            if particle.best_pos_cost < swarm.global_best_pos_cost:
                swarm.global_best_pos = particle.best_pos.copy()
                swarm.global_best_pos_cost = particle.best_pos_cost
                swarm.global_best_path = particle.personal_path.copy()

        dist = np.linalg.norm(swarm.global_best_pos - np.array(default_goal))
        if dist < CONVERGENCE_DISTANCE and swarm.global_best_pos_cost < 1000:
            print(f"PSO converged on map {map_id} at iteration {curr_iter + 1}")
            break

    # Convert to integer path
    int_path = [(int(round(p[0])), int(round(p[1]))) for p in swarm.global_best_path]

    # Export waypoints
    latlon_path = [convert_grid_to_lat_lon(x, y) for x, y in int_path]
    export_waypoints(latlon_path, filename=f"PSO_map{map_id}.waypoints")

    # Optional plot
    if show_plot:
        plt.figure(figsize=(6, 6))
        color_grid = np.ones((*grid.shape, 3))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] == 1:
                    color_grid[x, y] = [1, 1, 0]  # Obstacle: Yellow
                elif neighborhood_layers[x, y] >= 1:
                    color_grid[x, y] = [1, 0, 0]  # Danger: Red
                else:
                    color_grid[x, y] = [1, 1, 1]  # Free: White
        plt.imshow(color_grid, origin='upper')

        if len(int_path) > 1:
            px = [p[1] for p in int_path]
            py = [p[0] for p in int_path]
            plt.plot(px, py, color='blue', label='PSO Path', linewidth=2)

        plt.plot(default_start[1], default_start[0], 'go', label='Start')
        plt.plot(default_goal[1], default_goal[0], 'ro', label='Goal')
        plt.title(f"PSO - Map {map_id}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return int_path,node_expand
# --- Main Execution ---
if __name__ == "__main__":
    for i in range(1,5):
        particle_swarm_optimization(map_id=i)
