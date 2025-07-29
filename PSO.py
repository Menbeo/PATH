import numpy as np
import random
import math
import matplotlib.pyplot as plt
from gridmap import compute_neighborhood_layers, grid_map, default_goal, default_start
from gridmap import convert_grid_to_lat_lon, create_grid_map
from convert_to_waypoints import export_waypoints

# Configuration
NUM_WAYPOINTS = 10
NUM_PARTICLES = 200
NUM_ITERATIONS = 500
W = 0.5   # Inertia
C1 = 2.0  # Cognitive
C2 = 1.5  # Social
V_MAX = 1.0



# Evaluate entire path cost
def path_cost(path, grid, inflated_grid):
    total_cost = 0
    for i in range(1, len(path)):
        dist = math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        total_cost += dist
        x, y = int(round(path[i][0])), int(round(path[i][1]))
        if grid[x, y] == 1:
            total_cost += 1000
        elif inflated_grid[x, y] >= 1:
            total_cost += 10000
        if i > 1:
            v1_x, v1_y = path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1]
            v2_x, v2_y = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            cos_angle = (v1_x * v2_x + v1_y * v2_y) / (math.hypot(v1_x, v1_y) * math.hypot(v2_x, v2_y) + 1e-6)
            total_cost += 10 * (1 - cos_angle)
        
    return total_cost

def clamp(p, grid_shape):
    x = max(0, min(grid_shape[0] - 1, int(round(p[0]))))
    y = max(0, min(grid_shape[1] - 1, int(round(p[1]))))
    return (x, y)

def initialize_particles(grid):
    particles = []
    velocities = []
    for _ in range(NUM_PARTICLES):
        path = [default_start]
        for t in range(1, NUM_WAYPOINTS + 1):
            t = t / (NUM_WAYPOINTS + 1)
            x = int(default_start[0] + t * (default_goal[0] - default_start[0]) + random.uniform(-5, 5))
            y = int(default_start[1] + t * (default_goal[1] - default_start[1]) + random.uniform(-5, 5))
            x, y = clamp((x, y), grid.shape)
            while grid[x, y] == 1:  # Ensure no obstacle
                x = random.randint(0, grid.shape[0] - 1)
                y = random.randint(0, grid.shape[1] - 1)
            path.append((x, y))
        path.append(default_goal)
        particles.append(path)
        velocities.append([(random.uniform(-V_MAX, V_MAX), random.uniform(-V_MAX, V_MAX)) for _ in range(NUM_WAYPOINTS + 2)])
    return particles, velocities

def run_pso(grid):
    inflated = compute_neighborhood_layers(grid)
    particles, velocities = initialize_particles(grid)
    personal_best = particles[:]
    personal_best_scores = [path_cost(p, grid, inflated) for p in particles]
    global_best = personal_best[np.argmin(personal_best_scores)]

    for it in range(NUM_ITERATIONS):
        for i in range(NUM_PARTICLES):
            for j in range(1, NUM_WAYPOINTS + 1):  # skip start and goal
                r1, r2 = random.random(), random.random()
                px, py = particles[i][j]
                vx, vy = velocities[i][j]
                pbest_x, pbest_y = personal_best[i][j]
                gbest_x, gbest_y = global_best[j]

                vx = W * vx + C1 * r1 * (pbest_x - px) + C2 * r2 * (gbest_x - px)
                vy = W * vy + C1 * r1 * (pbest_y - py) + C2 * r2 * (gbest_y - py)
                if inflated[int(px), int(py)] >= 1:
                    vx += random.uniform(-0.5, 0.5) * V_MAX
                    vy += random.uniform(-0.5, 0.5) * V_MAX
                vx = max(-V_MAX, min(V_MAX, vx))
                vy = max(-V_MAX, min(V_MAX, vy))
                new_pos = clamp((px + vx, py + vy), grid.shape)

                particles[i][j] = new_pos
                velocities[i][j] = (vx, vy)

            score = path_cost(particles[i], grid, inflated)
            if score < personal_best_scores[i]:
                personal_best[i] = list(particles[i])
                personal_best_scores[i] = score

        best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[best_idx]
        print(f"Iteration {it+1}/{NUM_ITERATIONS}, Best Score: {personal_best_scores[best_idx]:.2f}")

    return global_best

if __name__ == "__main__":
    for map_id in range(1, 5):
        print(f"\nRunning PSO on Map {map_id}")
        grid = grid_map(map_id)
        best_path = run_pso(grid)

        path_for_plot = [(int(p[0]), int(p[1])) for p in best_path]
        lat_lon_path = [convert_grid_to_lat_lon(y, x) for x, y in path_for_plot]

        filename = f"PSO_Map{map_id}.waypoints"
        export_waypoints(lat_lon_path, filename=filename, default_altitude=100)
        create_grid_map(grid, path_for_plot)
        plt.title(f"Optimized Path - Map {map_id}")
        print("Waypoints:", path_for_plot)
        plt.show()