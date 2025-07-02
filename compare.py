# compare_path_planners.py
import time
import numpy as np
from gridmap import grid_map, create_grid_map, default_start, default_goal

# Import algorithms
from Dijkstra import dijkstra
from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
from PSO import fitness_function as pso_fitness_function  # Avoid re-running global PSO
import matplotlib.pyplot as plt

# Helper function
def compute_path_length(path):
    if not path or len(path) < 2:
        return float('inf')
    return sum(np.linalg.norm(np.array(p2) - np.array(p1)) for p1, p2 in zip(path[:-1], path[1:]))

# 1. Load Grid
grid = grid_map()

# === Dijkstra ===
start_time = time.time()
path_dijkstra = dijkstra(default_start, default_goal, grid)
time_dijkstra = time.time() - start_time
length_dijkstra = compute_path_length(path_dijkstra)

# === PRM ===
start_time = time.time()
samples = sample_points(600, grid)
samples.append(default_start)
samples.append(default_goal)
start_idx = len(samples) - 2
goal_idx = len(samples) - 1
graph = connect_nodes(samples, radius=50, grid=grid)
path_indices = prm_dijkstra(graph, start_idx, goal_idx)
path_prm = [samples[i] for i in path_indices] if path_indices else []
time_prm = time.time() - start_time
length_prm = compute_path_length(path_prm)

# === PSO ===
# We reload PSO from scratch to execute it freshly and retrieve path
import importlib.util
spec = importlib.util.spec_from_file_location("PSO", "PSO.py")
pso_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pso_module)
path_pso = pso_module.path
time_pso = pso_module.MAX_ITER * 0.01  # Approx estimation or you can time again
length_pso = compute_path_length(path_pso)

# === Comparison Summary ===
print("\n=== Path Planning Comparison ===")
print(f"Dijkstra: Length = {length_dijkstra:.2f}, Time = {time_dijkstra:.3f}s")
print(f"PRM     : Length = {length_prm:.2f}, Time = {time_prm:.3f}s")
print(f"PSO     : Length = {length_pso:.2f}, Time = approx {time_pso:.3f}s")

# === Plot all paths on same map ===
plt.figure(figsize=(12, 12))
plt.title("Comparison of Path Planners")
plt.imshow(grid, cmap='gray_r', origin='upper')

def plot_path(path, label, color):
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color=color, label=label, linewidth=2)

plot_path(path_dijkstra, "Dijkstra", "blue")
plot_path(path_prm, "PRM", "orange")
plot_path(path_pso, "PSO", "green")

plt.plot(default_start[1], default_start[0], 'go', markersize=10, label='Start')
plt.plot(default_goal[1], default_goal[0], 'ro', markersize=10, label='Goal')
plt.legend()
plt.grid(True)
plt.show()
