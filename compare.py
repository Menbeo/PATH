import csv
import psutil
import time
import numpy as np

from gridmap import grid_map, default_start, default_goal, compute_neighborhood_layers
from Dijkstra import Dijkstra
from Julastar import astar, simplify_path as simplify_astar
from Julrrt import rrt, simplify_path as simplify_rrt
from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
from PSO import particle_swarm_optimization

# ========== Setup CSVs ==========
csv_files = {
    "length": open("path_length.csv", "w", newline=''),
    "time": open("execution_time.csv", "w", newline=''),
    "memory": open("memory_usage.csv", "w", newline=''),
    "nodes": open("node_expansion.csv", "w", newline=''),
}
csv_writers = {key: csv.writer(f) for key, f in csv_files.items()}
for writer in csv_writers.values():
    writer.writerow(["Algorithm", "Map", "Run", "Value"])

def memory_usage_MB():
    return psutil.Process().memory_info().rss / 1024**2

def record(algorithm, map_id, run_id, path, start_time, memory_before,node_expand):
    elapsed_time = time.time() - start_time
    mem_used = memory_usage_MB() - memory_before
    path_len = len(path) if path else 0

    csv_writers["length"].writerow([algorithm, map_id, run_id, path_len])
    csv_writers["time"].writerow([algorithm, map_id, run_id, elapsed_time])
    csv_writers["memory"].writerow([algorithm, map_id, run_id, mem_used])
    csv_writers["nodes"].writerow([algorithm, map_id, run_id, node_expand])

# ========== Main Loop ==========
for map_id in range(1, 5):
    print(f"=== MAP {map_id} ===")
    for run_id in range(1, 31):  # 30 runs
        print(f"\n[Map {map_id} - Run {run_id}]")

        grid = grid_map(map_id)
        inflation = compute_neighborhood_layers(grid)

        # --- Dijkstra ---
        memory_before = memory_usage_MB()
        start = time.time()
        path, node_expand = Dijkstra(grid, default_start, default_goal, inflation_layer=inflation)
        record("Dijkstra", map_id, run_id, path, start, memory_before, node_expand)

        # --- A* ---
        memory_before = memory_usage_MB()
        start = time.time()
        path, node_expand = astar(grid, default_start, default_goal)
        path = simplify_astar(grid, path) if path else []
        record("Astar", map_id, run_id, path, start, memory_before, node_expand)

        # --- RRT ---
        memory_before = memory_usage_MB()
        start = time.time()
        path, node_expand = rrt(grid, inflation, default_start, default_goal)
        path = simplify_rrt(grid, path) if path else []
        record("RRT", map_id, run_id, path, start, memory_before, node_expand)

        # --- PRM ---
        samples = sample_points(300, grid)
        samples.append(default_start)
        samples.append(default_goal)
        start_idx = len(samples) - 2    
        goal_idx = len(samples) - 1
        graph = connect_nodes(samples, radius=50, grid=grid, inflation=inflation)

        memory_before = memory_usage_MB()
        start = time.time()
        try:
            path_idx, node_expand = prm_dijkstra(graph, start_idx, goal_idx)
            path = [samples[i] for i in path_idx] if path_idx else []
        except:
            path = []
        record("PRM", map_id, run_id, path, start, memory_before, node_expand)

        # --- PSO ---
        memory_before = memory_usage_MB()
        start = time.time()
        path, node_expand = particle_swarm_optimization(map_id, show_plot=False)
        record("PSO", map_id, run_id, path, start, memory_before, node_expand)

# ========== Close Files ==========
for f in csv_files.values():
    f.close()

print("\n✅ Done. Data saved in:")
print("- path_length.csv")
print("- execution_time.csv")
print("- memory_usage.csv")
print("- node_expansion.csv")
