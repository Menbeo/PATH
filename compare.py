import csv
import psutil
import time
import numpy as np
import math

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
    "smoothness": open("path_smoothness.csv", "w", newline=''),
    "turn_angles": open("total_turning_angles.csv", "w", newline=''),
    "turn_count": open("turn_count.csv", "w", newline='')
}
csv_writers = {key: csv.writer(f) for key, f in csv_files.items()}
for writer in csv_writers.values():
    writer.writerow(["Algorithm", "Map", "Run", "Value"])


# ========== Helper Functions ==========
def memory_usage_MB():
    return psutil.Process().memory_info().rss / 1024**2

def compute_turn_metrics(path):
    """Return smoothness, total_turn_angle, turn_count"""
    if len(path) < 3:
        return 0.0, 0.0, 0

    total_angle = 0.0
    turn_count = 0
    heading_changes = []

    for i in range(1, len(path) - 1):
        p1 = np.array(path[i - 1], dtype=float)
        p2 = np.array(path[i], dtype=float)
        p3 = np.array(path[i + 1], dtype=float)

        v1 = p2 - p1
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue

        # Angle between v1 and v2
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.acos(cos_angle)  # radians
        angle_deg = math.degrees(angle)

        if angle_deg > 1e-3:  # small threshold to ignore straight lines
            total_angle += angle_deg
            turn_count += 1
            heading_changes.append(angle_deg)

    # Smoothness: smaller total_angle = smoother path
    smoothness = 1 / (1 + total_angle)  # normalized: closer to 1 = smoother

    return smoothness, total_angle, turn_count

def record(algorithm, map_id, run_id, path, start_time, memory_before):
    elapsed_time = time.time() - start_time
    mem_used = memory_usage_MB() - memory_before
    path_len = len(path) if path else 0

    csv_writers["length"].writerow([algorithm, map_id, run_id, path_len])
    csv_writers["time"].writerow([algorithm, map_id, run_id, elapsed_time])
    csv_writers["memory"].writerow([algorithm, map_id, run_id, mem_used])


    if path:
        smoothness, total_angle, turn_count = compute_turn_metrics(path)
    else:
        smoothness, total_angle, turn_count = 0.0, 0.0, 0

    csv_writers["smoothness"].writerow([algorithm, map_id, run_id, smoothness])
    csv_writers["turn_angles"].writerow([algorithm, map_id, run_id, total_angle])
    csv_writers["turn_count"].writerow([algorithm, map_id, run_id, turn_count])


# ========== Main Loop ==========
for map_id in range(1, 5):
    print(f"=== MAP {map_id} ===")
    for run_id in range(1, 101):
        print(f"\n[Map {map_id} - Run {run_id}]")

        grid = grid_map(map_id)
        inflation = compute_neighborhood_layers(grid)

        # --- Dijkstra ---
        memory_before = memory_usage_MB()
        start = time.time()
        path = Dijkstra(grid, default_start, default_goal, inflation_layer=inflation)
        record("Dijkstra", map_id, run_id, path, start, memory_before)

        # --- A* ---
        memory_before = memory_usage_MB()
        start = time.time()
        path = astar(grid, default_start, default_goal)
        path = simplify_astar(grid, path) if path else []
        record("Astar", map_id, run_id, path, start, memory_before)

        # --- RRT ---
        memory_before = memory_usage_MB()
        start = time.time()
        path = rrt(grid, inflation, default_start, default_goal)
        path = simplify_rrt(grid, path) if path else []
        record("RRT", map_id, run_id, path, start, memory_before)

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
        record("PRM", map_id, run_id, path, start, memory_before)

        # --- PSO ---
        memory_before = memory_usage_MB()
        start = time.time()
        path = particle_swarm_optimization(map_id)
        record("PSO", map_id, run_id, path, start, memory_before)

# ========== Close Files ==========
for f in csv_files.values():
    f.close()

print("\nData saved in:")
for name in csv_files.keys():
    print(f"- {name}.csv")
