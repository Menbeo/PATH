import csv
import psutil
import time
import tracemalloc
import numpy as np
from gridmap import create_grid_map, default_start, default_goal, compute_neighborhood_layers, grid_map


# CSV files
csv_files = {
    "length": open("path_length.csv", "w", newline=''),
    "time": open("execution_time.csv", "w", newline=''),
    "node": open("node_usage.csv", "w", newline=''),
    "memory": open("memory_usage.csv", "w", newline='')
}

csv_writers = {key: csv.writer(f) for key, f in csv_files.items()}
for writer in csv_writers.values():
    writer.writerow(["Algorithm", "Map", "Run", "Value"])

def memory_usage_MB():
    return psutil.Process().memory_info().rss / 1024**2

def record(algorithm, map_id, run_id, path, start_time, memory_before, node_count=None):
    elapsed_time = time.time() - start_time
    mem_used = memory_usage_MB() - memory_before
    path_len = len(path) if path else 0

    csv_writers["length"].writerow([algorithm, map_id, run_id, path_len])
    csv_writers["time"].writerow([algorithm, map_id, run_id, elapsed_time])
    csv_writers["memory"].writerow([algorithm, map_id, run_id, mem_used])
    if node_count is not None:
        csv_writers["node"].writerow([algorithm, map_id, run_id, node_count])
    else:
        csv_writers["node"].writerow([algorithm, map_id, run_id, "N/A"])

# Main loop
for map_id in range(1, 5):
    from Dijkstra import Dijkstra
    print(f"\n=== Map {map_id} ===")
    grid = grid_map(map_id=map_id)
    inflation = compute_neighborhood_layers(grid, inflation_radius=1.8, meters_per_cell=1.0)

    # --- Dijkstra ---
    path = Dijkstra(grid, default_start, default_goal, inflation_layer=inflation)
    if not path:
        print(f"Map {map_id}: Dijkstra - No path found")
    else:
        print(f"Map {map_id}: Dijkstra - Path with {len(path)} steps")

    # --- A* ---
    from Julastar import astar, simplify_path as simplify_astar
    path = astar(grid, default_start, default_goal)
    simplified_astar = simplify_astar(grid, path) if path else []
    if not simplified_astar:
        print(f"Map {map_id}: A* - No path found")
    else:
        print(f"Map {map_id}: A* - Path with {len(simplified_astar)} steps")

    # --- RRT ---
    from Julrrt import rrt, simplify_path as simplify_rrt
    path = rrt(grid, inflation, default_start, default_goal)
    simplified_rrt = simplify_rrt(grid, path) if path else []
    if not simplified_rrt:
        print(f"Map {map_id}: RRT - No path found")
    else:
        print(f"Map {map_id}: RRT - Path with {len(simplified_rrt)} steps")

    # --- PRM ---
    from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
    samples = sample_points(300, grid)
    samples.append(default_start)
    samples.append(default_goal)
    start_idx = len(samples) - 2
    goal_idx = len(samples) - 1
    graph = connect_nodes(samples, radius=50, grid=grid, inflation=inflation)

    try:
        path_idx = prm_dijkstra(graph, start_idx, goal_idx)
        path_prm = [samples[i] for i in path_idx] if path_idx else []
        if not path_prm:
            print(f"Map {map_id}: PRM - No path found")
        else:
            print(f"Map {map_id}: PRM - Path with {len(path_prm)} steps")
    except:
        print(f"Map {map_id}: PRM - No path found")

    # --- PSO ---
    from PSO import particle_swarm_optimization
    path = particle_swarm_optimization(map_id, show_plot=False)
    if not path:
        print(f"Map {map_id}: PSO - No path found")
    else:
        print(f"Map {map_id}: PSO - Path with {len(path)} steps")

# Close CSV files
for f in csv_files.values():
    f.close()

print("✅ Benchmarking complete. Results saved in CSV files.")
