import time
import csv
import tracemalloc
from memory_profiler import memory_usage

from gridmap import grid_map, default_start, default_goal, compute_neighborhood_layers
from convert_to_waypoints import export_waypoints
from Dijkstra import Dijkstra
from Julastar import astar, simplify_path as simplify_astar
from Julrrt import rrt, simplify_path as simplify_rrt
from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
from PSO import particle_swarm_optimization

# Helper function to profile memory and time
def profile_function(func, *args, **kwargs):
    start_time = time.time()
    tracemalloc.start()
    mem_before = memory_usage(-1, interval=0.01, timeout=1)[0]

    result = func(*args, **kwargs)

    mem_after = memory_usage(-1, interval=0.01, timeout=1)[0]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exec_time = time.time() - start_time
    memory_used = mem_after - mem_before  # in MB
    return result, exec_time, memory_used

def path_length(path):
    return len(path) if path else 0

# Initialize CSV files
csv_files = {
    "path_length": open("path_length.csv", "w", newline=""),
    "execution_time": open("execution_time.csv", "w", newline=""),
    "nodes_used": open("nodes_used.csv", "w", newline=""),
    "memory_usage": open("memory_usage.csv", "w", newline=""),
}
csv_writers = {k: csv.writer(f) for k, f in csv_files.items()}
for writer in csv_writers.values():
    writer.writerow(["Map", "Algorithm", "Run", "Value"])

# Benchmark loop
for map_id in range(1, 5):
    print(f"\n=== Processing Map {map_id} ===")
    for run in range(1, 501):
        print(f"Run {run}/500")
        grid = grid_map(map_id)
        inflation = compute_neighborhood_layers(grid)

        # Dijkstra
        def run_dijkstra():
            return Dijkstra(inflation, default_start, default_goal)
        path, exec_time, mem = profile_function(run_dijkstra)
        csv_writers["path_length"].writerow([map_id, "Dijkstra", run, path_length(path)])
        csv_writers["execution_time"].writerow([map_id, "Dijkstra", run, exec_time])
        csv_writers["memory_usage"].writerow([map_id, "Dijkstra", run, mem])
        csv_writers["nodes_used"].writerow([map_id, "Dijkstra", run, len(path)])

        # A*
        def run_astar():
            return simplify_astar(grid, astar(grid, default_start, default_goal))
        path, exec_time, mem = profile_function(run_astar)
        csv_writers["path_length"].writerow([map_id, "Astar", run, path_length(path)])
        csv_writers["execution_time"].writerow([map_id, "Astar", run, exec_time])
        csv_writers["memory_usage"].writerow([map_id, "Astar", run, mem])
        csv_writers["nodes_used"].writerow([map_id, "Astar", run, len(path)])

        # RRT
        def run_rrt():
            return simplify_rrt(grid, rrt(grid, inflation, default_start, default_goal))
        path, exec_time, mem = profile_function(run_rrt)
        csv_writers["path_length"].writerow([map_id, "RRT", run, path_length(path)])
        csv_writers["execution_time"].writerow([map_id, "RRT", run, exec_time])
        csv_writers["memory_usage"].writerow([map_id, "RRT", run, mem])
        csv_writers["nodes_used"].writerow([map_id, "RRT", run, len(path)])

        # PRM
        def run_prm():
            samples = sample_points(300, grid)
            samples.append(default_start)
            samples.append(default_goal)
            start_idx = len(samples) - 2
            goal_idx = len(samples) - 1
            graph = connect_nodes(samples, 50, grid, inflation)
            idx_path = prm_dijkstra(graph, start_idx, goal_idx)
            return [samples[i] for i in idx_path] if idx_path else []
        try:
            path, exec_time, mem = profile_function(run_prm)
        except:
            path, exec_time, mem = [], 0, 0
        csv_writers["path_length"].writerow([map_id, "PRM", run, path_length(path)])
        csv_writers["execution_time"].writerow([map_id, "PRM", run, exec_time])
        csv_writers["memory_usage"].writerow([map_id, "PRM", run, mem])
        csv_writers["nodes_used"].writerow([map_id, "PRM", run, len(path)])

        # PSO
        def run_pso():
            return particle_swarm_optimization(map_id, show_plot=False)
        path, exec_time, mem = profile_function(run_pso)
        csv_writers["path_length"].writerow([map_id, "PSO", run, path_length(path)])
        csv_writers["execution_time"].writerow([map_id, "PSO", run, exec_time])
        csv_writers["memory_usage"].writerow([map_id, "PSO", run, mem])
        csv_writers["nodes_used"].writerow([map_id, "PSO", run, len(path)])

# Close CSV files
for f in csv_files.values():
    f.close()

print("\nBenchmark complete. Results saved in CSV files.")
