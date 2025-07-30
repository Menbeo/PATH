import time
import csv
import tracemalloc
from memory_profiler import memory_usage
from concurrent.futures import ProcessPoolExecutor, as_completed

def single_run(map_id, run_id):
    import tracemalloc
    import time
    from memory_profiler import memory_usage
    from gridmap import grid_map, default_start, default_goal, compute_neighborhood_layers
    from Dijkstra import Dijkstra
    from Julastar import astar, simplify_path as simplify_astar
    from Julrrt import rrt, simplify_path as simplify_rrt
    from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
    from PSO import particle_swarm_optimization

    def profile(func):
        tracemalloc.start()
        start_time = time.time()
        mem_before = memory_usage(-1, interval=0.01, timeout=1)[0]
        try:
            result = func()
        except:
            result = []
        mem_after = memory_usage(-1, interval=0.01, timeout=1)[0]
        exec_time = time.time() - start_time
        memory_used = mem_after - mem_before
        tracemalloc.stop()
        return result, exec_time, memory_used

    def path_len(p): return len(p) if p else 0

    grid = grid_map(map_id)
    inflation = compute_neighborhood_layers(grid)

    results = []

    # Dijkstra
    def run_dijkstra():
        return Dijkstra(inflation, default_start, default_goal)
    dpath, dtime, dmem = profile(run_dijkstra)
    results.append((map_id, "Dijkstra", run_id, path_len(dpath), dtime, len(dpath), dmem))

    # A*
    def run_astar():
        return simplify_astar(grid, astar(grid, default_start, default_goal))
    apath, atime, amem = profile(run_astar)
    results.append((map_id, "Astar", run_id, path_len(apath), atime, len(apath), amem))

    # RRT
    def run_rrt():
        return simplify_rrt(grid, rrt(grid, inflation, default_start, default_goal))
    rpath, rtime, rmem = profile(run_rrt)
    results.append((map_id, "RRT", run_id, path_len(rpath), rtime, len(rpath), rmem))

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
        ppath, ptime, pmem = profile(run_prm)
    except:
        ppath, ptime, pmem = [], 0, 0
    results.append((map_id, "PRM", run_id, path_len(ppath), ptime, len(ppath), pmem))

    # PSO
    def run_pso():
        return particle_swarm_optimization(map_id, show_plot=False)
    psopath, psotime, psomem = profile(run_pso)
    results.append((map_id, "PSO", run_id, path_len(psopath), psotime, len(psopath), psomem))

    return results

# Run all jobs in parallel
all_results = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(single_run, map_id, run_id) for map_id in range(1, 5) for run_id in range(1, 50)]
    for future in as_completed(futures):
        all_results.extend(future.result())

# Write combined results
with open("results_combined.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Map", "Algorithm", "Run", "Path Length", "Execution Time", "Nodes", "Memory Usage"])
    writer.writerows(all_results)

print("\nParallel benchmark complete. Results saved to results_combined.csv")
