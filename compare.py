import csv
import time
import tracemalloc
import psutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from memory_profiler import memory_usage

from gridmap import create_grid_map  # Your function to generate/load a map
from Dijkstra import Dijkstra
from Julastar import astar, simplify_path as simplify_astar
from Julrrt import rrt, simplify_path as simplify_rrt
from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
from PSO import particle_swarm_optimization

algorithms = {
    'Dijkstra': Dijkstra,
    'A*': astar,
    'RRT': rrt,
    'PRM': prm_dijkstra,
    'PSO': particle_swarm_optimization
}

headers = ['Algorithm', 'Map', 'Run', 'Path Length', 'Execution Time', 'Memory Usage (MB)', 'Node Usage']


def run_algorithm(algorithm_name, pathfinder, grid, inflated_grid):
    tracemalloc.start()
    start_time = time.perf_counter()
    mem_usage, (path, nodes) = memory_usage((pathfinder, (grid, inflated_grid)), retval=True, max_usage=True)
    exec_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if path is None or len(path) < 2:
        path_length = float('inf')
    else:
        path_length = sum(((x0 - x1)**2 + (y0 - y1)**2)**0.5 for ((x0, y0), (x1, y1)) in zip(path[:-1], path[1:]))

    return path_length, exec_time, mem_usage, len(nodes)


def single_run(map_id, run_id):
    results = []
    grid, inflated_grid = create_grid_map(map_id)
    for algo_name, pathfinder in algorithms.items():
        path_length, exec_time, mem, node_count = run_algorithm(algo_name, pathfinder, grid, inflated_grid)
        results.append([
            algo_name, f"Map {map_id}", run_id, path_length, exec_time, mem, node_count
        ])
    return results


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    output_file = 'results_combined.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        all_results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(single_run, map_id, run_id)
                       for map_id in range(1, 5)  # Maps 1 to 4
                       for run_id in range(1, 21)]  # 500 runs each

            for future in as_completed(futures):
                result_rows = future.result()
                all_results.extend(result_rows)

        writer.writerows(all_results)

    print(f"All results written to {output_file}")
