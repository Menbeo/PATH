from gridmap import grid_map, default_start, default_goal, convert_grid_to_lat_lon, compute_neighborhood_layers
from convert_to_waypoints import export_waypoints
from Dijkstra import Dijkstra
from Julastar import astar, simplify_path as simplify_astar
from Julrrt import rrt, simplify_path as simplify_rrt
from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from PSO import particle_swarm_optimization
import time


summary_lines = []

def path_length(path):
    return len(path) if path else 0

def save_and_report(name, map_id, path, grid=None):
    if path:
        length = len(path)
        latlon_path = [convert_grid_to_lat_lon(x, y) for (x, y) in path]
        filename = f"{name}_map{map_id}.waypoints"
        export_waypoints(latlon_path, filename)
        print(f"{name} on Map {map_id}: Path length = {length}")
        summary_lines.append(f"{name} on Map {map_id}: Path length = {length}")
    else:
        print(f"{name} on Map {map_id}: No path found")
        summary_lines.append(f"{name} on Map {map_id}: No path found")

def plot_path(grid, paths_dict, map_id):
    color_map = {
        "Dijkstra": 'blue',
        "Astar": 'green',
        "RRT": 'orange',
        "PRM": 'purple',
        "PSO": 'red',
    }
    
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='gray_r', origin='upper')
    plt.title(f"Paths on Map {map_id}")
    
    for name, path in paths_dict.items():
        if path:
            path_y = [p[1] for p in path]
            path_x = [p[0] for p in path]
            plt.plot(path_y, path_x, label=name, color=color_map.get(name, 'black'))

    # Mark start and goal
    plt.plot(default_start[1], default_start[0], 'go', label='Start', markersize=8)
    plt.plot(default_goal[1], default_goal[0], 'ro', label='Goal', markersize=8)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================
# Compare across all algorithms
# ==============================
for map_id in range(1, 5):
    print(f"\n=== Map {map_id} ===")
    grid = grid_map(map_id=map_id)
    inflation = compute_neighborhood_layers(grid)

    paths_to_plot = {}

    # --- Dijkstra ---
    start_time = time.time()
    path_dijkstra = Dijkstra(inflation, default_start, default_goal)
    elapsed_time = time.time() - start_time
    print(f"Dijkstra on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    summary_lines.append(f"Dijkstra on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    save_and_report("Dijkstra", map_id, path_dijkstra)
    paths_to_plot["Dijkstra"] = path_dijkstra

    # --- A* ---
    start_time = time.time()
    path_astar = astar(grid, default_start, default_goal)
    elapsed_time = time.time() - start_time
    print(f"Astar on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    summary_lines.append(f"Astar on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    simplified_astar = simplify_astar(grid, path_astar) if path_astar else []
    save_and_report("Astar", map_id, simplified_astar)
    paths_to_plot["Astar"] = simplified_astar

    # --- RRT ---
    start_time = time.time()
    path_rrt = rrt(grid, inflation, default_start, default_goal)
    elapsed_time = time.time() - start_time
    print(f"RRT on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    summary_lines.append(f"RRT on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    simplified_rrt = simplify_rrt(grid, path_rrt) if path_rrt else []
    save_and_report("RRT", map_id, simplified_rrt)
    paths_to_plot["RRT"] = simplified_rrt

    # --- PRM ---
    from probalisitc_road_map import is_free, line_free
    samples = sample_points(300, grid)
    samples.append(default_start)
    samples.append(default_goal)
    start_idx = len(samples) - 2
    goal_idx = len(samples) - 1
    graph = connect_nodes(samples, radius=50, grid=grid, inflation=inflation)

    try:
        start_time = time.time()
        path_idx = prm_dijkstra(graph, start_idx, goal_idx)
        elapsed_time = time.time() - start_time
        print(f"PRM on Map {map_id}: Time = {elapsed_time:.4f} seconds")
        summary_lines.append(f"PRM on Map {map_id}: Time = {elapsed_time:.4f} seconds")
        path_prm = [samples[i] for i in path_idx] if path_idx else []
        save_and_report("PRM", map_id, path_prm)
        paths_to_plot["PRM"] = path_prm
    except Exception:
        print(f"PRM on Map {map_id}: No path found")
        summary_lines.append(f"PRM on Map {map_id}: No path found")
        paths_to_plot["PRM"] = []

    # --- PSO ---
    start_time = time.time()
    path_pso = particle_swarm_optimization(map_id, show_plot=False)
    elapsed_time = time.time() - start_time
    print(f"PSO on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    summary_lines.append(f"PSO on Map {map_id}: Time = {elapsed_time:.4f} seconds")
    save_and_report("PSO", map_id, path_pso)
    paths_to_plot["PSO"] = path_pso

    # === PLOT ALL PATHS FOR THIS MAP ===
    plot_path(grid, paths_to_plot, map_id)
    
# === Save all summary results to a text file ===
with open("pathfinding_summary.txt", "w") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("\nSummary saved to 'pathfinding_summary.txt'")
