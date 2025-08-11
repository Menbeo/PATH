from gridmap import grid_map, default_start, default_goal, compute_neighborhood_layers, create_grid_map
from Dijkstra import Dijkstra
from Julastar import astar, simplify_path as simplify_astar
from Julrrt import rrt, simplify_path as simplify_rrt
from probalisitc_road_map import sample_points, connect_nodes, dijkstra as prm_dijkstra
from PSO import particle_swarm_optimization
from new_apply import bspline_smooth  # your collision-aware B-spline

colors = {
    "Dijkstra": "blue",
    "A*": "orange",
    "RRT": "green",
    "PRM": "purple",
    "PSO": "red"
}

for map_id in range(1, 5):
    print(f"\n=== Map {map_id} ===")
    grid = grid_map(map_id)
    inflation = compute_neighborhood_layers(grid)
    all_smoothed_paths = {}

    # Dijkstra
    raw = Dijkstra(grid, default_start, default_goal, inflation_layer=inflation)
    all_smoothed_paths["Dijkstra"] = bspline_smooth(raw, grid, inflated_grid=inflation)

    # A*
    raw = astar(grid, default_start, default_goal, inflated_grid=inflation, inflation_penalty=5000)
    raw = simplify_astar(grid, raw) if raw else []
    all_smoothed_paths["A*"] = bspline_smooth(raw, grid, inflated_grid=inflation)

    # RRT
    raw = rrt(grid, inflation, default_start, default_goal)
    raw = simplify_rrt(grid, raw) if raw else []
    all_smoothed_paths["RRT"] = bspline_smooth(raw, grid, inflated_grid=inflation)

    # PRM
    samples = sample_points(300, grid)
    samples.append(default_start)
    samples.append(default_goal)
    start_idx = len(samples) - 2
    goal_idx = len(samples) - 1
    graph = connect_nodes(samples, radius=50, grid=grid, inflation=inflation)
    path_idx, _ = prm_dijkstra(graph, start_idx, goal_idx, samples, grid, inflation)
    raw = [samples[i] for i in path_idx] if path_idx else []
    all_smoothed_paths["PRM"] = bspline_smooth(raw, grid, inflated_grid=inflation)

    # PSO
    raw = particle_swarm_optimization(map_id=map_id)
    all_smoothed_paths["PSO"] = bspline_smooth(raw, grid, inflated_grid=inflation)

    # Show all paths on one map
    create_grid_map(
        inflation,
        list(all_smoothed_paths.values()),
        colors=list(colors.values()),  # use our color list
        labels=list(all_smoothed_paths.keys())  # legend labels
    )
