for map_id in range(1, 5):
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
