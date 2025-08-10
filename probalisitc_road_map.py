from gridmap import create_grid_map, grid_map, default_goal, default_start
from gridmap import convert_grid_to_lat_lon, compute_neighborhood_layers
from convert_to_waypoints import export_waypoints
import numpy as np 
import random 
import math 
import networkx as nx 
from new_apply import bspline_smooth


def is_free(x, y, grid):
    x = int(x)
    y = int(y)
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 0

def line_free(p1, p2, grid,inflation):
    steps = int(max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))) + 1
    for i in range(steps + 1):
        x = int(p1[0] + (p2[0] - p1[0]) * i / steps)
        y = int(p1[1] + (p2[1] - p1[1]) * i / steps)
        if not is_free(x, y, grid):
            return False
        if inflation[x, y] == 1:  # Dangerous zone
            return False
    return True

def sample_points(n, grid):
    samples = []
    h, w = grid.shape
    while len(samples) < n:
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        if is_free(x, y, grid):
            samples.append((x, y))
    return samples

def connect_nodes(samples, radius, grid, inflation):
    graph = {i: [] for i in range(len(samples))}
    for i, p1 in enumerate(samples):
        for j, p2 in enumerate(samples):
            if i != j and math.dist(p1, p2) <= radius:
                if line_free(p1, p2, grid, inflation):
                    weight = math.dist(p1, p2)
                    graph[i].append((j, weight))  # ✅ tuple
    return graph

def dijkstra(graph, start_idx,  goal_idx):
    dist = {i: float('inf') for i in graph}
    prev = {}
    dist[start_idx] = 0
    visited = set()
    inflation = compute_neighborhood_layers(grid)
    node_expand = 0 
    while True:
        current = None
        min_dist = float('inf')
        for node in graph:
            if node not in visited and dist[node] < min_dist:
                current = node
                min_dist = dist[node]
        if current is None or current == goal_idx:
            break
        visited.add(current)
        node_expand += 1

        for neighbor, weight in graph[current]:
            if neighbor in visited:
                continue
            x, y = samples[neighbor]
            layer = inflation[int(x), int(y)]

            if layer == 1:
                layer_cost = 100
            else:
                layer_cost = 1

            new_dist = dist[current] + weight * layer_cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current

    # Reconstruct path
    path = []
    node = goal_idx
    while node in prev:
        path.append(node)
        node = prev[node]
    if path:
        path.append(start_idx)
        path.reverse()
    return path

if __name__ == "__main__":
    for map_id in range(1,5):

        print(f"\n=== Running PRM on Map {map_id} ===")
        grid = grid_map(map_id=map_id)
        samples = sample_points(300, grid)
        samples.append(default_start)
        samples.append(default_goal)

        start_idx = len(samples) - 2
        goal_idx = len(samples) - 1
        inflation = compute_neighborhood_layers(grid)
        graph = connect_nodes(samples, radius=50, grid=grid, inflation=inflation)

    
        try:
            path_idx = dijkstra(graph, start_idx, goal_idx)
            path = [samples[i] for i in path_idx]
            smooth = bspline_smooth(path, grid, inflation)
            lat_lon_path = [
                    convert_grid_to_lat_lon(point_y, point_x) for point_x, point_y in smooth
                ]
            filename = f"PRM_map{4}.waypoints"
            export_waypoints(lat_lon_path, filename=filename, default_altitude=100)
             
            create_grid_map(grid,smooth)

        except nx.NetworkXNoPath:
            print("No path found")
            create_grid_map(grid,None)
           
