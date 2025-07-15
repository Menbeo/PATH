from gridmap import create_grid_map, grid_map, default_goal, default_start
from gridmap import convert_grid_to_lat_lon, compute_neighborhood_layers
from convert_to_waypoints import export_waypoints
import numpy as np 
import random 
import math 
import heapq


def is_free(x, y, grid):
    x = int(x)
    y = int(y)
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x,y] == 0 
   
def line_free(p1, p2, grid):
    steps = int(max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))) + 1
    for i in range(steps + 1):
        x = int(p1[0] + (p2[0] - p1[0]) * i / steps)
        y = int(p1[1] + (p2[1] - p1[1]) * i / steps)
        if not is_free(x, y, grid):
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

def connect_nodes(samples, radius, grid):
    graph = {i: [] for i in range(len(samples))}
    for i, p1 in enumerate(samples):
        for j, p2 in enumerate(samples):
            if i != j and math.dist(p1, p2) <= radius:
                if line_free(p1, p2, grid):
                    graph[i].append((j, math.dist(p1, p2)))
    return graph

def dijkstra(graph, start_idx, goal_idx):
    dist = {i: float('inf') for i in graph}
    prev = {}
    dist[start_idx] = 0
    visited = set()

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
        for neighbor, weight in graph[current]:
            if neighbor in visited:
                continue
            new_dist = dist[current] + weight
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
    for map_id in range(1, 5):
        print(f"\n=== Running PRM on Map {map_id} ===")
        grid = grid_map(map_id=map_id)

        assert is_free(default_start[0], default_start[1], grid), "Start in obstacle"
        assert is_free(default_goal[0], default_goal[1], grid), "Goal in obstacle"

        samples = sample_points(600, grid)
        samples.append(default_start)
        samples.append(default_goal)

        start_idx = len(samples) - 2
        goal_idx = len(samples) - 1

        graph = connect_nodes(samples, radius=50, grid=grid)
        path_idx = dijkstra(graph, start_idx, goal_idx)

        if path_idx:
            path = [samples[i] for i in path_idx]
            create_grid_map(grid,path)

            lat_lon_path = [
                convert_grid_to_lat_lon(point_y, point_x) for point_x, point_y in path
            ]

            print("\n--- Path in Latitude and Longitude ---")
            for i, (lat, lon) in enumerate(lat_lon_path):
                print(f"Point {i+1}: Latitude: {lat:.6f}, Longitude: {lon:.6f}")

         
            filename = f"PRM_map{map_id}.waypoints"
            export_waypoints(lat_lon_path, filename=filename)

        else:
            print("No path found.")
            