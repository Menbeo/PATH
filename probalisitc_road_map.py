from gridmap import create_grid_map, grid_map, default_goal, default_start
# Import the conversion function from gridmap
from gridmap import convert_grid_to_lat_lon

import numpy as np 
import random 
import math 
import matplotlib.pyplot as plt

def animate_path(grid, path, delay=0.01):
    plt.figure(figsize=(10, 10))
    plt.title("PRM Path Animation")
    plt.imshow(grid, cmap='gray_r', origin='upper')

    # Draw start and goal
    plt.plot(default_start[1], default_start[0], 'go', markersize=10, label='Start')
    plt.plot(default_goal[1], default_goal[0], 'ro', markersize=10, label='Goal') # Changed to 'ro' for consistency
    plt.legend()

    # Draw path one point at a time
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        plt.plot([y0, y1], [x0, x1], 'b-', linewidth=2)
        plt.pause(delay)

    plt.grid(True)
    plt.show()

def is_free(x, y, grid):
    x = int(x)
    y = int(y)
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 0

def line_free(p1, p2, grid):
    steps = int(max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))) + 1
    # Handle cases where p1 and p2 are the same point to avoid division by zero if steps is 0
    if steps == 1: 
        return is_free(p1[0], p1[1], grid)
    
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

    import heapq 

    pq = [(0, start_idx)] 

    while pq:
        d, current = heapq.heappop(pq)

        if current == goal_idx:
            break

        if current in visited:
            continue
        visited.add(current)
        
        # Check if current exists in graph before iterating (for robustness)
        if current not in graph:
            continue

        for neighbor, weight in graph[current]:
            if neighbor in visited:
                continue
            new_dist = dist[current] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

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

def export_waypoints(lat_lon_path, filename="PRM.waypoints", default_altitude=50):
    with open(filename, 'w') as f:
        f.write("QGC WPL 110 \n")
        for i, (lat, lon) in enumerate(lat_lon_path):
            is_current = 1 if i == 0 else 0
            autocontinue = 3
            command = 16
            param1 = param2 = param3 = param4 = 0.0
            frame = 1
            line = (
                f"{i}\t{is_current}\t{autocontinue}\t{command}\t"
                f"{param1:.8f}\t{param2:.8f}\t{param3:.8f}\t{param4:.8f}\t"
                f"{lat:.8f}\t{lon:.8f}\t{default_altitude:.2f}\t{frame}\n"
            )
            f.write(line)
    print(f"Exported to: {filename}")


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
        graph = connect_nodes(samples, radius=70, grid=grid)
        path_idx = dijkstra(graph, start_idx, goal_idx)
        if path_idx:
            path = [samples[i] for i in path_idx]
            animate_path(grid, path)
            lat_lon_path = [
                convert_grid_to_lat_lon(point_y, point_x) for point_x, point_y in path
            ]
        else:
            print("No path found.")
            create_grid_map(grid, None)
