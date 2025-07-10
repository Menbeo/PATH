from gridmap import create_grid_map, grid_map, default_goal, default_start, convert_grid_to_lat_lon
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import bisect

# Configuration parameters
NUM_SAMPLES = 300
CONNECTION_RADIUS = 50
OBSTACLE_MARGIN = 1

def is_free(x, y, grid, margin=0):
    """Check if cell and surrounding margin are obstacle-free"""
    x, y = int(x), int(y)
    for dx in range(-margin, margin+1):
        for dy in range(-margin, margin+1):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0):
                return False
    return True

def line_free(p1, p2, grid):
    """Bresenham's line algorithm for efficient collision checking"""
    x0, y0 = map(int, p1)
    x1, y1 = map(int, p2)
    
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    while True:
        if not is_free(x0, y0, grid, OBSTACLE_MARGIN):
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return True

def sample_points(n, grid, obstacle_bias=0.3):
    """Strategic sampling with obstacle-biased sampling"""
    samples = []
    h, w = grid.shape
    obstacle_edges = []
    
    # Find obstacle boundaries once
    for x in range(h):
        for y in range(w):
            if grid[x, y] == 1:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0:
                        obstacle_edges.append((nx, ny))
    
    while len(samples) < n:
        # Biased sampling near obstacles
        if obstacle_edges and random.random() < obstacle_bias:
            x, y = random.choice(obstacle_edges)
        else:
            # Uniform sampling in free space
            x, y = random.randint(0, h-1), random.randint(0, w-1)
        
        if is_free(x, y, grid):
            samples.append((x, y))
    return samples

def connect_nodes(samples, radius, grid):
    """Efficient radius-based connection without SciPy"""
    graph = defaultdict(list)
    n = len(samples)
    radius_sq = radius ** 2
    
    # Sort samples by x-coordinate for range search optimization
    sorted_samples = sorted([(p[0], p[1], i) for i, p in enumerate(samples)])
    x_coords = [p[0] for p in sorted_samples]
    
    for idx, (x1, y1, i) in enumerate(sorted_samples):
        # Find potential neighbors using x-coordinate range
        left = bisect.bisect_left(x_coords, x1 - radius)
        right = bisect.bisect_right(x_coords, x1 + radius)
        
        for j in range(left, right):
            x2, y2, neighbor_idx = sorted_samples[j]
            if i == neighbor_idx:
                continue
                
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            
            if dist_sq <= radius_sq:
                if line_free((x1,y1), (x2,y2), grid):
                    dist = math.sqrt(dist_sq)
                    graph[i].append((neighbor_idx, dist))
    
    return graph

def dijkstra(graph, start_idx, goal_idx):
    """Optimized Dijkstra's implementation"""
    dist = {node: float('inf') for node in graph}
    prev = {}
    dist[start_idx] = 0
    heap = [(0, start_idx)]
    
    while heap:
        current_dist, current = heapq.heappop(heap)
        
        if current == goal_idx:
            break
            
        if current_dist > dist[current]:
            continue
            
        for neighbor, weight in graph.get(current, []):
            new_dist = current_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current
                heapq.heappush(heap, (new_dist, neighbor))
    
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

def simplify_path(path, grid, max_waypoints=10):
    """Path simplification with waypoint limit"""
    if len(path) <= max_waypoints:
        return path
    
    # Try simple downsampling first
    step = max(1, len(path) // max_waypoints)
    simplified = path[::step][:max_waypoints]
    simplified[-1] = path[-1]  # Ensure goal is included
    
    # Verify the simplified path is collision-free
    for i in range(1, len(simplified)):
        if not line_free(simplified[i-1], simplified[i], grid):
            # Fall back to evenly spaced points if simplification fails
            step = len(path) // (max_waypoints - 1)
            indices = [i*step for i in range(max_waypoints-1)] + [len(path)-1]
            return [path[i] for i in indices]
    
    return simplified

def export_waypoints(lat_lon_path, filename="PRM.waypoints", default_altitude=50):
    """Optimized waypoint export"""
    header = "QGC WPL 110\n"
    template = "{i}\t{current}\t3\t16\t0.0\t0.0\t0.0\t0.0\t{lat:.8f}\t{lon:.8f}\t{alt:.2f}\t1\n"
    
    with open(filename, 'w') as f:
        f.write(header)
        for i, (lat, lon) in enumerate(lat_lon_path):
            f.write(template.format(
                i=i,
                current=1 if i == 0 else 0,
                lat=lat,
                lon=lon,
                alt=default_altitude
            ))
    print(f"Exported {len(lat_lon_path)} waypoints to {filename}")

if __name__ == "__main__":
    for map_id in range(1, 5):
        print(f"\nProcessing Map {map_id}")
        grid = grid_map(map_id=map_id)
        
        # Validate start/goal positions
        if not (is_free(*default_start, grid) and is_free(*default_goal, grid)):
            print("Start or goal position invalid")
            continue
            
        # Generate roadmap
        samples = sample_points(NUM_SAMPLES, grid)
        samples.extend([default_start, default_goal])
        start_idx, goal_idx = len(samples)-2, len(samples)-1
        
        # Build connection graph
        graph = connect_nodes(samples, CONNECTION_RADIUS, grid)
        
        # Find path
        path_idx = dijkstra(graph, start_idx, goal_idx)
        if not path_idx:
            print("No viable path found")
            continue
            
        # Process path
        path = [samples[i] for i in path_idx]
        simplified_path = simplify_path(path, grid)
        
        # Convert and export
        lat_lon_path = [convert_grid_to_lat_lon(y, x) for x, y in simplified_path]
        export_waypoints(lat_lon_path, f"PRM_map{map_id}.waypoints")
        
        # Visualization
        create_grid_map(grid, simplified_path)