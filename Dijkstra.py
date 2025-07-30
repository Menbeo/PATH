#import gridmap
from gridmap import create_grid_map, grid_map, default_goal,default_start
from gridmap import convert_grid_to_lat_lon,compute_neighborhood_layers
import numpy as np 
import matplotlib.pyplot as plt
import heapq
from convert_to_waypoints import export_waypoints

def Dijkstra(grid, start, goal, inflation_layer=None):
    rows, cols = grid.shape
    visited = set()
    previous = {}
    distance = {start: 0}
    pq = [(0, start)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    node_expand = 0
    while pq:
        cost, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        node_expand += 1
        if current == goal:
            break

        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 0:
                    # === Apply cost based on inflation layer ===
                    if inflation_layer is not None and inflation_layer[neighbor] == 1:
                        layer_cost = 1_000_000
                    else:
                        layer_cost = 1

                    new_cost = cost + layer_cost

                    if new_cost < distance.get(neighbor, float('inf')):
                        distance[neighbor] = new_cost
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_cost, neighbor))
    if goal not in previous:
        # If goal is not reached
        return [], node_expand
    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = previous.get(node)
        if node is None:
            return []
    path.append(start)
    path.reverse()
    return path, node_expand

        
if __name__ == "__main__":
    for map_id in range(1, 5):
        grid = grid_map(map_id=map_id)
        inflation = compute_neighborhood_layers(grid, inflation_radius=1.8, meters_per_cell=1.0)

        # Use original grid for obstacles, and inflation as layer cost
        path = Dijkstra(grid, default_start, default_goal, inflation_layer=inflation)

        if not path:
            print(f"Map {map_id}: No path found")
        else:
            print(f"Map {map_id}: Path found with {len(path)} steps")
            create_grid_map(inflation, path)  # Optional: show on inflated map
            lat_lon_path = [convert_grid_to_lat_lon(x, y) for x, y in path]
            filename = f"Dijkstra_map{map_id}.waypoints"
            export_waypoints(lat_lon_path, filename=filename)

