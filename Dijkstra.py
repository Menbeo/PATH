from gridmaplv3 import create_grid_map,grid_map, default_goal, default_start
import numpy as np 
import matplotlib.pyplot as plt
import heapq

def dijkstra(start, goal, grid):
    rows, cols = grid.shape
    visited = set()
    distance = {start: 0}
    previous = {}
    queue = [(0, start)]
    directions = [(-1,0), (1,0), (0,-1), (0,1)] 
    while queue:
        cost,current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
       
        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 0 and neighbor not in visited:
                    new_cost = cost + 1
                    if new_cost < distance.get(neighbor, float('inf')):
                        distance[neighbor] = new_cost
                        previous[neighbor] = current
                        heapq.heappush(queue,(new_cost,neighbor))
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = previous.get(node)
        if node is None:
            return [] 
    path.append(start)
    path.reverse()
    return path

if __name__ == "__main__":
    grid = grid_map()
    path = dijkstra(default_start, default_goal, grid)
    if not path:
        print("No path")
    else:
        create_grid_map(grid, path)

def export_waypoints(lat_lon_path: list[tuple[float,float]], filename  = "")
        


