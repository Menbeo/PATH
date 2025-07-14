#import gridmap
from gridmap import create_grid_map, grid_map, default_goal,default_start,convert_grid_to_lat_lon
import numpy as np 
import matplotlib.pyplot as plt
import heapq
from convert_to_waypoints import export_waypoints

#==== DIJKSTRA'S PATH ====
def Dijkstra(grid, start, goal):
    #declare rows and cols 
    rows,cols = grid.shape
    visited = set()
    previous = {}
    distance = {start:0}
    pq = [(0, start)]
    directions = [(-1, 0), (1,0), (0,-1), (0,1)]

    while pq:
        cost, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 0: 
                    new_cost= cost + 1
                    if new_cost < distance.get(neighbor, float('inf')):
                        distance[neighbor] = new_cost
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_cost, neighbor))
        
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
    for map_id in range(1,5):
        grid = grid_map(map_id=map_id)
        path = Dijkstra(grid,default_start, default_goal)
        if not path: 
            print("no path")
        else:
            create_grid_map(grid, path)

