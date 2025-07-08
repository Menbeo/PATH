from gridmap import create_grid_map,grid_map, default_goal, default_start, convert_grid_to_lat_lon
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


def export_waypoints(lat_lon_path: list[tuple[float,float]], filename  = "Dijkstra.waypoints", default_altitude=500):
    with open(filename, 'w') as f:
        f.write("QGC WPL 110 \n")
        for i, (lat,lon) in enumerate(lat_lon_path):
            waypoint_index = i
            is_current = 1 if i == 0 else 0
            autocontinue = 3
            command = 16

            #param
            param1 = 0.0
            param2 = 0.0 
            param3 = 0.0
            param4 = 0.0

            latitude = lat
            longitude = lon
            altitude = default_altitude
            frame = 1
            line = (
                f"{waypoint_index}\t{is_current}\t{autocontinue}\t{command}\t"
                f"{param1:.8f}\t{param2:.8f}\t{param3:.8f}\t{param4:.8f}\t"
                f"{latitude:.8f}\t{longitude:.8f}\t{altitude:.2f}\t{frame}\n"
            )
            f.write(line)
        print(f"Have exported to filename")


if __name__ == "__main__":
    grid = grid_map()
    path = dijkstra(default_start, default_goal, grid)
    if not path:
        print("No path")
    else:
        create_grid_map(grid, path)
        #convert to latitude and longitude 
        lat_lon_path = [convert_grid_to_lat_lon(x,y) for x,y in path]
        export_waypoints(lat_lon_path)
    

    
        


