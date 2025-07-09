from gridmap import create_grid_map,grid_map, default_goal, default_start, convert_grid_to_lat_lon
import numpy as np 
import matplotlib.pyplot as plt
import heapq

def dijkstra(start, goal, grid, min_straight_steps = 2):
    rows, cols = grid.shape
    visited = set()
    distance = {start: 0}
    previous = {}
    queue = []
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    heapq.heappush(queue,(0, start, None))
    distance[(start,None)] = 0

    while queue:
        cost,current,prev_dir, straight_steps = heapq.heappop(queue)
        
        if (current,prev_dir) in visited:
            continue
        visited.add((current,prev_dir))
        if current == goal:
            break
       
        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 0:
                    new_dir = (dx,dy)
                    if prev_dir is None or new_dir == prev_dir:
                        turn_penalty = 0
                        new_straight_steps = straight_steps + 1
                    else:
                        # disallow sharp turns if not moved enough
                        if straight_steps < min_straight_steps:
                            continue  # skip this turn
                        turn_penalty = 5  # optional: add a penalty
                        new_straight_steps = 1  # reset

                    new_cost = cost + 1 + turn_penalty

                    if new_cost < distance.get((neighbor, new_dir), float('inf')):
                        distance[(neighbor,new_dir)] = new_cost
                        previous[neighbor] = current
                        heapq.heappush(queue,(new_cost,neighbor,new_dir,new_straight_steps))
    # Reconstruct path
    node = goal
    path = []
    while node != start:
        path.append(node)
        node = previous.get(node)
        if node is None:
            return []
    path.append(start)
    path.reverse()
    return path


def export_waypoints(lat_lon_path: list[tuple[float,float]], filename  = "Dijkstra.waypoints", default_altitude=100):
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
    for map_id in range(1,4):
        grid = grid_map(map_id=map_id)
        path = dijkstra(default_start, default_goal, grid,min_straight_steps = 2)
        if not path:
            print("No path")
        else:
            create_grid_map(grid, path)
            #convert to latitude and longitude 
            lat_lon_path = [convert_grid_to_lat_lon(x,y) for x,y in path]
            filename = f"Dijkstra_map{map_id}.waypoints"
            export_waypoints(lat_lon_path, filename=filename)
        

    
        


