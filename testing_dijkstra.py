from gridmap import create_grid_map,grid_map, default_goal, default_start, convert_grid_to_lat_lon
import numpy as np 
import matplotlib.pyplot as plt
import heapq
import math

def simplify_path(path, max_waypoints=10, initial_tolerance=2.0):
    if len(path) <= max_waypoints:
        return path
    
    def rdp_simplify(points, tol):
        if len(points) <= 2:
            return points
        max_dist = 0
        index = 0
        for i in range(1, len(points)-1):
            dist = perpendicular_distance(points[i], points[0], points[-1])
            if dist > max_dist:
                max_dist = dist
                index = i
        if max_dist > tol:
            left = rdp_simplify(points[:index+1], tol)
            right = rdp_simplify(points[index:], tol)
            return left[:-1] + right
        return [points[0], points[-1]]
    
    def perpendicular_distance(point, line_start, line_end):
        if line_start == line_end:
            return math.dist(point, line_start)
        numerator = abs(
            (line_end[0]-line_start[0])*(line_start[1]-point[1]) - 
            (line_start[0]-point[0])*(line_end[1]-line_start[1])
        )
        denominator = math.dist(line_start, line_end)
        return numerator / denominator
    
    
    low, high = 0.0, initial_tolerance
    best_path = path
    for _ in range(10):
        mid = (low + high) / 2
        simplified = rdp_simplify(path, mid)
        if len(simplified) > max_waypoints:
            low = mid
        else:
            high = mid
            best_path = simplified
            if len(best_path) == max_waypoints:
                break
    
    # If still too many, take evenly spaced points
    if len(best_path) > max_waypoints:
        step = len(best_path) / (max_waypoints - 1)
        indices = [int(i*step) for i in range(max_waypoints-1)] + [len(best_path)-1]
        best_path = [best_path[i] for i in indices]
    
    return best_path

def dijkstra(start, goal, grid, obstacle_penalty_radius=4):
    rows, cols = grid.shape
    directions = [(-1,0), (1,0), (0,-1), (0,1)] 
    #Safety cost map (near obstacles)
    safety_cost = np.zeros_like(grid, dtype=float)
    for i in range(rows):
        for j in range(cols):
            if grid[i,j] == 1:  # Obstacle
                safety_cost[i,j] = float('inf')  # Unpassable
            else:
                # Check cells around (i,j) for obstacles
                min_dist_to_obstacle = float('inf')
                for di in range(-obstacle_penalty_radius, obstacle_penalty_radius+1):
                    for dj in range(-obstacle_penalty_radius, obstacle_penalty_radius+1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni,nj] == 1:
                            dist = (di**2 + dj**2)**0.5  # Euclidean distance
                            if dist < min_dist_to_obstacle:
                                min_dist_to_obstacle = dist
                # Assign higher cost to cells closer to obstacles
                safety_cost[i,j] = 1 + max(0, (obstacle_penalty_radius - min_dist_to_obstacle)) * 10
    
    visited = set()
    distance = {start: 0}
    previous = {}
    queue = [(0, start)]
    
    while queue:
        cost, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        
        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if safety_cost[neighbor] < float('inf'):  # Passable
                    new_cost = cost + safety_cost[neighbor]
                    if new_cost < distance.get(neighbor, float('inf')):
                        distance[neighbor] = new_cost
                        previous[neighbor] = current
                        heapq.heappush(queue, (new_cost, neighbor))
    
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
    return path


def export_waypoints(lat_lon_path: list[tuple[float,float]], filename  = "Dijkstra.waypoints", default_altitude=1000):
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
    for map_id in range(1,5):
        grid = grid_map(map_id=map_id)
        path = dijkstra(default_start, default_goal, grid)
        if not path:
            print("No path")
        else:
            simplified_path = simplify_path(path, max_waypoints=10, initial_tolerance=4.0)
            create_grid_map(grid, simplified_path)
            #convert to latitude and longitude 
            lat_lon_path = [convert_grid_to_lat_lon(x,y) for x,y in simplified_path]
            filename = f"Dijkstra_map{map_id}.waypoints"
            export_waypoints(lat_lon_path, filename=filename)
        

    
        


