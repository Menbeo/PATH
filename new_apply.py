import numpy as np
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

def turn_constraint(path, grid, min_safe_dist = 2.0, max_safe_dist = 4.0):
    if len(path) < 3:
        return path
    obstacle_indices = np.argwhere(grid == 1)
    if obstacle_indices.size == 0:
        obstacle_tree = None
    else:
        obstacle_tree = KDTree(obstacle_indices)

    smoothed_path = [path[0]]

    for i in range(1, len(path) - 1):
        current_point = path[i]
        
        if obstacle_tree:
            dist_to_obstacle, _ = obstacle_tree.query(current_point)
        else:
            dist_to_obstacle = max_safe_dist + 1 # Effectively infinite distance

        # Dynamically set the angle threshold based on obstacle proximity.
        if dist_to_obstacle <= min_safe_dist:
            # Very close to an obstacle, require a sharp turn.
            angle_threshold = 30  # Stricter angle
        elif dist_to_obstacle >= max_safe_dist:
            # Far from any obstacle, allow a very smooth, wide turn.
            angle_threshold = 45  # More lenient angle
        else:
            # Linearly interpolate the angle threshold based on distance.
            ratio = (dist_to_obstacle - min_safe_dist) / (max_safe_dist - min_safe_dist)
            angle_threshold = 30 + ratio * (45 - 30)

        # Calculate the angle of the turn at the current waypoint.
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])

        # A zero vector can occur if points are duplicated.
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue

        turn_angle = angle_between(v1, v2)

        # If the turn is sharp enough given the obstacle proximity, keep the point.
        if turn_angle > angle_threshold:
            smoothed_path.append(path[i])

    smoothed_path.append(path[-1])
    return smoothed_path



def bspline_smooth(waypoints, grid, num_points=200, degree=3):
   
    if len(waypoints) <= degree:
        return waypoints # Not enough points to create a spline of the given degree
    
    distance_map = distance_transform_edt(grid == 0)

    x, y = zip(*waypoints)

    # Adding more knots can help the spline follow the path more closely.
    tck, u = interpolate.splprep([x, y], s=0, k=degree)
    
    unew = np.linspace(0, 1, num_points)
    smooth_points = interpolate.splev(unew, tck)
    
    smoothed_path = []
    for i in range(num_points):
        px, py = smooth_points[0][i], smooth_points[1][i]
        
        # Ensure the point is within the grid boundaries
        if not (0 <= px < grid.shape[0] and 0 <= py < grid.shape[1]):
            continue

        # Check for collision at the grid cell of the smoothed point
        if grid[int(px), int(py)] == 1:
            # If there is a collision, you could implement a more sophisticated
            # collision avoidance strategy, but for now, we just skip the point.
            # This is a simplification. A better method would be to find the nearest safe point.
            continue
            
        smoothed_path.append((px, py))

    return smoothed_path