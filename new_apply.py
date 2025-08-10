import numpy as np
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi


# def smooth_path(path, angle_threshold=45):
#     if len(path) < 3:
#         return path
#     smoothed = [path[0]]
#     for i in range(1, len(path)-1):
#         v1 = np.array(path[i]) - np.array(path[i-1])
#         v2 = np.array(path[i+1]) - np.array(path[i])
#         angle = angle_between(v1, v2)
#         if angle > angle_threshold:
#             smoothed.append(path[i])
#     smoothed.append(path[-1])
#     return smoothed


def bspline_smooth(waypoints, grid, num_points=200, degree=3, smoothing_factor=None, inflation_radius=3):
    
    if len(waypoints) <= degree:
        return waypoints  # Not enough points for the spline

    if smoothing_factor is None:
        smoothing_factor = len(waypoints)

    x, y = zip(*waypoints)
    tck, u = interpolate.splprep([x, y], s=smoothing_factor, k=degree)
    unew = np.linspace(0, 1, num_points)
    initial_smooth_points = interpolate.splev(unew, tck)

    inv_grid = (grid == 1)
    distance_map_to_free = distance_transform_edt(inv_grid == 0)


    inflated_obstacles = distance_map_to_free <= inflation_radius
    grad_y, grad_x = np.gradient(distance_map_to_free)
    smoothed_path = []
    grid_height, grid_width = grid.shape

    for i in range(num_points):
        px, py = initial_smooth_points[0][i], initial_smooth_points[1][i]
        p_col = int(np.clip(px, 0, grid_width - 1))
        p_row = int(np.clip(py, 0, grid_height - 1))

        dist = distance_map_to_free[p_row, p_col]
        g_y = grad_y[p_row, p_col]
        g_x = grad_x[p_row, p_col]

        if g_y != 0 or g_x != 0:
            direction = np.array([-g_x, -g_y]) / np.linalg.norm([g_x, g_y])
        else:
            direction = None

        # 1. If inside obstacle → push out strongly
        if grid[p_row, p_col] == 1 and direction is not None:
            px += direction[0] * dist * 1.1
            py += direction[1] * dist * 1.1

        # 2. If in inflated "danger zone" → apply softer penalty away from obstacle
        elif inflated_obstacles[p_row, p_col] and direction is not None:
            penalty_strength = (inflation_radius - dist) / inflation_radius # 0 → far edge, 1 → close
            
            push_amount = (penalty_strength * penalty_strength) / 1_000_000  # normalize huge cost
            px += direction[0] * push_amount
            py += direction[1] * push_amount

        smoothed_path.append((px, py))

    return smoothed_path