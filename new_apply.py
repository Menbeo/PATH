import numpy as np
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from gridmap import compute_neighborhood_layers
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



def bspline_smooth(waypoints, grid, num_points=200, degree=3, smoothing_factor=None,penalty_strength = 0.1, meters_per_cell = 1.0, inflation_radius = 1.8):
    
    if len(waypoints) <= degree:
        return waypoints  # Not enough points for the spline

      # --- STEP 1: Generate the cost grid and its gradient ---
    cost_grid = compute_neighborhood_layers(grid, inflation_radius, meters_per_cell)
    # The gradient of the cost grid points "uphill" towards higher cost (i.e., towards obstacles)
    grad_y, grad_x = np.gradient(cost_grid)

    # --- STEP 2: Generate the initial smooth B-spline ---
    if smoothing_factor is None:
        smoothing_factor = len(waypoints)
    
    x, y = zip(*waypoints)
    tck, u = interpolate.splprep([x, y], s=smoothing_factor, k=degree)
    unew = np.linspace(0, 1, num_points)
    initial_smooth_points = interpolate.splev(unew, tck)

    # --- STEP 3: Iteratively refine the path based on cost ---
    smoothed_path = []
    grid_height, grid_width = grid.shape

    for i in range(num_points):
        px, py = initial_smooth_points[0][i], initial_smooth_points[1][i]

        # Get integer coordinates for grid lookup, ensuring they are within bounds
        p_col = int(np.clip(px, 0, grid_width - 1))
        p_row = int(np.clip(py, 0, grid_height - 1))
        
        # Get the cost at the current point's location
        cost = cost_grid[p_row, p_col]

        # If the cost is lethal (hard obstacle), we must push it out.
        # This is a fallback for when the penalty push isn't enough.
        if cost >= 255:
            # Fallback: find direction to nearest free space (as in previous answer)
            # This logic is simplified here; a full implementation would use a distance-to-free transform.
            # For simplicity, we just use the cost gradient which is already computed.
            gy, gx = grad_y[p_row, p_col], grad_x[p_row, p_col]
            if gy != 0 or gx != 0:
                # Move strongly away from the obstacle
                direction = -np.array([gx, gy]) / np.linalg.norm([gx, gy])
                px += direction[0] * meters_per_cell # Push by one cell
                py += direction[1] * meters_per_cell
                
        # If we are in a penalty zone (but not a hard obstacle)
        elif cost > 0:
            # Get the gradient vector at the point. This points towards the obstacle.
            gy, gx = grad_y[p_row, p_col], grad_x[p_row, p_col]
            
            if gy != 0 or gx != 0:
                # We want to move *away* from the obstacle, so we use the negative gradient.
                push_direction = -np.array([gx, gy]) / np.linalg.norm([gx, gy])
                
                # The magnitude of the push is proportional to the cost and the penalty_strength.
                # A higher cost means a stronger push.
                push_magnitude = penalty_strength * (cost / 254.0)
                
                # Apply the push
                px += push_direction[0] * push_magnitude
                py += push_direction[1] * push_magnitude
            
        smoothed_path.append((px, py))

    return smoothed_path