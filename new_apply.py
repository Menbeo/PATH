import numpy as np
from scipy import interpolate

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

def turn_constraint(path, obstacle_distance, caution_distance=1.0, safe_distance=2.0):
    if len(path) < 3:
        return path
    
    # Determine dynamic angle threshold
    if obstacle_distance <= caution_distance:
        angle_threshold = 30
    elif obstacle_distance >= safe_distance:
        angle_threshold = 45
    else:
        ratio = (obstacle_distance - caution_distance) / (safe_distance - caution_distance)
        angle_threshold = 30 + ratio * (45 - 30)

    smoothed = [path[0]]
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i]) - np.array(path[i - 1])
        v2 = np.array(path[i + 1]) - np.array(path[i])
        angle = angle_between(v1, v2)
        if angle > angle_threshold:
            smoothed.append(path[i])
    smoothed.append(path[-1])
    return smoothed

def is_point_safe(point, grid):
    x, y = int(round(point[0])), int(round(point[1]))
    height, width = grid.shape
    if 0 <= y < height and 0 <= x < width:
        return grid[y, x] == 0  # 0 = free, 1 = obstacle
    return False

def bspline_smooth(path, grid, smoothing_factor=None, num_points=10):
    path = np.array(path, dtype=float)
    if len(path) < 3:
        return path

    x = path[:, 0]
    y = path[:, 1]
    k = min(3, len(path) - 1)

    if smoothing_factor is None:
        smoothing_factor = len(path) * 2.0

    # Fit B-spline
    tck, _ = interpolate.splprep([x, y], s=smoothing_factor, k=k)

    # Evaluate spline
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)

    smooth_path = []
    for pt in zip(x_fine, y_fine):
        if is_point_safe(pt, grid):
            smooth_path.append(pt)
        else:
            # If hit obstacle, stop smoothing and use original path
            return path

    return np.array(smooth_path)