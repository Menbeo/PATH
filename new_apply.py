import numpy as np
from scipy import interpolate

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

def turn_constraint(path, obstacle_distance, caution_distance=1.0, safe_distance=2.0):
    if len(path) < 3:
        return path

    if obstacle_distance <= caution_distance:
        angle_threshold = 30
    elif obstacle_distance >= safe_distance:
        angle_threshold = 45
    else:
        ratio = (obstacle_distance - caution_distance) / (safe_distance - caution_distance)
        angle_threshold = 30 + ratio * (45 - 30)

    smoothed = [tuple(path[0])]
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i]) - np.array(path[i - 1])
        v2 = np.array(path[i + 1]) - np.array(path[i])
        angle = angle_between(v1, v2)
        if angle > angle_threshold:
            smoothed.append(tuple(path[i]))
    smoothed.append(tuple(path[-1]))
    return smoothed

def is_point_safe(point, grid):
    # keep the same indexing convention as your older code:
    # point = (x, y) where grid is indexed as grid[y, x]
    x, y = int(round(point[0])), int(round(point[1]))
    height, width = grid.shape
    if 0 <= y < height and 0 <= x < width:
        return grid[y, x] == 0  # 0 = free, 1 = obstacle
    return False

def nearest_safe_point(point, grid, max_radius=5):
    """Find nearest free cell to 'point' by increasing Manhattan radius.
       Returns float (x,y) of the free cell center or None if not found."""
    cx = int(round(point[0]))
    cy = int(round(point[1]))
    height, width = grid.shape

    for r in range(max_radius + 1):
        # iterate over square ring of radius r
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r and r != 0:
                    # skip inner points when r>0 to only check the ring
                    continue
                nx = cx + dx
                ny = cy + dy
                if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] == 0:
                    return (float(nx), float(ny))
    return None

def bspline_smooth(path, grid, smoothing_factor=None, num_points=250, max_clamp_radius=4):
    """B-spline smoothing that clamps spline sample points to nearest free grid cell instead of aborting."""
    path = np.array(path, dtype=float)
    if len(path) < 3:
        return path

    x = path[:, 0]
    y = path[:, 1]
    k = min(3, len(path) - 1)

    if smoothing_factor is None:
        # Allow more deviation from original points; tweak multiplier to taste
        smoothing_factor = len(path) * 12.0

    # Fit B-spline
    try:
        tck, _ = interpolate.splprep([x, y], s=smoothing_factor, k=k)
    except Exception as e:
        # fallback: return original path if fitting fails
        return path

    # Evaluate spline densely
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)

    smooth_path = []
    for pt in zip(x_fine, y_fine):
        if is_point_safe(pt, grid):
            smooth_path.append((float(pt[0]), float(pt[1])))
        else:
            # clamp to nearest free cell (if found)
            ns = nearest_safe_point(pt, grid, max_radius=max_clamp_radius)
            if ns is not None:
                smooth_path.append(ns)
            else:
                # if no nearby free cell found, fall back to last safe sample or original rounded point
                if smooth_path:
                    smooth_path.append(smooth_path[-1])
                else:
                    smooth_path.append((float(round(pt[0])), float(round(pt[1]))))

    # remove consecutive duplicates (exact floats) which may create sharp discrete steps
    dedup = []
    for p in smooth_path:
        if not dedup or (abs(dedup[-1][0] - p[0]) > 1e-6 or abs(dedup[-1][1] - p[1]) > 1e-6):
            dedup.append(p)

    return np.array(dedup)
