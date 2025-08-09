import numpy as np
from scipy import interpolate
def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

def turn_constraint(path, angle_threshold=45):
    if len(path) < 3:
        return path
    smoothed = [path[0]]
    for i in range(1, len(path)-1):
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        angle = angle_between(v1, v2)
        if angle > angle_threshold:
            smoothed.append(path[i])
    smoothed.append(path[-1])
    return smoothed


def bspline_smooth(path, smoothing_factor=None, num_points=300):
    """
    Aggressive B-spline smoothing for A* paths.
    
    path : list of (x, y) points
    smoothing_factor : None = auto based on path length
    num_points : int, number of output points
    """
    path = np.array(path, dtype=float)

    if len(path) < 3:
        return path

    x = path[:, 0]
    y = path[:, 1]

    k = min(3, len(path) - 1)

    # Auto smoothing factor: more points → higher s
    if smoothing_factor is None:
        smoothing_factor = len(path) * 2.0  

    tck, _ = interpolate.splprep([x, y], s=smoothing_factor, k=k)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)

    return np.column_stack((x_fine, y_fine))
