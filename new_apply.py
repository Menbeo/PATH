import numpy as np
from scipy import interpolate
def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

def turn_constraint(path, angle_threshold=10):
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

def bspline_smooth(path, smoothing_factor=0, num_points=100):
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]
    
    # Parameterize the path
    t = np.linspace(0, 1, len(path))
    
    # B-spline representation
    tck, _ = interpolate.splprep([x, y], s=smoothing_factor)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)
    
    return np.column_stack((x_fine, y_fine))
