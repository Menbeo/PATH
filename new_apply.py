import numpy as np
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi





def bspline_smooth(waypoints, grid, inflated_grid=None, num_points=200, degree=3, smoothing_factor=None):
    if len(waypoints) <= degree: return waypoints
    if smoothing_factor is None: smoothing_factor = len(waypoints)

    x, y = zip(*waypoints)
    tck, _ = interpolate.splprep([x, y], s=smoothing_factor, k=degree)
    unew = np.linspace(0, 1, num_points)
    pts = interpolate.splev(unew, tck)

    mask = inflated_grid if inflated_grid is not None else grid
    dist_map = distance_transform_edt(mask == 0)
    gy, gx = np.gradient(dist_map)

    path = []
    h, w = grid.shape
    for px, py in zip(*pts):
        c, r = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        if mask[r, c]:
            d, dx, dy = dist_map[r, c], gx[r, c], gy[r, c]
            if dx or dy:
                dir = np.array([-dx, -dy]) / np.linalg.norm([dx, dy])
                px += dir[0] * d * 1.1
                py += dir[1] * d * 1.1
        path.append((px, py))
    return path

