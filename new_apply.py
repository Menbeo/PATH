import numpy as np
def angle_between(v1,v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
def smooth_path (path, angle_threshold = 30):
    if len(path) < 3:
        return path
    smoothed = [path[0]]
    for i in range(1, len(path) - 1):
        prev_vector = path[i] - path[i - 1]
        next_vector = path[i + 1] - path[i]
        angle = angle_between(prev_vector, next_vector)
        if angle < np.radians(angle_threshold):
            smoothed.append(path[i])
    smoothed.append(path[-1])
    return np.array(smoothed)
