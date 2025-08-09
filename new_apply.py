import numpy as np
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
