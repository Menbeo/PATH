import numpy as np 
from scipy.ndimage import distance_transform_edt
def angle_between_calculate(v1,v2):
    #Calculate angle between 
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2) 
    pi = 3.14
    return np.arccos(np.clip(np.dot(v1,v2), -1.0, 1.0)) * 180/pi

def turn_constraint(path, grid, min_angle = 30, max_angle = 45, min_dist = 2, max_dist = 10):
    #constrain corner adapt base on distance to obstacles. 
    # Near obstacle -> tighter turns, farer -> smoother turns.
    
    if len(path) < 3: 
        return path 
    
    #compute distance 
    obstacle_grid = (grid == 1) 
    distance_map = distance_transform_edt(~obstacle_grid)
    
    def angle_threshold(point):
        x,y = point
        d  = distance_map[int(x), int(y)]
        d  = np.clip(d, min_dist, max_dist) 
        scale = (d - min_dist) / (max_dist - min_dist)
        return min_angle + (max_angle - min_angle) * scale
    
    smoothed = [path[0]]
    for i in range(1, len(path)-1):
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        angle = angle_between_calculate(v1, v2)
        dynamic_threshold = angle_threshold(path[i])
        if angle > dynamic_threshold:
            smoothed.append(path[i])
    smoothed.append(path[-1])
    return smoothed