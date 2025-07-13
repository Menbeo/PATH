#import gridmap
from gridmap import create_grid_map, grid_map, default_goal,default_start,convert_grid_to_lat_lon
import numpy as np 
import matplotlib.pyplot as plt
import heapq
import math 

#==== DIJKSTRA'S PATH ====
def Dijkstra(grid, start, goal):
    #declare rows and cols 
    rows,cols = grid.shape
    visited = set()
    distance = {start:0}
    pq = [(0, start)]

    while pq:
        cost, current = heapq.heappop(pq)
        

        
