from gridmaplv3 import create_grid_map,grid_map, default_goal, default_start
import numpy as np 
import matplotlib.pyplot as plt

def animate_path(grid,path,delay=0.01):
    plt.figure(figsize=(10,10))
    plt.imshow(grid, cmap='gray_r', origin='upper')
    plt.plot(default_start[1], default_goal[0], 'go', marketsize=10, label='Start')
    plt.plot(default_goal[1],default_start[0],'ro', marketsize=10, label='Goal')
    plt.legend()
    for i in range(1,len(path)):
        x0,y0 = path[i-1]
        x1,y1 = path[i]
        plt.plot([y0, y1], [x0,x1], 'b-', linewidth=2)
        plt.pause(delay)
    plt.grid(True)
    plt.show()
def dijkstra(start, goal, grid):
    rows, cols = grid.shape
    visited = set()
    distance = {start: 0}
    previous = {}
    queue = [start]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        current = min(queue, key=lambda node: distance.get(node, float('inf')))
        queue.remove(current)
        if current == goal:
            break
        visited.add(current)
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 1 or neighbor in visited:
                    continue
                new_cost = distance[current] + 1
                if new_cost < distance.get(neighbor, float('inf')):
                    distance[neighbor] = new_cost
                    previous[neighbor] = current
                    if neighbor not in queue:
                        queue.append(neighbor)
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = previous.get(node)
        if node is None:
            return [] 
    path.append(start)
    path.reverse()
    return path

if __name__ == "__main__":
    grid = grid_map()
    path = dijkstra(default_start, default_goal, grid)
    animate_path(grid,path)
    create_grid_map(grid, path)