from gridmap import create_grid_map,grid_map
from gridmap import default_goal, default_start
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
    create_grid_map(grid, path)