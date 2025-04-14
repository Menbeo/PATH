from typing import List, Tuple, Dict
import numpy as np
import heapq
import matplotlib.pyplot as plt

# --- Node Utilities ---
def create_node(position: Tuple[int, int], g: float = float('inf'), parent: Dict = None) -> Dict:
    return {
        'position': position,
        'g': g,
        'parent': parent
    }

# --- Neighbor Detection ---
def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = position
    rows, cols = grid.shape
    possible_moves = [
        (x+1, y), (x-1, y),
        (x, y+1), (x, y-1)
    ]
    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols
        and grid[nx, ny] == 0
    ]

# --- Path Reconstruction ---
def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    path = []
    current = goal_node
    while current is not None:
        path.append(current['position'])
        current = current['parent']
    return path[::-1]

# --- Dijkstra's Algorithm ---
def find_path_dijkstra(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    start_node = create_node(position=start, g=0)

    open_list = [(start_node['g'], start)]
    open_dict = {start: start_node}
    closed_set = set()

    while open_list:
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        if current_pos == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_pos)

        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            if neighbor_pos in closed_set:
                continue

            tentative_g = current_node['g'] + 1

            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['g'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['parent'] = current_node

    return []

# --- Grid to Latitude/Longitude Conversion ---
def grid_to_latlon(position: Tuple[int, int], grid_shape: Tuple[int, int],
                   top_left: Tuple[float, float], bottom_right: Tuple[float, float]) -> Tuple[float, float]:
    rows, cols = grid_shape
    row, col = position

    lat1, lon1 = top_left
    lat2, lon2 = bottom_right

    lat = lat1 + (lat2 - lat1) * (row / (rows - 1))
    lon = lon1 + (lon2 - lon1) * (col / (cols - 1))

    return (lat, lon)

def convert_path_to_latlon(path: List[Tuple[int, int]], grid_shape: Tuple[int, int],
                           top_left: Tuple[float, float], bottom_right: Tuple[float, float]) -> List[Tuple[float, float]]:
    return [grid_to_latlon(pos, grid_shape, top_left, bottom_right) for pos in path]

# --- Visualization ---
def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]]):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("Dijkstra Pathfinding Result")
    plt.show()

# --- RUN TEST ---
if __name__ == "__main__":
    # Grid and obstacle definition
    grid = np.zeros((20, 20))
    grid[5:15, 5] = 1
    grid[5, 5:15] = 1

    start_pos = (0, 0)
    goal_pos = (18, 18)

    # Latitude and longitude bounds
    top_left_latlon = (0,0)       
    bottom_right_latlon = (10.7900, 106.7100)   

    # Find path
    path = find_path_dijkstra(grid, start_pos, goal_pos)
    if path:
        print(f"Path found with {len(path)} steps!")
        visualize_path(grid, path)

        latlon_path = convert_path_to_latlon(path, grid.shape, top_left_latlon, bottom_right_latlon)
        print("Latitude/Longitude path:")
        for latlon in latlon_path:
            print(latlon)
    else:
        print("No path found!")
