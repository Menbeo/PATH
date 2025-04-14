import heapq
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)  # You can change the logging level to ERROR, WARNING, etc.

def dijkstra_path(graph, start, goal):
    distance = {node: float('infinity') for node in graph}
    distance[start] = 0
    previous = {node: None for node in graph}
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == goal:
            break  # Stop when goal is reached

        if current_distance > distance[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            new_distance = current_distance + weight
            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    return path, distance[goal]


