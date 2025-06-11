from math import radians, sin, cos, sqrt, atan2, degrees
import heapq

# === CONFIG ===
ISLAND_CENTER = (10.72662, 106.71780)
ISLAND_RADIUS = 0.2  # 200m in km

# === HELPER FUNCTIONS ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def bearing(lat1, lon1, lat2, lon2):
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(radians(lat2))
    y = cos(radians(lat1)) * sin(radians(lat2)) - sin(radians(lat1)) * cos(radians(lat2)) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360

def yaw_pwm(b):
    delta = ((b + 180) % 360) - 180
    return max(1100, min(1900, int(1500 + delta * (400 / 180))))

def initialize_rc():
    print('Start Script')
    for ch in range(1, 9): Script.SendRC(ch, 1500, False)
    Script.SendRC(3, int(Script.GetParam("RC3_MIN")), True)
    Script.Sleep(5000)
    while cs.lat == 0 or cs.lng == 0: Script.Sleep(1000)

def is_near_island(lat, lon):
    return haversine(lat, lon, *ISLAND_CENTER) < ISLAND_RADIUS

# === MOVE WITH FORCED AVOIDANCE ===
def move(path):
    for tgt in path[1:]:
        lat, lon = waypoints[tgt]
        while haversine(cs.lat, cs.lng, lat, lon) > 0.01:

            # Always check for obstacles (near island)
            if is_near_island(cs.lat, cs.lng):
                print("Obstacle detected: Near island! Turning left and moving straight to avoid...")

                # Forced avoidance: Turn left for 3 seconds
                for _ in range(3):
                    Script.SendRC(3, 1500, False)
                    Script.SendRC(4, 1100, True)  # Hard left yaw
                    Script.Sleep(1000)

                # Move forward for ~200 meters (duration-based)
                for _ in range(15):  # adjust for duration based on speed
                    Script.SendRC(3, 1600, False)  # Slightly increase throttle for moving
                    Script.SendRC(4, 1500, True)   # Keep yaw centered
                    Script.Sleep(1000)

                continue  # Re-check position and continue toward the destination

            # Otherwise, follow normal path
            b = bearing(cs.lat, cs.lng, lat, lon)
            Script.SendRC(3, 1550, False)
            Script.SendRC(4, yaw_pwm(b), True)
            Script.Sleep(1000)
    Script.SendRC(3, 1500, True)
    print("Arrived at destination!")

# === A* PATH ===
def a_star(start, goal):
    heap, g = [(0, start, [])], {start: 0}
    while heap:
        _, current, path = heapq.heappop(heap)
        path += [current]
        if current == goal:
            return path
        for neighbor in [goal] if current == start else [start]:
            dist = g[current] + haversine(*waypoints[current], *waypoints[neighbor])
            if neighbor not in g or dist < g[neighbor]:
                g[neighbor] = dist
                h = haversine(*waypoints[neighbor], *waypoints[goal])
                heapq.heappush(heap, (dist + h, neighbor, path))
    return []

def return_home(path):
    print("Returning home...")
    move(path[::-1])

# === MAIN EXECUTION ===
initialize_rc()

waypoints = {
    "home": (cs.lat, cs.lng),
    "goal": (10.7258451, 106.7197323)
}
print(f"Home set at ({cs.lat:.7f}, {cs.lng:.7f})")

path = a_star("home", "goal")
if path:
    move(path)  # Move towards the goal
    return_home(path)  # Return home after reaching the goal
else:
    print("No path found")