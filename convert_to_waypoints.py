import numpy as np
def export_waypoints(lat_lon_path: list[tuple[float,float]], filename  = "Dijkstra.waypoints", default_altitude=2000): #Altitude of each waypoints is 2000
    num_original_waypoints = len(lat_lon_path)
    num_export_waypoints = min(num_original_waypoints, 12) #Limit to 12 waypoints
    if num_export_waypoints > 1:
        indices_to_export = np.linspace(0, num_original_waypoints - 1, num_export_waypoints, dtype=int)
    else: # If only one waypoint, take the first one
        indices_to_export = [0] if num_original_waypoints > 0 else []
    # Filter the path to include only the selected waypoints
    sampled_lat_lon_path =[lat_lon_path[i] for i in indices_to_export]
    with open(filename, 'w') as f:
        f.write("QGC WPL 110 \n")
        for i, (lat,lon) in enumerate(sampled_lat_lon_path):
            waypoint_index = i
            is_current = 1 if i == 0 else 0
            autocontinue = 3
            command = 16

            #param
            param1 = 0.0
            param2 = 0.0 
            param3 = 0.0
            param4 = 0.0

            latitude = lat
            longitude = lon
            altitude = default_altitude
            frame = 1
            line = (
                f"{waypoint_index}\t{is_current}\t{autocontinue}\t{command}\t"
                f"{param1:.8f}\t{param2:.8f}\t{param3:.8f}\t{param4:.8f}\t"
                f"{latitude:.8f}\t{longitude:.8f}\t{altitude:.2f}\t{frame}\n"
            )
            f.write(line)
        print(f"Have exported to filename")