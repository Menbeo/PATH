import numpy as np
import matplotlib.pyplot as plt
import math
import random
from gridmap import grid_map, create_grid_map, default_start, default_goal
from gridmap import compute_neighborhood_layers, convert_grid_to_lat_lon
from convert_to_waypoints import export_waypoints

# PSO Parameters
POP_SIZE = 50
MAX_ITER = 200
NUM_WAYPOINTS = 12
INERTIA = 0.5
C1 = 1.5
C2 = 1.5
V_MAX = 5
# PENALTY = 1e6 # This parameter is not used, can be removed

class Particle:
    def __init__(self, position, velocity, grid, inflation):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_score = float('inf')
        self.grid = grid
        self.inflation = inflation
        # Initialize best score with current position
        self.best_score = self.fitness_function(position)

    def fitness_function(self, waypoints_flattened):
        """Evaluate the fitness of a particle's solution"""
        # Ensure waypoints are within grid boundaries before reshaping
        # This clip is already done in particle_swarm_optimization, but good to ensure here too if this method is called independently
        waypoints_clipped = np.clip(waypoints_flattened, 0, self.grid.shape[0]-1)
        waypoints = [default_start] + list(waypoints_clipped.reshape(-1, 2)) + [default_goal]
        
        # 1. Path length cost
        path_length = sum(np.linalg.norm(np.array(waypoints[i])-np.array(waypoints[i-1])) 
                         for i in range(1, len(waypoints)))
        
        # 2. Collision penalty
        if not self.is_collision_free(waypoints):
            return float('inf')
        
        # 3. Proximity penalty
        proximity_cost = 0
        for wp in waypoints:
            x, y = int(round(wp[0])), int(round(wp[1]))
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                if self.inflation[x, y] == 1:
                    proximity_cost += 500 # High penalty for being in inflated zone
                elif self.inflation[x, y] == 0:
                    # Penalty for being close to obstacles even if not in inflated zone
                    for di in range(-3, 4): # Check 7x7 neighborhood
                        for dj in range(-3, 4):
                            xi, yj = x+di, y+dj
                            if (0 <= xi < self.grid.shape[0] and 0 <= yj < self.grid.shape[1] 
                                and self.grid[xi, yj] == 1): # Check for actual obstacles
                                # Inverse square law for distance, add a small epsilon to avoid division by zero
                                distance_sq = di**2 + dj**2
                                if distance_sq == 0: # If it's the obstacle itself (should be caught by inflation[x,y]==1)
                                    proximity_cost += 1000 # Very high penalty
                                else:
                                    proximity_cost += 50.0 / distance_sq
        
        return path_length + 10 * proximity_cost

    def is_collision_free(self, path, margin=0):
        """Check if path avoids obstacles using Bresenham's line algorithm with an optional margin."""
        for i in range(1, len(path)):
            x0, y0 = int(round(path[i-1][0])), int(round(path[i-1][1]))
            x1, y1 = int(round(path[i][0])), int(round(path[i][1]))
            
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0) # Using negative dy for Bresenham's initial error setup
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            
            current_x, current_y = x0, y0 # Initialize current point for Bresenham's loop
            
            while True:
                # Check current cell and its margin for collision
                for di in range(-margin, margin + 1):
                    for dj in range(-margin, margin + 1):
                        check_x, check_y = current_x + di, current_y + dj
                        if (0 <= check_x < self.grid.shape[0] and 0 <= check_y < self.grid.shape[1]):
                            if self.grid[check_x, check_y] == 1 or self.inflation[check_x, check_y] == 1:
                                return False
                
                if current_x == x1 and current_y == y1:
                    break # Reached the end point
                
                e2 = 2 * err
                if e2 >= dy: # Move in x direction
                    err += dy
                    current_x += sx
                if e2 <= dx: # Move in y direction
                    err += dx
                    current_y += sy
        return True

class Swarm:
    def __init__(self, grid, inflation, pop_size=POP_SIZE):
        self.grid = grid
        self.inflation = inflation
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        rows, cols = grid.shape
        for i in range(pop_size):
            # Randomly initialize waypoints within the grid boundaries
            position = np.random.randint(0, rows, NUM_WAYPOINTS * 2) # Waypoints are (x,y) pairs, so *2
            velocity = np.random.uniform(-V_MAX, V_MAX, NUM_WAYPOINTS * 2)
            
            particle = Particle(position, velocity, grid, inflation)
            self.particles.append(particle)
            
            # Initialize global best with the first particle, or if a better particle is found
            if i == 0 or particle.best_score < self.global_best_score:
                self.global_best_score = particle.best_score
                self.global_best_position = particle.best_position.copy() # Ensure it's a copy

def particle_swarm_optimization(grid, inflation):
    """Main PSO optimization routine"""
    swarm = Swarm(grid, inflation)
    rows, cols = grid.shape
    
    for iteration in range(MAX_ITER):
        for particle in swarm.particles:
            # Update velocity
            r1 = np.random.rand(NUM_WAYPOINTS * 2) # Random numbers for cognitive component
            r2 = np.random.rand(NUM_WAYPOINTS * 2) # Random numbers for social component
            
            cognitive = C1 * r1 * (particle.best_position - particle.position)
            social = C2 * r2 * (swarm.global_best_position - particle.position)
            particle.velocity = INERTIA * particle.velocity + cognitive + social
            
            # Clip velocity to V_MAX
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)
            
            # Update position
            particle.position = particle.position + particle.velocity
            
            # Clip position to stay within grid boundaries and convert to int
            particle.position = np.clip(particle.position, 0, rows-1).astype(int)
            
            # Evaluate fitness
            current_score = particle.fitness_function(particle.position)
            
            # Update personal best
            if current_score < particle.best_score:
                particle.best_score = current_score
                particle.best_position = particle.position.copy()
                
                # Update global best
                if current_score < swarm.global_best_score:
                    swarm.global_best_score = current_score
                    swarm.global_best_position = particle.position.copy()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d}, Best Score: {swarm.global_best_score:.2f}")
    
    # After MAX_ITER, return the best path found
    return swarm.global_best_position

def main():
    """Main function to run PSO path planning"""
    for map_id in range(1, 5):
        print(f"\nProcessing Map {map_id}...")
        grid = grid_map(map_id=map_id)
        # Compute inflation layer for collision and proximity checking
        inflation = compute_neighborhood_layers(grid) 
        
        best_waypoints_flattened = particle_swarm_optimization(grid, inflation)
        
        # Reshape the flattened waypoints and add start/goal
        # Ensure the waypoints are still within bounds after optimization
        if best_waypoints_flattened is None:
            print(f"No valid path found for Map {map_id}. Skipping waypoint export and visualization.")
            continue

        best_waypoints_clipped = np.clip(best_waypoints_flattened, 0, grid.shape[0]-1)
        waypoints_grid_coords = [default_start] + list(best_waypoints_clipped.reshape(-1, 2)) + [default_goal]
        
        # Convert grid coordinates to (int, int) tuples for plotting and clarity
        path_for_plot = [(int(p[0]), int(p[1])) for p in waypoints_grid_coords]
        
        # Convert grid coordinates to latitude and longitude for export
        # Note: convert_grid_to_lat_lon expects (y_grid, x_grid) for (latitude, longitude) conversion
        # but your path is (x_grid, y_grid). So we swap them here.
        # This aligns with how `convert_grid_to_lat_lon` is designed (y_grid for lat, x_grid for lon)
        lat_lon_path = [
            convert_grid_to_lat_lon(point_y, point_x) for point_x, point_y in path_for_plot
        ]
        
        filename = f"PSO_Map{map_id}.waypoints"
        export_waypoints(lat_lon_path, filename=filename, default_altitude=100) # Use 100 as altitude
        
        create_grid_map(grid, path_for_plot) # Pass the path in grid coordinates
        print(f"Map {map_id} completed. Waypoints saved to {filename}")

if __name__ == "__main__":
    main()