import numpy as np
import matplotlib.pyplot as plt

class SOM_TSP:
    def __init__(self, cities, n_neurons=None, learning_rate=0.8, radius=1.0, radius_decay=0.995, lr_decay=0.998):
        """
        Initialize the Self-Organizing Map for Traveling Salesman Problem
        
        Parameters:
        cities (numpy.ndarray): Array of city coordinates
        n_neurons (int): Number of neurons (default: 8 * number of cities)
        learning_rate (float): Initial learning rate
        radius (float): Initial neighborhood radius
        radius_decay (float): Decay rate for neighborhood radius
        lr_decay (float): Decay rate for learning rate
        """
        self.cities = cities
        n_cities = cities.shape[0]
        
        # If not specified, use 8 times the number of cities as neurons
        self.n_neurons = n_neurons if n_neurons else n_cities * 8
        
        # Initialize weights (neuron positions) in a circle
        t = np.linspace(0, 2*np.pi, self.n_neurons, endpoint=False)
        # Circle radius is set to cover roughly the area of cities
        self.weights = np.column_stack([np.cos(t), np.sin(t)])
        # Scale weights to roughly match the scale of cities
        min_x, max_x = np.min(cities[:, 0]), np.max(cities[:, 0])
        min_y, max_y = np.min(cities[:, 1]), np.max(cities[:, 1])
        
        scale_x = (max_x - min_x) / 2.0
        scale_y = (max_y - min_y) / 2.0
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        self.weights[:, 0] = self.weights[:, 0] * scale_x + center_x
        self.weights[:, 1] = self.weights[:, 1] * scale_y + center_y
        
        # Learning parameters
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.initial_radius = radius
        self.radius = radius
        self.radius_decay = radius_decay
        self.lr_decay = lr_decay
    
    def find_bmu(self, city):
        """Find the Best Matching Unit (BMU) for a given city"""
        # Calculate Euclidean distance to all neurons
        distances = np.linalg.norm(self.weights - city, axis=1)
        # Return the index of the closest neuron
        return np.argmin(distances)
    
    def neighborhood_function(self, bmu_idx, neuron_idx):
        """Calculate neighborhood value based on distance from BMU"""
        # Calculate distance on the ring
        distance = min(
            abs(bmu_idx - neuron_idx),
            self.n_neurons - abs(bmu_idx - neuron_idx)
        )
        
        # Apply Gaussian neighborhood function
        return np.exp(-(distance**2) / (2 * self.radius**2))
    
    def update_weights(self, city, bmu_idx):
        """Update weights of the SOM based on current input city"""
        for i in range(self.n_neurons):
            # Calculate influence based on distance from BMU
            influence = self.neighborhood_function(bmu_idx, i)
            
            # Update weights only if there's some influence
            if influence > 0.05:  # Small threshold for computational efficiency
                # Move neuron towards the city
                self.weights[i] += self.learning_rate * influence * (city - self.weights[i])
    
    def train(self, n_iterations):
        """Train the SOM network for a given number of iterations"""
        for iteration in range(n_iterations):
            # Randomly select a city
            city_idx = np.random.randint(len(self.cities))
            city = self.cities[city_idx]
            
            # Find the Best Matching Unit
            bmu_idx = self.find_bmu(city)
            
            # Update network weights
            self.update_weights(city, bmu_idx)
            
            # Decay learning rate and radius
            self.radius *= self.radius_decay
            self.learning_rate *= self.lr_decay
            
            # Optional: Print progress
            if (iteration + 1) % (n_iterations // 10) == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}, "
                      f"LR: {self.learning_rate:.4f}, Radius: {self.radius:.4f}")
    
    def get_route(self):
        """
        Get the route represented by the neuron ring.
        Returns the sequence of cities in the order they would be visited.
        """
        # For each city, find the closest neuron
        city_to_neuron = np.zeros(len(self.cities), dtype=int)
        for i, city in enumerate(self.cities):
            city_to_neuron[i] = self.find_bmu(city)
        
        # Sort cities by their neuron position on the ring
        route_indices = np.argsort(city_to_neuron)
        
        return route_indices
    
    def calculate_route_distance(self, route, distance_matrix):
        """Calculate the total distance of a route using the distance matrix"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Wrap around to the start
            total_distance += distance_matrix[from_city, to_city]
        
        return total_distance
    
    def plot_som(self, iteration=None):
        """Plot the current state of the SOM and cities"""
        plt.figure(figsize=(10, 8))
        
        # Plot cities
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='r', s=100, label='Cities')
        
        # Plot neurons (the ring)
        plt.scatter(self.weights[:, 0], self.weights[:, 1], c='b', s=20, label='Neurons')
        
        # Connect neurons to show the ring
        ring = np.append(self.weights, [self.weights[0]], axis=0)  # Close the loop
        plt.plot(ring[:, 0], ring[:, 1], 'b-', linewidth=1, alpha=0.5)
        
        # Annotate cities with their indices
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i + 1), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f"SOM for TSP - {iteration if iteration else 'Initial'} iterations")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        plt.show()


# Test the SOM-TSP implementation with our problem
def main():
    # Convert our adjacency matrix to 2D coordinates using Multidimensional Scaling (MDS)
    # For simplicity, we'll use a predefined layout based on the graph
    # In a real scenario, we'd use MDS or another technique to derive 2D coordinates
    
    city_coords = {
        1: (0, 0),      # Starting city at the origin
        2: (4, 0),      # City 2 is 4 units east of city 1
        3: (4, 3),      # City 3 is 3 units north of city 2
        4: (7, 6),      # City 4 positioned northeast
        5: (10, 3),     # City 5 positioned east
        6: (6, -2),     # City 6 positioned southeast
        7: (9, 0)       # City 7 positioned east
    }
    
    # Create array of city coordinates
    cities = np.array([city_coords[i+1] for i in range(7)])
    
    # Create the adjacency matrix (distances)
    distances = np.full((7, 7), np.inf)
    
    # City 1 connections
    distances[0, 1] = distances[1, 0] = 4
    distances[0, 2] = distances[2, 0] = 8
    distances[0, 5] = distances[5, 0] = 9
    
    # City 2 connections
    distances[1, 2] = distances[2, 1] = 3
    distances[1, 3] = distances[3, 1] = 8
    distances[1, 5] = distances[5, 1] = 5
    
    # City 3 connections
    distances[2, 3] = distances[3, 2] = 9
    distances[2, 5] = distances[5, 2] = 6
    distances[2, 6] = distances[6, 2] = 9
    
    # City 4 connections
    distances[3, 4] = distances[4, 3] = 8
    distances[3, 6] = distances[6, 3] = 7
    
    # City 5 connections
    distances[4, 5] = distances[5, 4] = 2
    distances[4, 6] = distances[6, 4] = 9
    
    # City 6 connections
    distances[5, 6] = distances[6, 5] = 3
    
    # Set diagonal to 0 (distance from a city to itself)
    np.fill_diagonal(distances, 0)
    
    # Initialize and train the SOM
    som = SOM_TSP(cities, n_neurons=100, learning_rate=0.8, radius=50)
    
    # Initial state
    som.plot_som(iteration="initial")
    
    # Train the SOM
    som.train(n_iterations=10000)
    
    # Final state
    som.plot_som(iteration="final")
    
    # Get and evaluate the route
    route = som.get_route()
    
    # Adjust route to start from city 1 (index 0) if needed
    if route[0] != 0:
        city1_idx = np.where(route == 0)[0][0]
        route = np.roll(route, -city1_idx)
    
    # Adjust for 1-indexing for output
    route_1_indexed = route + 1
    
    # Calculate total distance
    total_distance = som.calculate_route_distance(route, distances)
    
    print("\nSOM-TSP Route:", " -> ".join(map(str, route_1_indexed)) + f" -> {route_1_indexed[0]}")
    print("Total Distance:", total_distance)
    
    # Print step-by-step path with distances
    print("\nStep-by-Step Path:")
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        dist = distances[from_city, to_city]
        print(f"From City {from_city + 1} to City {to_city + 1}: Distance = {dist}")

if __name__ == "__main__":
    main()