
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.sparse.csgraph import shortest_path

plt.switch_backend('agg')  # Suppress graphical output issues

class SOM_TSP:
    def __init__(self, cities, distance_matrix, n_neurons=None, learning_rate=0.8, radius=None, radius_decay=0.999, lr_decay=0.999):
        self.cities = cities  # 2D coordinates for training
        self.distance_matrix = distance_matrix  # Graph distances for evaluation
        n_cities = cities.shape[0]
        self.n_neurons = n_neurons if n_neurons else n_cities * 2  # 14 neurons
        self.weights = self._initialize_weights()
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.initial_radius = radius if radius else self.n_neurons / 2  # 7.0
        self.radius = self.initial_radius
        self.radius_decay = radius_decay
        self.lr_decay = lr_decay

    def _initialize_weights(self):
        # Initialize neurons near city positions with small perturbations
        indices = np.random.choice(len(self.cities), self.n_neurons, replace=True)
        weights = self.cities[indices] + np.random.normal(0, 0.1, (self.n_neurons, 2))
        return weights

    def find_bmu(self, city):
        distances = np.linalg.norm(self.weights - city, axis=1)
        return np.argmin(distances)

    def neighborhood_function(self, bmu_idx, neuron_idx):
        distance = min(abs(bmu_idx - neuron_idx), self.n_neurons - abs(bmu_idx - neuron_idx))
        return np.exp(-(distance**2) / (2 * self.radius**2))

    def update_weights(self, city, bmu_idx):
        for i in range(self.n_neurons):
            influence = self.neighborhood_function(bmu_idx, i)
            if influence > 0.01:
                self.weights[i] += self.learning_rate * influence * (city - self.weights[i])

    def train(self, n_iterations):
        for iteration in range(n_iterations):
            city_idx = np.random.randint(len(self.cities))
            city = self.cities[city_idx]
            bmu_idx = self.find_bmu(city)
            self.update_weights(city, bmu_idx)
            self.radius = self.initial_radius * (self.radius_decay ** iteration)
            self.learning_rate = self.initial_learning_rate * (self.lr_decay ** iteration)
            if iteration % (n_iterations // 10) == 0:
                print(f"Iteration {iteration}, LR: {self.learning_rate:.4f}, Radius: {self.radius:.4f}")

    def get_route(self):
        city_to_neuron = []
        used_neurons = set()
        for city in self.cities:
            distances = np.linalg.norm(self.weights - city, axis=1)
            bmu = np.argmin(distances)
            while bmu in used_neurons:
                distances[bmu] = np.inf
                bmu = np.argmin(distances)
            used_neurons.add(bmu)
            city_to_neuron.append(bmu)
        route = np.argsort(city_to_neuron)
        return route

    def calculate_route_distance(self, route):
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            dist = self.distance_matrix[from_city, to_city]
            if np.isinf(dist):
                return np.inf  # Invalid route
            total_distance += dist
        return total_distance

    def plot_som(self, iteration=None):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='r', s=100, label='Cities')
        plt.scatter(self.weights[:, 0], self.weights[:, 1], c='b', s=20, label='Neurons')
        ring = np.append(self.weights, [self.weights[0]], axis=0)
        plt.plot(ring[:, 0], ring[:, 1], 'b-', alpha=0.5)
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i + 1), (x, y), xytext=(5, 5), textcoords='offset points')
        plt.title(f"SOM-TSP - {iteration if iteration else 'Initial'}")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(f'som_tsp_{iteration}.png')
        plt.close()

def main():
    # Define the distance matrix (graph)
    distances = np.full((7, 7), np.inf)
    distances[0, 1] = distances[1, 0] = 12  # 1-2
    distances[0, 2] = distances[2, 0] = 10  # 1-3
    distances[0, 6] = distances[6, 0] = 12  # 1-7
    distances[1, 2] = distances[2, 1] = 8   # 2-3
    distances[1, 3] = distances[3, 1] = 12  # 2-4
    distances[2, 3] = distances[3, 2] = 11  # 3-4
    distances[2, 5] = distances[5, 2] = 0   # 3-6
    distances[2, 6] = distances[6, 2] = 12   # 3-7
    distances[3, 4] = distances[4, 3] = 11  # 4-5
    distances[3, 5] = distances[5, 3] = 10  # 4-6
    distances[4, 5] = distances[5, 4] = 6   # 5-6
    distances[4, 6] = distances[6, 4] = 7   # 5-7
    distances[5, 6] = distances[6, 5] = 9   # 6-7
    np.fill_diagonal(distances, 0)

    # Compute shortest paths for MDS
    shortest_distances = shortest_path(distances, directed=False)
    shortest_distances[np.isinf(shortest_distances)] = 0  # Replace inf with 0 for MDS

    # Use MDS to generate 2D coordinates
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress='auto')
    cities = mds.fit_transform(shortest_distances)

    # Initialize SOM
    som = SOM_TSP(cities, distances, n_neurons=14)
    som.plot_som(iteration="initial")

    # Train SOM
    som.train(n_iterations=100000)
    som.plot_som(iteration="final")

    # Get and adjust route
    route = som.get_route()
    if route[0] != 0:
        city1_idx = np.where(route == 0)[0][0]
        route = np.roll(route, -city1_idx)

    # Calculate total distance
    total_distance = som.calculate_route_distance(route)
    route_1_indexed = route + 1

    # Print results
    print("\nRoute:", " -> ".join(map(str, route_1_indexed)) + f" -> {route_1_indexed[0]}")
    print("Total Distance:", total_distance)

    # Step-by-step path
    print("\nStep-by-Step Path:")
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        dist = distances[from_city, to_city]
        print(f"From City {from_city + 1} to City {to_city + 1}: Distance = {dist if not np.isinf(dist) else 'inf'}")

if __name__ == "__main__":
    main()