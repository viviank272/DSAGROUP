import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_cities = 7
num_neurons = 50  # More neurons than cities to allow flexibility
num_iterations = 2000
learning_rate = 0.8
initial_radius = num_neurons // 4

# Graph coordinates (for visualization only)
city_coords = np.array([
    [0, 0], [12, 0], [10, 8], [8, 12], [3, 11], [6, 6], [12, 9]
])

# Initialize SOM ring (neurons randomly placed)
neurons = np.random.rand(num_neurons, 2) * np.max(city_coords)

# Gaussian neighborhood function
def neighborhood_function(winner_idx, neuron_idx, radius):
    distance = min(abs(neuron_idx - winner_idx), num_neurons - abs(neuron_idx - winner_idx))
    return np.exp(-distance**2 / (2 * (radius**2)))

# SOM Training Loop
for iteration in range(num_iterations):
    # Decay learning rate and radius
    lr = learning_rate * (1 - iteration / num_iterations)
    radius = initial_radius * (1 - iteration / num_iterations)

    # Select a random city
    city = city_coords[np.random.randint(0, num_cities)]

    # Find the closest neuron (winner)
    distances = np.linalg.norm(neurons - city, axis=1)
    winner_idx = np.argmin(distances)

    # Update neurons within the neighborhood
    for neuron_idx in range(num_neurons):
        influence = neighborhood_function(winner_idx, neuron_idx, radius)
        neurons[neuron_idx] += lr * influence * (city - neurons[neuron_idx])

# Plot results
plt.scatter(city_coords[:, 0], city_coords[:, 1], color='red', label="Cities")
plt.scatter(neurons[:, 0], neurons[:, 1], color='red', label="Neurons")
plt.plot(neurons[:, 0], neurons[:, 1], linestyle='-', color='blue', alpha=0.5)
plt.legend()
plt.title("SOM-TSP Path Approximation")
plt.show()
