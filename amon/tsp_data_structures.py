# Representing the TSP graph using an adjacency matrix
import numpy as np

# Initialize adjacency matrix with 7 cities (indexed 1-7 for readability)
# We'll use infinity (np.inf) for non-connected cities
num_cities = 7
adjacency_matrix = np.full((num_cities + 1, num_cities + 1), np.inf)

# Fill in the distances from the graph
# City 1 connections
adjacency_matrix[1, 2] = adjacency_matrix[2, 1] = 4
adjacency_matrix[1, 3] = adjacency_matrix[3, 1] = 8
adjacency_matrix[1, 6] = adjacency_matrix[6, 1] = 9

# City 2 connections
adjacency_matrix[2, 3] = adjacency_matrix[3, 2] = 3
adjacency_matrix[2, 4] = adjacency_matrix[4, 2] = 8
adjacency_matrix[2, 6] = adjacency_matrix[6, 2] = 5

# City 3 connections
adjacency_matrix[3, 4] = adjacency_matrix[4, 3] = 9
adjacency_matrix[3, 6] = adjacency_matrix[6, 3] = 6
adjacency_matrix[3, 7] = adjacency_matrix[7, 3] = 9

# City 4 connections
adjacency_matrix[4, 5] = adjacency_matrix[5, 4] = 8
adjacency_matrix[4, 7] = adjacency_matrix[7, 4] = 7

# City 5 connections
adjacency_matrix[5, 6] = adjacency_matrix[6, 5] = 2
adjacency_matrix[5, 7] = adjacency_matrix[7, 5] = 9

# City 6 connections
adjacency_matrix[6, 7] = adjacency_matrix[7, 6] = 3

# Set diagonal to 0 (distance from a city to itself)
np.fill_diagonal(adjacency_matrix, 0)

# Alternative: Adjacency list representation
adjacency_list = {
    1: {2: 4, 3: 8, 6: 9},
    2: {1: 4, 3: 3, 4: 8, 6: 5},
    3: {1: 8, 2: 3, 4: 9, 6: 6, 7: 9},
    4: {2: 8, 3: 9, 5: 8, 7: 7},
    5: {4: 8, 6: 2, 7: 9},
    6: {1: 9, 2: 5, 3: 6, 5: 2, 7: 3},
    7: {3: 9, 4: 7, 5: 9, 6: 3}
}

# Print the matrix to verify (excluding the 0-indexed row/column)
print("Adjacency Matrix:")
print(adjacency_matrix[1:, 1:])
print("\nAdjacency List:")
for city, neighbors in adjacency_list.items():
    print(f"City {city}: {neighbors}")