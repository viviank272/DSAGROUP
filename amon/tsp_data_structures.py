# Representing the TSP graph using an adjacency matrix
import numpy as np

# Initialize adjacency matrix with 7 cities (indexed 1-7 for readability)
# We'll use infinity (np.inf) for non-connected cities
num_cities = 7
adjacency_matrix = np.full((num_cities + 1 , num_cities + 1), np.inf)

# adjacency_matrix = [[0 for _ in range(7)] for _ in range(7)]
print(adjacency_matrix)
# Fill in the distances from the graph

# City 1 connections
adjacency_matrix[1, 2] = adjacency_matrix[2, 1] = 12
adjacency_matrix[1, 3] = adjacency_matrix[3, 1] = 10
adjacency_matrix[1, 7] = adjacency_matrix[7, 1] = 12

# City 2 connections
adjacency_matrix[2, 3] = adjacency_matrix[3, 2] = 8
adjacency_matrix[2, 4] = adjacency_matrix[4, 2] = 12

# City 3 connections
adjacency_matrix[3, 4] = adjacency_matrix[4, 3] = 11
adjacency_matrix[3, 5] = adjacency_matrix[5, 3] = 3


adjacency_matrix[3, 7] = adjacency_matrix[7, 3] = 9


# City 4 connections
adjacency_matrix[4, 5] = adjacency_matrix[5, 4] = 11
adjacency_matrix[4, 6] = adjacency_matrix[6, 4] = 10

    # City 5 connections
adjacency_matrix[5, 7] = adjacency_matrix[7, 5] = 7
adjacency_matrix[5, 6] = adjacency_matrix[6, 5] = 6

# City 6 connections
adjacency_matrix[7, 6] = adjacency_matrix[6, 7] = 9

# Set diagonal to 0 (distance from a city to itself)
np.fill_diagonal(adjacency_matrix, 0)

# Alternative: Adjacency list representation
adjacency_list = {

    1: {(2, 12), (3, 10), (7, 12)},
    2: {(1, 12), (3, 8), (4, 12)},
    3: {(1, 10), (2, 8), (4, 11), (5, 3), (7, 9)},
    4: {(2, 12), (3, 11), (5, 11), (6, 10)},
    5: {(3, 3), (4, 11), (6, 6), (7, 7)},
    6: {(4, 10), (5, 6), (7, 9)},
    7: {(1, 12), (3, 9), (5, 7), (6, 9)},
}

# Print the matrix to verify (excluding the 0-indexed row/column)
# print("Adjacency Matrix:")
# print(adjacency_matrix[1:, 1:])
# print("\nAdjacency List:")
# for city, neighbors in adjacency_list.items():
#     print(f"City {city}: {neighbors}")

print(adjacency_matrix)