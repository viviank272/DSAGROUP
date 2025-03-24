import numpy as np
from itertools import combinations

def tsp_dynamic_programming(distances, start_city=1):
    """
    Solve the Traveling Salesman Problem using Dynamic Programming
    
    Parameters:
    distances (numpy.ndarray): Adjacency matrix with distances between cities
    start_city (int): Starting city (default: 1)
    
    Returns:
    tuple: (min_distance, optimal_path)
    """
    n = distances.shape[0] - 1  # Exclude the 0-indexed row/column if present
    
    # Dictionary to store results of subproblems
    # Key: (subset, end_city), Value: (distance, predecessor)
    memo = {}
    
    # Convert city numbering to 0-indexed for easier bit manipulation
    start = start_city - 1
    
    # Initialize base cases: from start to city
    for city in range(n):
        if city != start:
            memo[(1 << start | 1 << city, city)] = (distances[start_city][city + 1], start)
    
    # Iterate over all subset sizes
    for subset_size in range(3, n + 1):
        # Generate all subsets of size subset_size containing start city
        for subset in combinations(range(n), subset_size):
            # Skip if start city is not in subset
            if start not in subset:
                continue
            
            # Convert subset to bit representation
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            
            # For each city in subset that's not the start city
            for end in subset:
                if end == start:
                    continue
                
                # Find the best path to end city
                min_dist = float('inf')
                min_prev = -1
                
                # Try all possible previous cities
                prev_subset = bits & ~(1 << end)
                for prev in subset:
                    if prev == start or prev == end:
                        continue
                    
                    # Calculate distance through prev city
                    curr_dist = memo[(prev_subset, prev)][0] + distances[prev + 1][end + 1]
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        min_prev = prev
                
                memo[(bits, end)] = (min_dist, min_prev)
    
    # Find optimal path from final state
    bits = (1 << n) - 1  # All cities visited
    
    # Find the best end city before returning to start
    min_dist = float('inf')
    min_last = -1
    
    for end in range(n):
        if end == start:
            continue
        
        curr_dist = memo[(bits, end)][0] + distances[end + 1][start_city]
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_last = end
    
    # Reconstruct the path
    path = [start]
    curr = min_last
    bits = (1 << n) - 1
    
    # Work backwards from the last city
    while curr != -1:
        path.append(curr)
        new_bits = bits & ~(1 << curr)
        _, curr = memo.get((bits, curr), (0, -1))
        bits = new_bits
    
    # Convert back to 1-indexed cities for output
    path = [p + 1 for p in path]
    
    # Add the start city at the end to complete the tour
    path.append(start_city)
    
    return min_dist, path

# Test with our adjacency matrix
def main():
    # Initialize adjacency matrix with 7 cities (indexed 1-7 for readability)
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
    
    # Fill any missing connections with a large value to ensure they're not used
    # This is needed because our implementation assumes a complete graph
    max_distance = np.max(adjacency_matrix[adjacency_matrix != np.inf]) * 10
    adjacency_matrix[adjacency_matrix == np.inf] = max_distance
    
    # Solve TSP
    min_distance, optimal_path = tsp_dynamic_programming(adjacency_matrix)
    
    # Print results
    print("Optimal TSP Path:", " -> ".join(map(str, optimal_path)))
    print("Total Distance:", min_distance)
    
    # Print step-by-step path with distances
    print("\nStep-by-Step Path:")
    total = 0
    for i in range(len(optimal_path) - 1):
        from_city = optimal_path[i]
        to_city = optimal_path[i + 1]
        distance = adjacency_matrix[from_city, to_city]
        total += distance
        print(f"From City {from_city} to City {to_city}: Distance = {distance}")
    print(f"Total Distance: {total}")

if __name__ == "__main__":
    main()