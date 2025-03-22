#Graph representation for the adjacency matrix
graph =[
    
    [0,  12, 10,  0,   0,  0,  12],#City 1
    [12, 0,  8,   12,  0,  0,  0],#city 2
    [10, 8,  0,   11,  3,  0 , 9],#City 3
    [0,  12, 11,  0,   11, 10, 0],#City 4
    [0,  0,  3,   11,  0,  6,  7],#City 5
    [0,  0,  0,   10,  6,  0,  9],#City 6
    [12, 0,  9,   0,   7,  9,  0],#City 7
]
 
#Why adjacency matrix
#Its efficient for storing and retreiving distances between cities
# It is suitable for dense graphs like this one ,where most cities are connected
 # The TSP objective is to find the shortest possible route that visits each city exctly once and returns to the starting city.
 # Assumptions made 
 # the graph is undirected(traveling from city i to j is the same as traveling from city j to city i)
 # Each edge has a positive weight distance


 #CLASSICAL TSP SOLUTION
   #Nearest Neighbour algorithm.Here you start from the given city and repeatedly visit the nearest unvisted city and return to the start city once all the cities have been visited.
def tsp_nearest_neighbour(graph, start=0):
    n = len(graph)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    total_cost = 0

    for _ in range(n-1):
        last = tour[-1]
        nearest_city = None
        min_dist = float('inf')

#Find the nearest unvisited city
        for city in range(n):
            if not visited[city] and graph[last][city] < min_dist and graph[last][city] != 0:
                nearest_city = city
                min_dist = graph[last][city]

#If no valid nearest city is found ,break out of the loop
        if nearest_city is None:
                    print("Error: No valid nearest city found from node", last)
                    break   
                    
 #Add the nearest city to the tour                   
        tour.append(nearest_city)
        visited[nearest_city] = True
        total_cost += min_dist

    # Return to the starting city
    if nearest_city is not None:
        total_cost += graph[tour[-1]][start]
        tour.append(start)

    return tour, total_cost

# Run the algorithm
tour, cost = tsp_nearest_neighbour(graph)
print("Optimal Tour:", tour)
print("Total Distance:", cost)
#Output shows you the sequence of cities visited and the total cost of the route found by the Nearest Neighbour Algorithm

