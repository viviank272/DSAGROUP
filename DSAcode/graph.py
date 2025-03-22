import numpy as np
#Displays the ajacency matrix
graph =np.array([
    [0, 12, 10, np.inf, np.inf, np.inf, 12],  # City 1
    [12, 0, 8, 12, np.inf, np.inf, np.inf],   # City 2
    [10, 8, 0, 11, 3, np.inf, 9],             # City 3
    [np.inf, 12, 11, 0, 11, 10, np.inf],      # City 4
    [np.inf, np.inf, 3, 11, 0, 6, 7],         # City 5
    [np.inf, np.inf, np.inf, 10, 6, 0, 9],    # City 6
    [12, np.inf, 9, np.inf, 7, 9, 0]          # City 7
    
]) 

def print_graph():
    for row in graph:
        print(row)

if __name__ =="__main__":
    print_graph()        
