V = 4
answer = []
min_path = []

# Function to find the minimum weight Hamiltonian Cycle and its path
def tsp(graph, v, currPos, n, count, cost, path):

    # If last node is reached and it has a link to the starting node
    if count == n and graph[currPos][0]:
        total_cost = cost + graph[currPos][0]
        answer.append(total_cost)

        # Check if this path's cost is the minimum encountered so far
        if total_cost == min(answer):
            global min_path
            min_path = path[:] + [0]  # Include 0 to complete the cycle

        print(path)
        return

    # BACKTRACKING STEP
    # Loop to traverse the adjacency list of currPos node
    for i in range(n):
        if not v[i] and graph[currPos][i]:
            
            # Mark as visited
            v[i] = True
            # Recur with updated path and cost
            tsp(graph, v, i, n, count + 1, cost + graph[currPos][i], path + [i])
            
            # Backtrack: Mark ith node as unvisited
            v[i] = False

# Driver code
if __name__ == '__main__':
    n = 4
    graph = [[0, 10, 15, 20],
             [10, 0, 35, 25],
             [15, 35, 0, 30],
             [20, 25, 30, 0]]

    # Boolean array to check if a node has been visited or not
    v = [False] * n
    
    # Mark 0th node as visited and start from it
    v[0] = True
    
    # Initial path starting from node 0
    initial_path = [0]
    
    # Find the minimum weight Hamiltonian Cycle
    tsp(graph, v, 0, n, 1, 0, initial_path)

    # Display the minimum cost and path
    min_cost = min(answer)
    print("Minimum cost:", min_cost)
    print("Path:", min_path)
