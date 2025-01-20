n = 7  # number of nodes in the graph (1-based)

# dist[i][j] represents the shortest distance from node i to node j for i < j
dist = [
    [0, 10, 15, 20, 25, 30, 35],  # distances from node 1 to 2, 3, 4, 5, 6, 7
    [0, 0, 25, 25, 30, 35, 40],   # distances from node 2 to 3, 4, 5, 6, 7
    [0, 0, 0, 30, 35, 40, 45],    # distances from node 3 to 4, 5, 6, 7
    [0, 0, 0, 0, 40, 45, 50],     # distances from node 4 to 5, 6, 7
    [0, 0, 0, 0, 0, 50, 55],      # distances from node 5 to 6, 7
    [0, 0, 0, 0, 0, 0, 60],       # distance from node 6 to 7
    [0, 0, 0, 0, 0, 0, 0]         # no distance needed for node 7
]
# Function to retrieve distance from i to j, considering the symmetric structure
def get_distance(i, j):
    if i < j:
        return dist[i-1][j-1]
    else:
        return dist[j-1][i-1]

# Memoization for top-down recursion
memo = [[-1] * (1 << (n + 1)) for _ in range(n + 1)]
path_memo = [[None] * (1 << (n + 1)) for _ in range(n + 1)]  # to store paths

def fun(i, mask, goal):
    # Base case: if only the goal node and ith node are visited
    if mask == ((1 << i) | (1 << goal) | 3):  # Only goal and i visited
        return get_distance(goal, i)

    # Memoization
    if memo[i][mask] != -1:
        return memo[i][mask]

    res = 10 ** 9  # Initialize result as a large number
    best_j = -1    # To store the node that gives minimum cost

    # Try to go from all nodes j in mask except i and goal
    for j in range(1, n + 1):
        if (mask & (1 << j)) != 0 and j != i and j != goal:
            candidate_cost = fun(j, mask & (~(1 << i)), goal) + get_distance(j, i)
            if candidate_cost < res:
                res = candidate_cost
                best_j = j

    memo[i][mask] = res  # Storing the minimum value
    path_memo[i][mask] = best_j  # Storing the path to backtrack
    return res

# Driver program to test above logic
start = 1  # Set starting node
goal = 3   # Set goal node

ans = 10 ** 9
best_start_node = -1

# Try going from start node, visiting all nodes, then ending at goal node
for i in range(1, n + 1):
    if i != goal:  # Ensure goal is the last node visited
        candidate_cost = fun(i, (1 << (n + 1)) - 1, goal) + get_distance(i, goal)
        if candidate_cost < ans:
            ans = candidate_cost
            best_start_node = i

# Reconstruct the path using path_memo
path = []
current_node = best_start_node
mask = (1 << (n + 1)) - 1

# Backtrack to get the full path
while current_node is not None:
    path.append(current_node)
    next_node = path_memo[current_node][mask]
    mask &= ~(1 << current_node)
    current_node = next_node

path.append(goal)  # Add the goal node at the end

print("The cost of the most efficient tour =", ans)
print("The sequence of nodes to traverse:", ' -> '.join(map(str, path)))
