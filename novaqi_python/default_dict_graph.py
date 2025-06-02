from collections import defaultdict

# Create the graph as an adjacency list
graph = defaultdict(set)

# Function to add an undirected edge
def add_edge(u, v):
    graph[u].add(v)
    graph[v].add(u)

# Function to remove a node and all its edges
def remove_node(node):
    if node not in graph:
        return
    
    for neighbor in list(graph[node]):
        print(graph[node])
        graph[neighbor].remove(node)
    del graph[node]

# Function to print the current state of the graph
def print_graph():
    print("Current Graph:")
    for node, neighbors in graph.items():
        print(f"{node}: {sorted(neighbors)}")
    print("-" * 30)

# --- Example usage ---

# Build graph
add_edge(1, 2)
add_edge(1, 3)
add_edge(2, 4)
add_edge(3, 4)
add_edge(4, 5)

graph2 = defaultdict(list)

for i in graph:
    degree = len(graph[i])
    graph2[i] = [0] * degree 

print(graph)
print(graph2)
# print(graph.items())

# # # Remove node 4
# remove_node(2)

# print(graph.items())

# print_graph()

# # Try removing a node that doesn’t exist
# remove_node(10)

# print_graph()
