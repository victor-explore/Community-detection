import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import random

# Function to import file, create a list of edges and return the list
def import_wiki_vote_data(file_path):
    """
    Import Wikipedia vote data from a file and create a list of unique edges.

    Args:
    file_path (str): Path to the input file containing vote data.

    Returns:
    numpy.ndarray: A 2D array of unique edges, where each row represents an edge [source, target].
    """
    # Initialize an empty list to store the edges
    edges = []

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments (lines starting with '#') and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            # Split the line into source and target nodes
            # and convert them to integers
            source, target = map(int, line.strip().split())
            
            # Add the edge to the list
            edges.append([source, target])
    
    # Convert the list to a numpy array and remove duplicate edges
    nodes_connectivity_list = np.unique(np.array(edges), axis=0)
    
    return nodes_connectivity_list

# Function to visualize a subset of the network and return a graph object
def visualize_network_subset(nodes_connectivity_list, num_nodes=100, output_file="network_subset.png"):
    # Create a graph from the edge list
    G = nx.DiGraph()
    G.add_edges_from(nodes_connectivity_list)

    # Select a subset of nodes for visualization
    subset_nodes = list(G.nodes())[:num_nodes]
    subgraph = G.subgraph(subset_nodes)

    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, arrows=True)

    # Add a title
    plt.title(f"Subset of Network (First {num_nodes} Nodes)")

    # Save the plot
    plt.savefig(output_file)
    plt.close()

    print(f"Network visualization saved as '{output_file}'")
# Function to calculate edge betweenness centrality and return a dictionary of edge betweenness values
def calculate_edge_betweenness(edge_list, k=1000):
    """
    Calculate edge betweenness centrality for a sample of edges.
    
    :param edge_list: NumPy array of edges
    :param k: Number of sample nodes for betweenness calculation
    :return: Dictionary of edge betweenness values
    """
    # Create a dictionary to represent the graph
    graph = {}
    for edge in edge_list:
        source, target = edge
        if source not in graph:
            graph[source] = set()
        graph[source].add(target)

    edge_betweenness = {tuple(edge): 0 for edge in edge_list}
    nodes = list(set(edge_list.flatten()))
    sampled_nodes = random.sample(nodes, min(k, len(nodes)))
    
    for source in sampled_nodes:
        # Run BFS from the source node
        distances = {node: float('inf') for node in nodes}
        distances[source] = 0
        queue = deque([(source, 0)])
        paths = {node: [] for node in nodes}
        paths[source] = [[source]]
        
        while queue:
            node, dist = queue.popleft()
            if node in graph:
                for neighbor in graph[node]:
                    if distances[neighbor] > dist + 1:
                        distances[neighbor] = dist + 1
                        paths[neighbor] = [path + [neighbor] for path in paths[node]]
                        queue.append((neighbor, dist + 1))
                    elif distances[neighbor] == dist + 1:
                        paths[neighbor].extend([path + [neighbor] for path in paths[node]])
        
        # Calculate edge betweenness
        for target in nodes:
            if target != source:
                total_paths = len(paths[target])
                if total_paths > 0:
                    for path in paths[target]:
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i+1])
                            if edge in edge_betweenness:
                                edge_betweenness[edge] += 1 / total_paths
    
    # Normalize by the number of sampled nodes
    for edge in edge_betweenness:
        edge_betweenness[edge] /= len(sampled_nodes)
    
    return edge_betweenness



# Update the file path to use a relative path
file_path = os.path.join("data", "Wiki-Vote.txt")
nodes_connectivity_list_wiki = import_wiki_vote_data(file_path)

# Create the graph G in the global scope
G = nx.DiGraph()
G.add_edges_from(nodes_connectivity_list_wiki)

# Example usage:
visualize_network_subset(nodes_connectivity_list_wiki, num_nodes=100, output_file="wiki_vote_network_subset.png")


# Use nodes_connectivity_list_wiki directly
edge_betweenness = calculate_edge_betweenness(nodes_connectivity_list_wiki)

# Normalize the betweenness values
max_betweenness = max(edge_betweenness.values())
normalized_betweenness = {edge: value / max_betweenness for edge, value in edge_betweenness.items()}

# Print the top 10 edges with highest normalized betweenness centrality
print("Top 10 edges with highest normalized betweenness centrality:")
for edge, centrality in sorted(normalized_betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Edge {edge}: {centrality:.6f}")