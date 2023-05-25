import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

with open('ecoli.data', 'r') as file:
    lines = file.readlines()

for line in lines:
    values = line.split()
    bacteria = values[0]
    G.add_node(bacteria)
    connected_bacteria = [values[i] for i in range(1, 8)]
    for connected_bacterium in connected_bacteria:
        G.add_edge(bacteria, connected_bacterium)

nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()