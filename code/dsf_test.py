from dsf import *
import random

SQUARE_SIZE = 100

graph_nodes = []
for i in range(SQUARE_SIZE):
	for j in range(SQUARE_SIZE):
		graph_nodes.append((i, j))

forest = DisjointSetForest(graph_nodes)

loc_to_index_map = {node.data : node.index for node in forest.nodes}

edge_data = []
for i in range(SQUARE_SIZE):
	for j in range(SQUARE_SIZE):
		end_loc = loc_to_index_map[(i, j)]
		
		if i != 0:
			start_loc = loc_to_index_map[(i - 1, j)]
			edge_data.append((random.random(), start_loc, end_loc))

		if j != 0:
			start_loc = loc_to_index_map[(i, j - 1)]
			edge_data.append((random.random(), start_loc, end_loc))

edge_data.sort(key=lambda x: x[0])

edges_used = set()
while forest.numClusters > 1:
	_, i, j = edge_data.pop(0)

	if forest.union(i, j) >= 0:
		edges_used.add((i, j))

	rep_i = forest.find(i)
	rep_j = forest.find(j)

for i in range(SQUARE_SIZE):
	node_row = ""
	vert_row = ""
	for j in range(SQUARE_SIZE):
		node_index = loc_to_index_map[(i, j)]

		if i != 0:
			horiz_neighbor_index = loc_to_index_map[(i - 1, j)]
			vert_row += "| " if (horiz_neighbor_index, node_index) in edges_used else "  "

		if j != 0:
			vert_neighbor_index = loc_to_index_map[(i, j - 1)]
			node_row += "-" if (vert_neighbor_index, node_index) in edges_used else " "

		node_row += "o"

	print(vert_row)
	print(node_row)


