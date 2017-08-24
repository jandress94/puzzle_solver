class DSFNode:
	def __init__(self, index, data):
		self.clusterSize = 1
		self.index = index
		self.parent = self
		self.data = data

class DisjointSetForest:
	def __init__(self, data_list, should_combine_fn = None, combine_data_fn = None):
		self.numClusters = len(data_list)
		self.nodes = [DSFNode(i, data_list[i]) for i in range(len(data_list))]
		self.should_combine_fn = should_combine_fn
		self.combine_data_fn = combine_data_fn

	# Given the index of a node, finds the cluster representative for that node.
	# Compresses paths as it goes
	def find(self, i):
		return self.find_node(self.nodes[i])

	# returns the representative, as well as the local rotation and local coordinates of the input node
	# note that because we are doing path compression, these are with respect to the representative
	def find_node(self, n):
		if n.parent != n:
			n.parent = self.find_node(n.parent)

		return n.parent

	# Merges the clusters holding the nodes at index i and j.
	# Returns the index of the new rep if it made a merge, -1 otherwise
	def union(self, i, j):
		rep_i = self.find(i)
		rep_j = self.find(j)

		# check that the pieces aren't already in the same cluster
		if rep_i == rep_j:
			return -1

		if rep_i.clusterSize >= rep_j.clusterSize:
			clust_big = rep_i
			clust_small = rep_j
		else:
			clust_big = rep_j
			clust_small = rep_i

		if self.should_combine_fn is not None and not self.should_combine_fn(clust_big.data, clust_small.data):
			return -1

		clust_small.parent = clust_big
		clust_big.clusterSize += clust_small.clusterSize
		self.numClusters -= 1
		if self.combine_data_fn is not None:
			clust_big.data = self.combine_data_fn(clust_big.data, clust_small.data)
		
		return clust_big.index