import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """                                     #IM ASSUMING ADJ_MAT IS FORMATTED CORRECTLY AS SPECIFIED ^^
        size = self.adj_mat.shape[0]
        self.mst = np.zeros((size, size))       #initialize mst to be same dimensions as adj_mat

        visited = [False] * size                #initialize visited list to track nodes visited

        heap = []
        heapq.heapify(heap)                     #initialize heap (sorts weights to return smallest)

        out = 0
        visited[out] = True                     #first node is 0; it is now visited

        while False in visited:                 #while we haven't visited all nodes yet
            for into in range(size):
                weight = self.adj_mat[out][into]
                if weight > 0:
                    heapq.heappush(heap, (weight, out, into))   #add all neighbors w/ weights to heap if > 0

            weight, out, into = heapq.heappop(heap)
            while visited[into]:
                weight, out, into = heapq.heappop(heap)         #pop smallest node, until its destination hasn't been visited yet
            
            self.mst[out][into] = weight                        #add weights to mst at position (into, out) and (out, into)
            self.mst[into][out] = weight                        #(should be symmetric along diagonal since undirected)

            visited[into] = True                                #set this new node we visited to true
            out = into                                          #after looping, add new neighbors to most recently visited

