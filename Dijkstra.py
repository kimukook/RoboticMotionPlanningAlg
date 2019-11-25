'''
This is a simple script that implements the Dijkstra's algorithm that modified from the following link.
Reference link: https://dev.to/mxl/dijkstras-algorithm-in-python-algorithms-for-beginners-dkc
Another good reference: https://www.geeksforgeeks.org/printing-paths-dijkstras-shortest-path-algorithm/
=========================
Author  :  Muhan Zhao
Date    :  Jul. 16, 2019
Location:  West Hill, LA, CA
=========================
'''
import copy
inf = 1e+20


class Edge:
    def __init__(self, edge):
        assert len(edge) == 3, 'The input for edges should be 3: start node, end node and edge cost. ' \
                               'Wrong edge =' + (str(edge))
        self.start = edge[0]
        self.end   = edge[1]
        self.cost  = edge[2]

    def __repr__(self):
        return 'Edge: StartNode = %s; EndNode = %s; EdgeCost = %s' % (self.start, self.end, self.cost)


class Graph:
    def __init__(self, *edges):
        # The reason for the following line: *edges will pack the list as a set structure.
        # edge[0] will extract the list object.

        # self.edges store all the edge info
        edges = edges[0]
        self.edges = [self.makeEdge(edge) for edge in edges]

        # self.nodes store all the nodes
        # self.nodes = []
        # [self.nodes.extend([edge.start, edge.end]) for edge in self.edges]
        # self.nodes = set(self.nodes)
        self.nodes = set()
        [self.nodes.update(edge.start, edge.end) for edge in self.edges]

        # self.neighbors store all the nodes as key, the edge end node and cost as element
        self.neighbors = {node: [] for node in self.nodes}
        self.makeNeighbors()

        # class info for Dijkstra algorithm
        self.distances = {}  # record the distance of the node i to the source node;
        self.parent = {}  # record the parent of node i, recover the optimal path;
        self.queue = []  # record unvisited nodes;

    def makeEdge(self, edge):
        return Edge(edge)

    def makeNeighbors(self):
        for edge in self.edges:
            # loop over end node and start node
            self.neighbors[edge.start].append([edge.end, edge.cost])
            self.neighbors[edge.end].append([edge.start, edge.cost])

    def dijkstra(self, start_node, end_node):
        self.distances = {node: inf for node in self.nodes}
        self.distances[start_node] = 0
        self.parent = {node: None for node in self.nodes}
        self.queue = copy.copy(self.nodes)  # shallow copy, deep copy will refer queue as nodes and change nodes as well

        while len(self.queue) > 0 and end_node in self.queue:  # while queue is not empty and end_node not visited
            current_node = min(self.queue, key=lambda node: self.distances[node])
            self.queue.remove(current_node)

            for neighbor, cost in self.neighbors[current_node]:
                # Only update the nodes that are still in the queue.
                # The points removed from the queue is optimzied.
                if neighbor in self.queue:
                    alt_cost = self.distances[current_node] + cost
                    # If a shorter path is found, update the cost of this neighbor
                    if alt_cost < self.distances[neighbor]:
                        self.distances[neighbor] = alt_cost
                        self.parent[neighbor] = current_node

        optimal_path = list(end_node)
        previous = self.parent[end_node]
        while previous is not None:
            optimal_path.append(previous)
            previous = self.parent[previous]
        optimal_path.reverse()
        return self.distances[end_node], optimal_path


if __name__ == '__main__':
    e = [("a", "b", 7),  ("a", "c", 9),  ("a", "f", 14), ("b", "c", 10),
    ("b", "d", 15), ("c", "d", 11), ("c", "f", 2),  ("d", "e", 6),
    ("e", "f", 9)]

    G = Graph(e)
    opt_distance, opt_path = G.dijkstra('a', 'd')
    print(opt_distance)
    print(opt_path)
    # solution:
    # optimal_distance: 20
    # optimal_path
    # Out: ['a', 'c', 'f', 'e']
