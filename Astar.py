'''
This is a simple script that implements the A* algorithm.
Wikipage (sucks, no closed list): https://en.wikipedia.org/wiki/A*_search_algorithm
Be careful about the reference online, I found most of them not equivalent with each other. Some doesn't make sense.
Reference link:  http://robotics.caltech.edu/wiki/images/e/e0/Astar.pdf
Reference link:  refer to Nikolay's pdf.
Another good reference: https://www.redblobgames.com/pathfinding/a-star/implementation.html
=========================
Author  :  Muhan Zhao
Date    :  Jul. 16, 2019
Location:  West Hill, LA, CA
=========================
'''
# Dijkstra's algorithm, as another example of a uniform-cost search algorithm, can be viewed as a special case of A*
# where h(x) = 0 for all x.
# If h(x) underestimates the cost, guaranteed to find the optimal path but not efficient;
# If h(x) overestimates the cost, not guaranteed to find the optimal path but runs faster.
import matplotlib.pyplot as plt
inf = 1e+20


class Node:
    def __init__(self, i, j, children, cost):
        '''

        :param i        :  Row position
        :param j        :  Column position
        :param children :  Children of current node, a list of lists, each sub-list is a child
        :param cost     :  Dict-object, key -> child position (i,j), value -> cost
        '''
        # Each node also maintains a pointer to its parent,
        # so that later we can retrieve the best solution found, if one is found.
        self.parent = None
        self.children = []
        self.position = [i, j]
        self.children = children
        self.cost = cost

    def __repr__(self):
        return 'Node: Position = (%s, %s)' % (self.position[0], self.position[1])


class Graph:
    def __init__(self, input_matrix, move_guide):
        self.connect_matrix = input_matrix
        self.nodes        = {}
        self.open_list    = []  # visited not expanded
        self.closed_list  = []  # visited and expanded
        self.nodes_f      = []  # store f function values of each node
        self.nodes_g      = []  # store g function values of each node
        self.optimal_path = []  # store the optimal path
        self.optimal_cost = []  # store the optimal cost
        self.n = len(input_matrix)
        self.m = len(input_matrix[0])
        self.size = self.n * self.m
        # order the nodes in row-wise
        if move_guide == 'non-diagonal':
            # only move in horizontal and vertical directions
            self.movement = [[-1, 0],
                             [1, 0],
                             [0, -1],
                             [0, 1]]
        else:  # move in horizontal/vertical/diagonal directions
            self.movement = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.nodes_maker()
        self.optimal_path = []

    def nodes_maker(self):
        for i in range(self.n):
            for j in range(self.m):
                position = [i, j]
                children = []
                cost = {}
                potential_children = [ [direction[i] + position[i] for i in range(2)] for direction in self.movement]
                for child in potential_children:
                    if 0 <= child[0] <= self.n - 1 and 0 <= child[1] <= self.m - 1:
                        # child within the connect matrix
                        child_cost = self.connect_matrix[child[0]][child[1]]
                        if child_cost >= 0:
                            # negative connect matrix element -> obstacle
                            children.append(child)
                            cost[tuple(child)] = child_cost
                self.nodes[i, j] = Node(i, j, children, cost)

    def manhattan_heuristic(self, current_node, end_node):
        '''
        Calculate the Manhattan heuristic of current node and end node
        :param current_node:  current node position, [i, j]
        :param end_node:   end node position, [i, j]
        :return:
        '''
        return max([ abs(x - y) for x, y in zip(current_node, end_node)])

    def euclidean_heuristic(self, current_node, end_node):
        return sum([ (x - y) ** 2 for x, y in zip(current_node, end_node)])

    def astar(self, start, end):
        # initialize the f values and g values to be infinity at the beginning
        self.nodes_g = {(i, j): inf for j in range(self.m) for i in range(self.n)}
        self.nodes_f = {(i, j): inf for j in range(self.m) for i in range(self.n)}
        self.nodes_g[tuple(start)] = 0
        self.nodes_f[tuple(start)] = self.nodes_g[tuple(start)] + self.manhattan_heuristic(start, end)
        self.open_list.append(start)

        while len(self.open_list) > 0:
            # Pick the node in open list with the lowest f-value
            current_node = min(self.open_list, key=lambda point: self.nodes_f[tuple(point)])
            if current_node == end:
                # end node reached, reconstruct the optimal path
                break
            # Remove the current node from the open list
            self.open_list.remove(current_node)
            # Add the current node to the closed list
            self.closed_list.append(current_node)

            for child in self.nodes[tuple(current_node)].children:
                if child not in self.closed_list:
                    alt_cost = self.nodes_g[tuple(current_node)] + self.nodes[tuple(current_node)].cost[tuple(child)]
                    if alt_cost < self.nodes_g[tuple(child)]:
                        self.nodes_g[tuple(child)] = alt_cost
                        self.nodes_f[tuple(child)] = self.nodes_g[tuple(child)] + self.manhattan_heuristic(child, end)
                        self.nodes[tuple(child)].parent = current_node
                        self.open_list.append(child)

        self.optimal_path = [end]
        previous = self.nodes[tuple(end)].parent
        while previous is not None:
            self.optimal_path.append(previous)
            previous = self.nodes[tuple(previous)].parent
        self.optimal_path.reverse()
        return self.nodes_g[tuple(end)], self.optimal_path

    def plot_optimal_path(self):
        plt.figure()
        # draw the grid
        for i in range(self.n):  # draw the horizontal lines
            plt.plot([0, self.m - 1], [i, i], color='k')
        for j in range(self.m):  # draw the vertical lines
            plt.plot([j, j], [0, self.n - 1], color='k')

        # scatter plot the obstacles and points on the optimal path
        for point in self.optimal_path:
            plt.scatter(point[1], self.n - 1 - point[0], c='r', s=30)
        # scatter plot the obstacles as the black dot
        for i in range(self.n):
            for j in range(self.m):
                if self.connect_matrix[i][j] < 0:
                    plt.scatter(j, self.n - 1 - i, c='k', s=30)
        # plot the path
        for i in range(len(self.optimal_path)):
            if i == len(self.optimal_path) - 1:
                break
            else:
                point = self.optimal_path[i]
                next = self.optimal_path[i+1]
                plt.plot([point[1], next[1]], [self.n - 1 - point[0], self.n - 1 - next[0]], color='r', linewidth=3)
        plt.show()


if __name__ == '__main__':
    # Below is an example of the map:
    # negative element indicates that there is obstacle at the position
    # positive element indicates the cost to move, uniform (for now) in 4 directions (up/down/left/right)
    e = [[1, 1, 1, -1, 1],
         [1, 10, 5, -1, 1],
         [1, 10, 5, -1, 1],
         [1, 10, 1, -1, 1],
         [1, -1, 1, 1, 1]]

    g = Graph(e, 'non-diagonal')
    # start: position of the start node, given in x and y axis;
    # end  : position of the end node.
    s = [0, 0]
    d = [4, 4]
    opt_cost, opt_path = g.astar(s, d)
    print('optimal cost = ', opt_cost)
    print('optimal path = ', opt_path)
    g.plot_optimal_path()
