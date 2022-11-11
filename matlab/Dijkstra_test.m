clear, clc, close all
n = 10; % #points
position = [randperm(n, n)', randperm(n, n)']; % generate 2D position matrix for each node

tri = delaunay(position); % construct the Delaunay triangulation of the randomly generated nodes
triplot(tri, position(:, 1), position(:, 2)); hold on% visualize Delaunay triangulation in face-vertex format
for i = 1 : n
    text(position(i, 1)+0.2, position(i, 2), num2str(i), 'FontSize', 12)
end
% The goal here is not to distinguish the nodes with odd edges, but purely
% find the optimal path from each node in the graph to all the others;

% First construct the connectivity cell, complexity O(3*size(tri)+n).
connectivity = cell(n, 1); % ID{i}: The neighbors for node i;
for i = 1 : size(tri, 1)
    for j = 1 : 3
        index_list_except_j = [1, 2, 3];
        connectivity{tri(i, j)} = [connectivity{tri(i, j)}, tri(i, setdiff([1,2,3], j))];
    end
end
% delete repeated nodes for each row
for i = 1 : n
    connectivity{i} = unique(connectivity{i});
end

% Dijkstra can not have repeat nodes!
pipe = Dijkstra(position, connectivity);
[seq, dist] = find_optimal_path(pipe, 2);
plot_optimal_path(pipe, seq)