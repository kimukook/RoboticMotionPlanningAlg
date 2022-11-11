classdef Dijkstra
properties
    n % #nodes in the graph
    position % array: size n by 2 vector. n->number of nodes.
    connectivity % cell: size n by *. 
    queue % vector, size n by 1. 
    parent % vector, size n by 1.
end
methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = Dijkstra(position, connectivity)
        % intialization
        obj.n = size(position, 1);
        obj.position = position;
        obj.connectivity = connectivity;
        obj.queue = [];
        obj.parent = [];
        % TODO: repeat nodes detector
    end % function Dijkstra 
    
    function [shortest_path_seq, dist] = find_optimal_path(obj, start_node)
        % compute the shorest path from the given start node to any other
        % node in the graph
        % Input: 
        %           start: the index of the start node; (potentially could be the position of the start node)
        shortest_path_seq = cell(obj.n, 1);
        

        % Initialize the queue sequence, adding all nodes to the queue.
        obj.queue = 1 : obj.n;
        
        % Initialize the value of shortest path for each node in the graph
        dist = inf(obj.n, 1);
        dist(start_node) = 0;
        
        % Initialize the parent of each node
        obj.parent = zeros(obj.n, 1);
        % the parent node of the start node is itself
        obj.parent(start_node) = start_node;
        
        iter = 0;
        while ~isempty(obj.queue)
            iter = iter + 1;
            % find the node with the shortest dist in the queue, i.e., find
            % the iter-th smallest number in the dist. 
            [~, index] = mink(dist, iter);
            current_node_index = index(iter);
            
            % remove current_node_index from the queue
            obj.queue(obj.queue == current_node_index) = [];
            
            % add the parent of current_node_index, and the shortest path 
            % sequence of its parent to the shorest_path_seq of current_node_index
%             keyboard
            if obj.parent(current_node_index) == start_node
%                 shortest_path_seq{current_node_index} = [shortest_path_seq{current_node_index}, obj.parent(current_node_index)];
                if current_node_index ~= start_node
                    shortest_path_seq{current_node_index} = [obj.parent(current_node_index), current_node_index];
                else
                    shortest_path_seq{current_node_index} = obj.parent(current_node_index);
                end
            else
                try
                    shortest_path_seq{current_node_index} = [shortest_path_seq{obj.parent(current_node_index)}, current_node_index];
                catch
                    keyboard
                end
            end
            % loop over neightbors of current_node_index in queue
            for k = 1 : numel(obj.connectivity{current_node_index})
%                 keyboard
                if any(obj.queue == obj.connectivity{current_node_index}(k))
                    % neighbor k of current_node_index is still in Q
                    new_distance = dist(current_node_index) + compute_distance(obj, current_node_index, obj.connectivity{current_node_index}(k));
                    if new_distance < dist(obj.connectivity{current_node_index}(k))
                        dist(obj.connectivity{current_node_index}(k)) = new_distance;
                        obj.parent(obj.connectivity{current_node_index}(k)) = current_node_index;
                    end
                end
            end
                
        end                    
    end % function shortest_path_seq
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function dis = compute_distance(obj, start_node, end_node)
        % compute the Euclidean distance between the connected edge E, 
        % E=(start_node, end_node).
        dis = sqrt(sum((obj.position(start_node, :) - obj.position(end_node, :)).^2));
    end % function compute_distance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function plot_optimal_path(obj, seq)
        for i = 1 : size(seq, 1)
            plot(obj.position(seq{i}, 1), obj.position(seq{i}, 2), 'r'); hold on
            scatter(obj.position(seq{i}, 1), obj.position(seq{i}, 2), 'r')
        end
    end
end
end % classdef Dijkstra