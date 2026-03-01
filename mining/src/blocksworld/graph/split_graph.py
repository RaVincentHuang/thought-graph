import os
import argparse
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def split_graph(graph_file, save_path, mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nodes = {}
    edges = []
    with open(graph_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == 'v':
                node_id = int(parts[1])
                attribute = parts[2]  
                level = parts[3]
                success = parts[4]
                nodes[node_id] = [attribute, level,success]
            elif parts[0] == 'e':
                source = int(parts[1])
                target = int(parts[2])
                
                weight = parts[3]                
                action = " ".join(parts[4:]).strip()
                edges.append((source, target, weight, action))
                nodes[source].append((target, weight, action))
                nodes[target].append((source, weight, action))
    
    def find_tree(start_node, visited):
        stack = [start_node]
        tree_nodes = set()
        tree_edges = []

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                tree_nodes.add(node)
                for neighbor, weight, action in nodes[node][3:]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                        tree_edges.append((node, neighbor, weight, action))

        return tree_nodes, tree_edges

    visited = set()
    tree_index = 0

    for node in nodes:
        if node not in visited:
            tree_nodes, tree_edges = find_tree(node, visited)
            min_node_id = min(tree_nodes)  
            sorted_tree_nodes = sorted(tree_nodes)  
            sorted_tree_edges = sorted(tree_edges, key=lambda e: (e[0], e[1]))  
            with open(os.path.join(save_path, f'tree_{tree_index}.txt'), 'w',encoding='utf-8') as tree_file:
                if mode == "dfs":
                    tree_file.write("t # 0\n")
                for n in sorted_tree_nodes:
                    attribute = nodes[n][0]  
                    level = nodes[n][1]      
                    success = nodes[n][2]    
                    adjusted_id = n - min_node_id  
                    tree_file.write(f'v {adjusted_id} {attribute} {level} {success}\n')
                for e in sorted_tree_edges:
                    adjusted_edge = (e[0] - min_node_id, e[1] - min_node_id, e[2], e[3])
                    tree_file.write(f'e {adjusted_edge[0]} {adjusted_edge[1]} {adjusted_edge[2]} {adjusted_edge[3]}\n')  
            tree_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for a specific experiment.")
    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--benchmark_model', type=str, required=True, help='The benchmark model to use.')
    parser.add_argument('--cluster_id', type=int, required=True, help='The cluster ID to process.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, required=True, help='mode dfs or bfs')

    args = parser.parse_args()
    exp_type = args.exp_name
    model_list = load_config(args.config).get("model_list", [])
    benmark_model = args.benchmark_model
    cluster_num = args.cluster_id
    mode = args.mode
    for model in model_list:
        graph_file = rf"processed_data/blocksworld/{exp_type}_results_{benmark_model}/graph_cluster_{cluster_num}_{mode}/{model}.txt"
        save_path = rf"processed_data/blocksworld/{exp_type}_results_{benmark_model}/{mode}_graph_{cluster_num}/{model}"
        split_graph(graph_file, save_path,mode)