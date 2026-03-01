import networkx as nx
import numpy as np
from scipy.stats import entropy
import json
import os
import argparse

def random_walk_sampling(G, root, num_samples=1000):
    path_counts = {}
    for _ in range(num_samples):
        current_node = root
        path = [G.nodes[current_node]['tag']]
        while G.out_degree(current_node) > 0:
            neighbors = list(G.successors(current_node))
            current_node = np.random.choice(neighbors)
            path.append(G.nodes[current_node]['tag'])
        path_tuple = tuple(path)
        if path_tuple in path_counts:
            path_counts[path_tuple] += 1
        else:
            path_counts[path_tuple] = 1
    total_paths = sum(path_counts.values())
    path_distribution = {path: count / total_paths for path, count in path_counts.items()}
    return path_distribution

def weighted_random_walk_sampling(G, root, num_samples=1000):
    path_counts = {}
    for _ in range(num_samples):
        current_node = root
        path = [G.nodes[current_node]['tag']]
        while G.out_degree(current_node) > 0:
            neighbors = list(G.successors(current_node))
            weights = [G[current_node][neighbor]['value'] for neighbor in neighbors]
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
            
            current_node = np.random.choice(neighbors, p=probabilities)
            path.append(G.nodes[current_node]['tag'])
        path_tuple = tuple(path)
        if path_tuple in path_counts:
            path_counts[path_tuple] += 1
        else:
            path_counts[path_tuple] = 1
    total_paths = sum(path_counts.values())
    path_distribution = {path: count / total_paths for path, count in path_counts.items()}

    return path_distribution

def calculate_kl_divergence(base_distribution, target_distribution, epsilon=0.1):
    
    all_keys = set(base_distribution.keys()).union(set(target_distribution.keys()))
    base_values = np.array([base_distribution.get(key, epsilon) for key in all_keys])
    target_values = np.array([target_distribution.get(key, epsilon) for key in all_keys])
    
    base_values /= np.sum(base_values)
    target_values /= np.sum(target_values)
    kl_div = entropy(base_values, target_values)
    
    return kl_div


def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="KL divergence calculation")
    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--benchmark_model', type=str, required=True, help='The benchmark model to use.')
    parser.add_argument('--cluster_num', type=int, required=True, help='The cluster number to use.')
    parser.add_argument('--config', type=str, required=True, help='The path to the config file.')
    parser.add_argument('--mode', type=str, required=True, help='The mode type.')
    args = parser.parse_args()
    model_list = load_config(args.config).get("model_list", [])
    exp_type = args.exp_name
    benchmark_model = args.benchmark_model
    cluster_num = args.cluster_num
    mode = args.mode
    num_samples = 100
    distribution_list = [[] for i in range(len(model_list))]
    distribution_weighted_list = [[] for i in range(len(model_list))]
    for model in model_list:
        input_path = f"processed_data/gsm8k/{exp_type}_results_{benchmark_model}/{mode}_graph_{cluster_num}/{model}"
        result_log_path = f"raw_data/gsm8k/{exp_type}/result_log/{model}/result.log"
        for file in os.listdir(input_path):
            G = nx.DiGraph()
            
            with open(os.path.join(input_path, file), "r", encoding="utf-8",errors="ignore") as f:
                for line in f:
                    line = line.encode('utf-8', 'ignore').decode('utf-8')
                    parts = line.split()
                    if parts[0] == "v":
                        G.add_node(int(parts[1]), tag=int(parts[2]))
                    elif parts[0] == "e":
                        G.add_edge(int(parts[1]), int(parts[2]), value=float(parts[3]))
            
            root_node = 0
            distribution = random_walk_sampling(G, root_node, num_samples)
            distribution_weighted = weighted_random_walk_sampling(G, root_node, num_samples) 
            distribution_list[model_list.index(model)].append(distribution)
            distribution_weighted_list[model_list.index(model)].append(distribution_weighted)
    
    
    KL_matrix = np.zeros((len(model_list), len(model_list)))
    for i, base_distributions in enumerate(distribution_list):
        for j, target_distributions in enumerate(distribution_list):
            if i != j:
                tmp_result = 0.0
                for k in range(len(base_distributions)):
                    tmp_result += calculate_kl_divergence(base_distributions[k], target_distributions[k])
                KL_matrix[i][j] = tmp_result / len(base_distributions)

    
    result_dir = f"processed_data/gsm8k/{exp_type}_results_{benchmark_model}/{mode}_graph_{cluster_num}/kl_results"
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, "kl_matrix.npy"), KL_matrix)
    np.save(os.path.join(result_dir, "model_list.npy"), np.array(model_list))
    print(f"results save in {result_dir}")

    
    KL_matrix_weighted = np.zeros((len(model_list), len(model_list)))
    for i, base_distributions in enumerate(distribution_weighted_list):
        for j, target_distributions in enumerate(distribution_weighted_list):
            if i != j:
                tmp_result = 0.0
                for k in range(len(base_distributions)):
                    tmp_result += calculate_kl_divergence(base_distributions[k], target_distributions[k])
                KL_matrix_weighted[i][j] = tmp_result / len(base_distributions)
    
    np.save(os.path.join(result_dir, "kl_matrix_weighted.npy"), KL_matrix_weighted)












