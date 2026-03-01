import networkx as nx
import json
import re
from collections import deque
import argparse
import pandas as pd

def create_tree(data):
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['node_id'], state="")  
    for edge in data['edges']:
        G.add_edge(edge['from'], edge['to'], action=edge['action'], value=edge['value'])
    for node in G.nodes():
        if G.in_degree(node) == 0:
            G.nodes[node]['state'] = ""  
        else:
            predecessors = list(G.predecessors(node))
            for pred in predecessors:
                G.nodes[node]['state'] = G.nodes[pred]['state'] + "\n" + G.edges[pred, node]['action']
    return G

def match_by_success_path(plan, G):
    
    subgraphs = []
    for i in range(len(plan)):
        subgraphs.append(plan[i])
    
    match_result = []   
    paths_result = []   
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    paths = list(nx.all_simple_paths(G, source=0, target=leaves))
    
    for i in range(len(paths)):
        tmp = []
        for j in range(len(paths[i])-1):
            if G.get_edge_data(paths[i][j], paths[i][j+1])['action'] == subgraphs[j]:
                tmp.append(1)
            else:
                tmp.append(0)
        paths_result.append(tmp)
        tmp = tmp + [-1] * (len(subgraphs) - len(tmp))
        match_result.append(tmp)

    return match_result, paths_result

def calculate_stability(paths_result, level=False):
    result_high_after = 0.0
    result_high_before = 0.0
    result = 0.0
    for path in paths_result:
        
        
        tmp = 0.0
        tmp_high_after = 0.0
        tmp_high_before = 0.0
        for i in range(len(path)):
            if path[i] == 1:
                if level:
                    tmp_high_after += 1.0 * (i+1) / len(path)  
                    tmp_high_before += 1.0 * (len(path) - i) / len(path)  
                else:
                    tmp += 1.0  
            else:
                continue
        if level:
            result_high_after += tmp_high_after / (1.0 * (len(path)+1) / 2)   
            result_high_before += tmp_high_before / (1.0 * (len(path)+1) / 2) 
        else:
            result += tmp / (1.0 * len(path))  
    if level:
        return result_high_after / len(paths_result), result_high_before / len(paths_result)
    else:
        return result / len(paths_result)

def get_answer_path(log_path):
    with open(log_path,"r",encoding="utf-8") as file:
        log_data = file.read()
    output_pattern = re.compile(r"output='([^']*)'")
    plan_pattern = re.compile(r"'plan': '([^']*)'")
    correct_pattern = re.compile(r"correct=(\w+)")

    outputs = output_pattern.findall(log_data)
    plans = plan_pattern.findall(log_data)
    corrects = correct_pattern.findall(log_data)
    
    new_plans = []
    for plan in plans:
        tmp = plan[2:].split("\\n")[:-2]
        new_plans.append(tmp)
    return outputs, new_plans,corrects

def assign_levels(G, root):
    levels = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        current_level = levels[node]
        for neighbor in G.neighbors(node):
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)
    return levels

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def model_mapping(model_name):
    if model_name == "qwen":
        return "qwen-max-latest"
    elif model_name == "gpt4o":
        return "gpt-4o"
    elif model_name == "gpt4":
        return "gpt-4"
    elif model_name == "gpt35":
        return "gpt-3.5-turbo"
    elif model_name == "glm4":
        return "glm-4-plus"
    elif model_name == "claude":
        return "claude-3-5-sonnet"

def mapping_model_name(model):
    if model == "gpt4o":
        return "gpt-4o"
    elif model == "gpt35":
        return "gpt-3.5-turbo"
    elif model == "gpt4":
        return "gpt-4"
    elif model == "claude":
        return "claude-3-5-sonnet"
    elif model == "qwen":
        return "qwen-max-latest"
    elif model == "glm4":
        return "glm-4-plus"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="topological stability coefficient calculation")
    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--config', type=str, required=True, help='The path to the config file.')
    parser.add_argument('--step', type=int, required=True, help='The step to use.')
    parser.add_argument('--benchmark_model', type=str, required=True, help='The benchmark model to use.')
    args = parser.parse_args()
    model_list = load_config(args.config).get("model_list", [])
    exp_type = args.exp_name
    step = args.step
    benchmark_model = args.benchmark_model
    total_paths_result_dict = {}
    stability_results = []
    for model in model_list:
        avg_stability_high_after = 0.0
        avg_stability_high_before = 0.0
        avg_stability_without_level = 0.0
        
        input_path = rf"raw_data/blocksworld/{exp_type}/{model}/{model}_step{step}_cleaned_success.jsonl"
        result_log_path = rf"raw_data/blocksworld/{exp_type}/result_log/{model}_step{step}/result.log"
        total_trees = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                total_trees.append(data)
        
        outputs, plans,corrects = get_answer_path(result_log_path)

        total_match_result = []
        total_paths_result = []
        for tree, plan in zip(total_trees, plans):
            G = create_tree(tree)
            root_node = 0
            levels = assign_levels(G, root_node)

            for node in G.nodes():
                G.nodes[node]['level'] = levels[node]
            match_result, paths_result = match_by_success_path(plan, G)
            total_match_result.append(match_result)
            total_paths_result.append(paths_result)
            stability_high_after, stability_high_before = calculate_stability(paths_result, level=True)
            stability_without_level = calculate_stability(paths_result, level=False)
            avg_stability_high_after += stability_high_after
            avg_stability_high_before += stability_high_before
            avg_stability_without_level += stability_without_level
        
        avg_stability_high_after /= len(total_trees)
        avg_stability_high_before /= len(total_trees)
        avg_stability_without_level /= len(total_trees)
        print(model, "CTRS:", round(avg_stability_high_after, 3))
        stability_results.append({
            'Model': mapping_model_name(model),
            'CTRS': round(avg_stability_high_after, 3),
        })
        total_paths_result_dict[model] = total_paths_result
    model_list_2 = ["gpt4o","gpt4","gpt35","qwen","glm4","claude"]
    stability_results_new = []
    for model in model_list_2:
        for result in stability_results:
            if result['Model'] == model_mapping(model):
                stability_results_new.append(result)
    df = pd.DataFrame(stability_results_new)
    df.to_excel(f"processed_data/blocksworld/{exp_type}_results_{benchmark_model}/ctrs_results.xlsx", 
                index=False)