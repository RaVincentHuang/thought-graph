import networkx as nx
import json
import re
from collections import deque
import argparse
import pandas as pd

def create_tree(data):
    
    G = nx.DiGraph()
    
    for node in data['nodes']:
        try:
            if node["state"][0] == []:
                acc = node["state"][-3]["acc"]
                logi = node["state"][-3]["logi"]
                infer = node["state"][-3]["infer"]
            else:
                acc = node["state"][-2]["acc"]
                logi = node["state"][-2]["logi"]
                infer = node["state"][-2]["infer"]                
        except Exception:
            print(node["state"])
            
        G.add_node(node['node_id'], state="",success=node['success'], acc=acc, logi=logi, infer=infer)  
    
    for edge in data['edges']:
        G.add_edge(edge['from'], edge['to'], action=edge['action'], value=edge['value'])
    
    
    for node in G.nodes():
        if G.in_degree(node) == 0:
            G.nodes[node]['state'] = ""  
        else:
            predecessors = list(G.predecessors(node))
            for pred in predecessors:
                G.nodes[node]['state'] = G.nodes[pred]['state'] + "\n" + G.edges[pred, node]['action']
    
    
    for node in G.nodes():
        if G.out_degree(node) == 0:  
            try:
                if G.nodes[node]['success']:  
                    
                    current = node
                    while G.in_degree(current) > 0:
                        parent = list(G.predecessors(current))[0]  
                        G.nodes[parent]['success'] = True
                        current = parent
            except Exception:
                continue
    
    return G


def match_by_success_path(G):
    global max_path_length_count
    match_result = []   
    paths_result = []   
    
    
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    paths = list(nx.all_simple_paths(G, source=0, target=leaves))
    
    
    for path in paths:
        tmp = []
        
        for i in range(len(path)-1):
            max_path_length_count = max(max_path_length_count, len(path))
            next_node = path[i+1]
            
            acc = G.nodes[next_node].get('acc', 0)
            logi = G.nodes[next_node].get('logi', 0)
            infer = G.nodes[next_node].get('infer', 0)
            if G.nodes[next_node].get('success', 0) == 1:
                tmp.append(1)
            else:
                if exp_type == "exp1":
                    acc_therehold = 7.77
                else:
                    acc_therehold = 7.5
                if acc >= acc_therehold:
                    tmp.append(1)
                else:
                    tmp.append(0)
        
        paths_result.append(tmp)
        
        max_path_length = max(len(p)-1 for p in paths)
        
        tmp = tmp + [-1] * (max_path_length - len(tmp))
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

def mapping_model_name(model):
    if model == "gpt35":
        return "gpt-3.5-turbo"
    elif model == "qwen15_7b":
        return "qwen-1.5-7b"
    elif model == "mistral_7b":
        return "mistral-7b"
    elif model == "glm4_9b":
        return "glm-4-9b"
    elif model == "llama3_8b":
        return "llama-3-8b"

if __name__ == "__main__":
    max_path_length_count = -1
    
    parser = argparse.ArgumentParser(description="topological stability coefficient calculation")
    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--config', type=str, required=True, help='The path to the config file.')
    parser.add_argument('--benchmark_model', type=str, required=True, help='The benchmark model to use.')
    args = parser.parse_args()
    model_list = load_config(args.config).get("model_list", [])
    exp_type = args.exp_name
    benchmark_model = args.benchmark_model
    total_paths_result_dict = {}
    stability_results = []
    for model in model_list:
        avg_stability_high_after = 0.0
        avg_stability_high_before = 0.0
        avg_stability_without_level = 0.0
        
        input_path = f"raw_data/gsm8k/{exp_type}/{model}/{model}_cleaned_success.jsonl"
        result_log_path = f"raw_data/gsm8k/{exp_type}/result_log/{model}/result.log"
        total_trees = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                total_trees.append(data)
        
        outputs, plans,corrects = get_answer_path(result_log_path)

        total_match_result = []
        total_paths_result = []
        for tree, _ in zip(total_trees, corrects):
            G = create_tree(tree)
            root_node = 0
            levels = assign_levels(G, root_node)

            for node in G.nodes():
                G.nodes[node]['level'] = levels[node]
            match_result, paths_result = match_by_success_path(G)
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

    
    model_list_2 = ["gpt35", "glm4_9b", "llama3_8b", "qwen15_7b", "mistral_7b"]  
    stability_results_new = []
    for model in model_list_2:
        for result in stability_results:
            if result['Model'] == mapping_model_name(model):
                stability_results_new.append(result)
    df = pd.DataFrame(stability_results_new)
    df.to_excel(f"processed_data/gsm8k/{exp_type}_results_{benchmark_model}/ctrs_results.xlsx", 
                index=False)
        
        
        













