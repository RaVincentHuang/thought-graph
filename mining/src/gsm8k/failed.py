import pandas as pd
import argparse
import os
import json

def graph_gen(nodes_file, edges_file,model_list, save_path, tag_file, mode):
    tag_df = pd.read_csv(tag_file)
    tag_dict = dict(zip(tag_df["node_id"], tag_df["cluster_id"]))
    
    df_actions = pd.read_csv(edges_file)
    df = pd.read_csv(nodes_file)
    for model in model_list:
        model_df = df[df["data_src"] == model]
        model_df.to_csv(os.path.join(save_path, model+".csv"), index=False)
        with open(os.path.join(save_path, model+".txt"), "w",encoding="utf-8") as f:
            f.write("t # 0\n")
            sorted_df = model_df.copy()
            sorted_df['node_id_int'] = sorted_df['node_id'].apply(lambda x: int(x[1:]))
            sorted_df = sorted_df.sort_values(by='node_id_int')     
            offset = sorted_df['node_id_int'].min()
            for index, data in sorted_df.iterrows():
                
                success_value = "1" if data["success"] else "0"
                tag = tag_dict[data["node_id"]]
                def process_tag(tag):
                    if pd.isna(tag):
                        return -1
                    return int(float(tag))
                tag = process_tag(tag)
                
                if mode == "bfs":
                    f.write("v " + str(int(data["node_id"][1:]) - offset) + " " + str(tag) + " " + success_value + 
                            " " + str(data["acc"]) + " " + str(data["logi"]) + " " + str(data["dis"]) + " " + str(data["infer"]))
                    f.write("\n")
                elif mode == "dfs":
                    
                    level = 0
                    current_node = data["node_id"]
                    while current_node != "root":
                        parent_node = model_df[model_df["node_id"] == current_node]["parent_id"].values[0]
                        if parent_node == "root":
                            break
                        current_node = parent_node
                        level += 1
                    node_value = str(level)  

                    f.write("v " + str(int(data["node_id"][1:]) - offset) + " " + str(tag) + " " + node_value + " " + success_value)
                    f.write("\n")

            
            tuple_list = []
            for index, data in model_df.iterrows():
                if data["parent_id"] != "root":
                    
                    if mode == "dfs":
                        tuple_list.append((int(data["parent_id"][1:]) - offset, int(data["node_id"][1:]) - offset, float(data["value"])))
                    elif mode == "bfs":
                        tmp_intensity = df_actions[(df_actions["Source"] == data["parent_id"]) & (df_actions["Target"] == data["node_id"])]["intensity"].values[0]        
                        tmp_action = df_actions[(df_actions["Source"] == data["parent_id"]) & (df_actions["Target"] == data["node_id"])]["action"].values[0].replace('\n', '')
                        tuple_list.append((int(data["parent_id"][1:]) - offset, int(data["node_id"][1:]) - offset, tmp_intensity, tmp_action))
            
            tuple_list.sort(key=lambda x: (x[0], x[1]))
            for tuple in tuple_list:
                if mode == "dfs":
                    f.write("e " + str(tuple[0]) + " " + str(tuple[1]) + " " + str(tuple[2]))
                elif mode == "bfs":
                    f.write("e " + str(tuple[0]) + " " + str(tuple[1]) + " " + str(tuple[2]) + " " + str(tuple[3]))
                f.write("\n")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def process_data(exp_name, benchmark_model, cluster_id, model_list, mode):
    nodes_file = rf"processed_data/gsm8k/{exp_name}_results_{benchmark_model}/tag_pool/nodes.csv"
    edges_file = rf"processed_data/gsm8k/{exp_name}_results_{benchmark_model}/tag_pool/edges.csv"
    tag_file = rf"processed_data/gsm8k/{exp_name}_results_{benchmark_model}/cluster_result/cluster_nodes_2-2_cluster_id_{cluster_id}.csv"
    save_path = rf"processed_data/gsm8k/{exp_name}_results_{benchmark_model}/graph_cluster_{cluster_id}_{mode}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("Model list:", model_list)
    graph_gen(nodes_file, edges_file,model_list, save_path, tag_file, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for a specific experiment.")
    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--benchmark_model', type=str, required=True, help='The benchmark model to use.')
    parser.add_argument('--cluster_id', type=int, required=True, help='The cluster ID to process.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, required=True, help='mode dfs or bfs')

    args = parser.parse_args()

    config = load_config(args.config)
    model_list = config.get("model_list", [])

    process_data(args.exp_name, args.benchmark_model, args.cluster_id, model_list, args.mode)


