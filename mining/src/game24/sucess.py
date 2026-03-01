import argparse
import os
import json
import pandas as pd
from jsonl2csv.jsonl2csv import data_process
from node_cluster.state_based_encoder.state_encoder import encoder
from node_cluster.state_based_encoder.cluster_id_mapping import cluster_mapping
from success_path.get_success_path import success_path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_correlation_table(save_path):
    columns = ['Cluster Count', 'Model', 'Node Influence']
    df = pd.DataFrame(columns=columns)
    excel_path = os.path.join(save_path, "correlation_table.xlsx")
    df.to_excel(excel_path, index=False)
    
    return df, excel_path

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for a specific experiment.")
    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--benchmark_model', type=str, required=True, help='The benchmark model to use.')
    parser.add_argument('--config', type=str, required=True, help='The path to the config file.')
    parser.add_argument('--num_clusters_min', type=int, required=True, help='The minimum number of clusters.')
    parser.add_argument('--num_clusters_max', type=int, required=True, help='The maximum number of clusters.')
    args = parser.parse_args()
    exp_name = args.exp_name
    benchmark_model = args.benchmark_model
    model_list = load_config(args.config).get("model_list", [])
    model_list = [model for model in model_list if model != args.benchmark_model]
    num_clusters_min = args.num_clusters_min
    num_clusters_max = args.num_clusters_max

    if not os.path.exists(os.path.join("processed_data/game24",exp_name+"_results_"+benchmark_model)):
        os.makedirs(os.path.join("processed_data/game24",exp_name+"_results_"+benchmark_model))
    required_dirs = ["state_encode", "cluster_result", "tag_pool", "visualize"]
    for dir_name in required_dirs:
        dir_path = os.path.join( "processed_data/game24", exp_name + "_results_"+benchmark_model, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    input_path = os.path.join("raw_data/game24",exp_name)
    save_path = os.path.join("processed_data/game24",exp_name+"_results_"+benchmark_model+"/tag_pool")
    log_path = os.path.join("raw_data/game24",exp_name,"result_log")
    state_nodes_path = os.path.join("processed_data/game24",exp_name+"_results_"+benchmark_model+"/state_encode/state_nodes_2-2.csv")
    visualize_save_path = os.path.join("processed_data/game24",exp_name+"_results_"+benchmark_model+"/visualize")
    node_file_path = data_process(input_path,save_path,log_path,benchmark_model)
    encoder(node_file_path, state_nodes_path)
    max_corr = -1
    cluster_corr = -1
    df_excel, excel_path = create_correlation_table(visualize_save_path)
    for i in range(num_clusters_min,num_clusters_max+1):
        num_clusters = i
        cluster_nodes_path = os.path.join("processed_data/game24",exp_name+"_results_"+benchmark_model+"/cluster_result/cluster_nodes_2-2_cluster_id_{}.csv".format(num_clusters))
        df = pd.read_csv(state_nodes_path,dtype={'similarity': str})
        df = cluster_mapping(df,num_clusters)
        df.to_csv(cluster_nodes_path,index=False)
        folder_name = os.path.splitext(os.path.basename(cluster_nodes_path))[0]

        current_visualize_save_path = os.path.join(visualize_save_path, folder_name)
        if not os.path.exists(current_visualize_save_path):
            os.makedirs(current_visualize_save_path)
        results = success_path(model_list, benchmark_model, cluster_nodes_path, current_visualize_save_path)

        model_list_ = ["gpt4o","gpt4","gpt35","qwen","glm4","claude"]
        for model in model_list:
            new_row = {
                'Cluster Count': i,
                'Model': model_mapping(model),
                'Node Influence': results[model]['correlation']
            }

            df_excel = pd.concat([df_excel, pd.DataFrame([new_row])], ignore_index=True)
        
        df_excel.to_excel(excel_path, index=False)

        corr = results["all"]['correlation']
        if corr > cluster_corr:
            cluster_corr = corr
            max_corr = i
    print("The best cluster number is: ",max_corr)



