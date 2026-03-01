import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

def get_benchmark_success_paths(df):
    paths = []
    current_path = []

    for _, row in df.iterrows():
        if row['success'] and row['benchmark']:
            if row['parent_id'] == "root":
                if current_path:
                    paths.append(current_path)
                current_path = []
            if row['parent_id'] != "root":
                current_path.append(row.to_dict())
    
    if current_path:
        paths.append(current_path)
    
    return paths

def get_failed_paths(df, step=2):
    node_dict = {row['node_id']: row for _, row in df.iterrows()}

    paths = []

    for _, row in df.iterrows():
        if row['step'] == step and not row['success']:
            current_path = []
            valid_path = True
            current_node = row

            while current_node is not None:
                if current_node['cluster_id'] in [-1, -2]:
                    valid_path = False
                    break

                if current_node['step'] != 0:
                    current_path.append(current_node.to_dict())

                parent_id = current_node['parent_id']
                current_node = node_dict.get(parent_id)

            if valid_path:
                paths.append(list(reversed(current_path)))

    return paths

def count_cluster_ids(paths):
    cluster_counter = Counter()
    for path in paths:
        unique_clusters = set(item['cluster_id'] for item in path)
        cluster_counter.update(unique_clusters)
    return cluster_counter

def count_avg_values(model_list, df, cluser_counter):
    model_values = {model: {key: [] for key in cluser_counter.keys()} for model in model_list}
    benchmark_values = {key: [] for key in cluser_counter.keys()}
    overall_values = {key: [] for key in cluser_counter.keys()}

    for _, row in df.iterrows():
        if row['parent_id'] == "root":
            continue

        cluster = row['cluster_id']

        if cluster in cluser_counter.keys():
            value = row['value']

            overall_values[cluster].append(value)

            if row['benchmark']:
                benchmark_values[cluster].append(value)
            else:
                for model in model_list:
                    if row['data_src'] == model:
                        model_values[model][cluster].append(value)
    model_avg = {model: {key: (sum(values) / len(values) if values else 0) for key, values in clusters.items()} for model, clusters in model_values.items()}
    benchmark_avg = {key: (sum(values) / len(values) if values else 0) for key, values in benchmark_values.items()}
    overall_avg = {key: (sum(values) / len(values) if values else 0) for key, values in overall_values.items()}

    return model_avg, benchmark_avg, overall_avg

def count_values(model_list, df, cluser_counter):
    model_values = {model: {key: [] for key in cluser_counter.keys()} for model in model_list}
    benchmark_values = {key: [] for key in cluser_counter.keys()}
    overall_values = {key: [] for key in cluser_counter.keys()}

    for _, row in df.iterrows():
        if row['parent_id'] == "root":
            continue

        cluster = row['cluster_id']

        if cluster in cluser_counter.keys():
            value = row['value']

            overall_values[cluster].append(value)

            if row['benchmark']:
                benchmark_values[cluster].append(value)
            else:
                for model in model_list:
                    if row['data_src'] == model:
                        model_values[model][cluster].append(value)

    model_avg = model_values
    benchmark_avg = benchmark_values
    overall_avg = overall_values

    return model_avg, benchmark_avg, overall_avg

def plot_boxplot(cluster_counts, values, title, save_path):

    clusters = list(cluster_counts.keys())
    counts = [cluster_counts[cluster] for cluster in clusters]
    data = [values[cluster] for cluster in clusters]

    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i])
    sorted_clusters = [clusters[i] for i in sorted_indices]
    sorted_data = [data[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.boxplot(sorted_data, widths=0.6)

    plt.xticks(ticks=range(1, len(sorted_clusters) + 1), labels=sorted_clusters)

    for i, count in enumerate(sorted_counts, start=1):
        plt.text(i, plt.ylim()[0], f'{count}', ha='center', va='top', fontsize=9, color='red')

    plt.xlabel('Cluster')
    plt.ylabel('Values')
    plt.title(title)
    plt.grid(True)
    save_path = save_path + r"/{}_boxplot.png".format(title)
    plt.savefig(save_path)

def visualize_cluster_data(cluster_counts, avg_values,title,save_path):
    clusters = list(cluster_counts.keys())
    counts = [cluster_counts[cluster] for cluster in clusters]
    avg_vals = [avg_values[cluster] for cluster in clusters]

    plt.figure(figsize=(10, 6))
    plt.scatter(counts, avg_vals, color='blue', alpha=0.6)
    
    for i, cluster in enumerate(clusters):
        plt.text(counts[i], avg_vals[i], str(cluster), fontsize=9, ha='right')

    plt.xlabel('Count')
    plt.ylabel('Average Value')
    plt.xticks(range(min(counts), max(counts) + 1))  
    plt.title(title)
    plt.grid(True)
    save_path = save_path + r"/{}.png".format(title)
    plt.savefig(save_path)


def calculate_relation(cluster_counts, avg_values):
    common_keys = set(cluster_counts.keys()).intersection(avg_values.keys())
    sorted_keys = sorted(common_keys)
    
    counts = [cluster_counts[key] for key in sorted_keys]
    values = [avg_values[key] for key in sorted_keys]
    
    correlation_matrix = np.corrcoef(counts, values)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def calculate_covariance(cluster_counts, avg_values):
    common_keys = set(cluster_counts.keys()).intersection(avg_values.keys())
    sorted_keys = sorted(common_keys)
    
    counts = [cluster_counts[key] for key in sorted_keys]
    values = [avg_values[key] for key in sorted_keys]
    counts_variance = np.var(counts)
    covariance_matrix = np.cov(counts, values)

    correlation_matrix = np.corrcoef(counts, values)
    correlation_coefficient = correlation_matrix[0, 1]

    values_mean = np.mean(values)
    values_std = np.std(values)
    values_cv = values_std / values_mean

    return covariance_matrix[0,1] / ((counts_variance) ** 0.5), covariance_matrix[0,1]

def success_path(model_list,benchmark_model, path_nodes_path,save_path,step=-2):
    results = {}
    df = pd.read_csv(path_nodes_path)
    if step > 0:
        paths = get_failed_paths(df, step=step)
    else:
        paths = get_benchmark_success_paths(df)
    cluster_counts = count_cluster_ids(paths)
    non_benchmark_avg_values, benchmark_avg_values, all_avg_values = count_avg_values(model_list,df, cluster_counts)
    non_benchmark_values, benchmark_values, all_values = count_values(model_list,df, cluster_counts)
    print("Cluster Counts:", cluster_counts)
    print("Average Values:", benchmark_avg_values)
    for model, avg_values in non_benchmark_avg_values.items():
        print(f"Non-Benchmark Average Values for {model}:", avg_values)
    
    correlation_file_path = os.path.join(save_path, "correlations_and_stats.txt")

    benchmark_model_name = benchmark_model

    with open(correlation_file_path, 'w', encoding='utf-8') as file:
        file.write(f"Cluster Counts: {cluster_counts}\n")
        print("Cluster Counts:", cluster_counts)

        file.write(f"Benchmark Average Values: {benchmark_avg_values}\n")
        print("Average Values:", benchmark_avg_values)

        for model, avg_values in non_benchmark_avg_values.items():
            file.write(f"Non-Benchmark Average Values for {model}: {avg_values}\n")
            print(f"Non-Benchmark Average Values for {model}:", avg_values)

        benchmark_correlation = calculate_relation(cluster_counts, benchmark_avg_values)
        benchmark_covariance_changed, benchmark_covariance = calculate_covariance(cluster_counts, benchmark_avg_values)
        file.write(f"benchmark_correlation: {benchmark_correlation}\n")
        print("benchmark_correlation:", benchmark_correlation)
        results[benchmark_model_name] = {
            "correlation": benchmark_correlation,
            "covariance_changed": benchmark_covariance_changed,
            "covariance": benchmark_covariance
        }

        for model, avg_values in non_benchmark_avg_values.items():
            model_correlation = calculate_relation(cluster_counts, avg_values)
            model_covariance_changed, model_covariance = calculate_covariance(cluster_counts, avg_values)
            file.write(f"non_benchmark_correlation for {model}: {model_correlation}\n")
            print(f"non_benchmark_correlation for {model}:", model_correlation)
            results[model] = {
                "correlation": model_correlation,
                "covariance_changed": model_covariance_changed,
                "covariance": model_covariance
            }

        all_correlation = calculate_relation(cluster_counts, all_avg_values)
        all_covariance = calculate_covariance(cluster_counts, all_avg_values)
        file.write(f"all_correlation: {all_correlation}\n")
        print("all_correlation:", all_correlation)
        results["all"] = {
            "correlation": all_correlation,
            "covariance_changed": 0.0,
            "covariance": all_covariance
        }

    visualize_cluster_data(cluster_counts, benchmark_avg_values, "benchmark", save_path)
    plot_boxplot(cluster_counts, benchmark_values, "benchmark", save_path)
    
    for model, values in non_benchmark_values.items():
        plot_boxplot(cluster_counts, values, f"non_benchmark_{model}", save_path)

    for model, avg_values in non_benchmark_avg_values.items():
        visualize_cluster_data(cluster_counts, avg_values, f"non_benchmark_{model}", save_path)
    
    visualize_cluster_data(cluster_counts, all_avg_values, "all", save_path)
    plot_boxplot(cluster_counts, all_values, "all", save_path)

    return results


