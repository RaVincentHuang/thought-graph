import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib as mpl
from matplotlib import gridspec

def plot_combined_heatmap(unweighted, weighted, model_list, title_left, title_right, save_path, vmin, vmax, cmap="coolwarm"):
    mapping = {
        "gpt4o": "GPT-4o",
        "gpt35": "GPT-3.5",
        "gpt4": "GPT-4",
        "claude": "Claude-S-3.5",
        "qwen": "Qwen-ml",
        "glm4": "Glm-4-plus"
    }
    
    desired_order = ["GPT-4o", "GPT-4", "GPT-3.5", "Qwen-ml", "Glm-4-plus", "Claude-S-3.5"]
    full_names = [mapping.get(model, model) for model in model_list]
    model_indices = {model: i for i, model in enumerate(full_names)}

    order_indices = []
    for model in desired_order:
        if model in model_indices:
            order_indices.append(model_indices[model])
    
    unweighted = unweighted[order_indices, :][:, order_indices]
    weighted = weighted[order_indices, :][:, order_indices]
    ordered_names = [name for name in desired_order if name in full_names]

    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.2)

    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(unweighted,
                ax=ax0,
                xticklabels=ordered_names,
                yticklabels=ordered_names,
                annot=True,
                annot_kws={'size': 12},
                fmt='.3f',
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                cbar=False,
                square=True)
    
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0, fontsize=16)
    
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(weighted,
                ax=ax1,
                xticklabels=ordered_names,
                yticklabels=[],  
                annot=True,
                annot_kws={'size': 12},
                fmt='.3f',
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                cbar=False,
                square=True)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    
    ax_cbar = fig.add_subplot(gs[2])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=14)  
    
    pos_heatmap = ax0.get_position()
    pos_cbar = ax_cbar.get_position()
    
    new_pos = [pos_cbar.x0, pos_heatmap.y0,  
               pos_cbar.width, pos_heatmap.height]  
    ax_cbar.set_position(new_pos)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot combined KL divergence heatmap")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--benchmark_model', type=str, required=True)
    parser.add_argument('--cluster_num', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    result_dir = f"processed_data/blocksworld/{args.exp_name}_results_{args.benchmark_model}/{args.mode}_graph_{args.cluster_num}/kl_results"
    
    kl_matrix = np.load(os.path.join(result_dir, "kl_matrix.npy"))
    kl_matrix_weighted = np.load(os.path.join(result_dir, "kl_matrix_weighted.npy"))
    model_list = np.load(os.path.join(result_dir, "model_list.npy"))
    global_min = min(kl_matrix.min(), kl_matrix_weighted.min())
    global_max = max(kl_matrix.max(), kl_matrix_weighted.max())
    combined_save_path = os.path.join(result_dir, "kl_heatmap_combined.png")
    plot_combined_heatmap(kl_matrix, kl_matrix_weighted, model_list,
                          title_left="KL Divergence Heatmap (Unweighted)",
                          title_right="KL Divergence Heatmap (Weighted)",
                          save_path=combined_save_path,
                          vmin=global_min, vmax=global_max)