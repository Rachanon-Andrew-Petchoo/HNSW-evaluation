import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from adjustText import adjust_text
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def load_results(result_filepaths):
    if isinstance(result_filepaths, str):
        result_filepaths = [result_filepaths]
    dfs = [pd.read_csv(filepath) for filepath in result_filepaths]
    df = pd.concat(dfs, ignore_index=True)

    # Normalize for graphs that plots "dimension", "dataset size"
    df['build_time_s_per_point'] = df['build_time_s'] / df['train_size']
    df['memory_mb_per_point'] = df['memory_mb'] / df['train_size']
    df['build_time_s_per_dim'] = df['build_time_s'] / df['dimension']
    df['memory_mb_per_dim'] = df['memory_mb'] / df['dimension']

    return df


def plot_metric_vs_param(result_filepaths, x_param, metric, hue_param=None, save_dir=None):
    results_df = load_results(result_filepaths)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", 5)
    if hue_param:
        sns.lineplot(data=results_df, x=x_param, y=metric, hue=hue_param,
                 markers=True, dashes=False, palette=palette)
    else: # Combine datasets
        sns.lineplot(data=results_df, x=x_param, y=metric, markers=True)
    plt.xlabel(x_param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {x_param}")
    plt.grid(True)

    # Add text labels
    if hue_param:
        texts = []
        for line in results_df.groupby([hue_param]):
            group = line[1].sort_values(by=x_param)
            last = group.iloc[-1]
            label = f"{line[0][0]}"
            texts.append(plt.text(last[x_param], last[metric], label, fontsize=9))
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}_vs_{x_param}.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")

def plot_2_metric_vs_param(result_filepaths, x_param, y_param, metric, save_dir=None):
    results_df = load_results(result_filepaths)

    aggregated = (
        results_df
        .groupby([x_param, y_param])[metric]
        .mean()
        .reset_index()
    )
    X = aggregated[x_param].values
    Y = aggregated[y_param].values
    Z = aggregated[metric].values

    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(14, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, 20, cmap='viridis')
    plt.colorbar(contour, label=metric)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"{metric} over {x_param} and {y_param}")
    plt.grid(True, which="both", ls="--")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{metric}_over_{x_param}_{y_param}.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")


def plot_3_metric_vs_param(result_filepaths, x_param, y_param, z_param, metric, save_dir=None):
    """
    Plots a 3D scatter plot with color mapping for a 4th dimension (metric).
    Averages metric values for duplicate (x, y, z) points.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    results_df = load_results(result_filepaths)

    # Aggregate by x, y, z using mean for the metric.
    aggregated = results_df.groupby([x_param, y_param, z_param])[
        metric].mean().reset_index()
    X = aggregated[x_param].values
    Y = aggregated[y_param].values
    Z = aggregated[z_param].values
    M = aggregated[metric].values

    scatter = ax.scatter(X, Y, Z, c=M, cmap='viridis', marker='x', s=50)

    ax.set_title(
        f"{x_param}, {y_param}, {z_param} → median {metric} (color)")
    ax.set_xlabel(f"{x_param}")
    ax.set_ylabel(f"{y_param}")
    ax.set_zlabel(f"{z_param}")

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(f"median {metric}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{metric}_over_{x_param}_{y_param}_{z_param}.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")


if __name__ == "__main__":
    result_filepaths = ['results/sift-128_results.csv', 'results/mnist_results.csv',
                        'results/last.fm_results.csv', 'results/nytimes_results.csv', 'results/coco-i2i_results.csv']

    plot_metric_vs_param(result_filepaths, x_param='train_size',
                         metric='memory_mb_per_dim', save_dir="plots/train_size")
    plot_metric_vs_param(result_filepaths, x_param='train_size',
                         metric='build_time_s_per_dim', save_dir="plots/train_size")

    plot_metric_vs_param(result_filepaths, x_param='dimension',
                         metric='memory_mb_per_point', save_dir="plots/dimension")
    plot_metric_vs_param(result_filepaths, x_param='dimension',
                         metric='build_time_s_per_point', save_dir="plots/dimension")
    plot_metric_vs_param(result_filepaths, x_param='dimension',
                         metric='avg_latency_ms', save_dir="plots/dimension")

    plot_metric_vs_param(result_filepaths, x_param='efConstruction',
                         metric='build_time_s', hue_param='dataset', save_dir="plots/ef_construct")
    plot_metric_vs_param(result_filepaths, x_param='efConstruction',
                         metric='memory_mb', hue_param='dataset', save_dir="plots/ef_construct")
    plot_metric_vs_param(result_filepaths, x_param='efConstruction',
                         metric='recall', hue_param='dataset', save_dir="plots/ef_construct")

    plot_metric_vs_param(result_filepaths, x_param='efSearch',
                         metric='recall', hue_param='dataset', save_dir="plots/ef_search")
    plot_metric_vs_param(result_filepaths, x_param='efSearch',
                         metric='memory_mb', hue_param='dataset', save_dir="plots/ef_search")
    plot_metric_vs_param(result_filepaths, x_param='efSearch',
                         metric='avg_latency_ms', hue_param='dataset', save_dir="plots/ef_search")

    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='build_time_s', hue_param='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='memory_mb', hue_param='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='recall', hue_param='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='avg_latency_ms', hue_param='dataset', save_dir="plots/M")

    plot_2_metric_vs_param(result_filepaths, x_param='dimension', y_param='M',
                           metric='build_time_s_per_point', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='dimension', y_param='efConstruction',
                           metric='build_time_s_per_point', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='efConstruction', y_param='M', 
                           metric='build_time_s', save_dir="plots/3D")

    plot_2_metric_vs_param(result_filepaths, x_param='dimension', y_param='M', 
                           metric='memory_mb_per_point', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='dimension', y_param='efConstruction', 
                           metric='memory_mb_per_point', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='efConstruction', y_param='M', 
                           metric='memory_mb', save_dir="plots/3D")

    plot_2_metric_vs_param(result_filepaths, x_param='efSearch', y_param='M', 
                           metric='recall', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='efConstruction', y_param='M', 
                           metric='recall', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='efConstruction', y_param='efSearch', 
                           metric='recall', save_dir="plots/3D")

    plot_2_metric_vs_param(result_filepaths, x_param='efSearch', y_param='M',
                           metric='avg_latency_ms', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='efSearch', y_param='dimension',
                           metric='avg_latency_ms', save_dir="plots/3D")
    plot_2_metric_vs_param(result_filepaths, x_param='dimension', y_param='M',
                           metric='avg_latency_ms', save_dir="plots/3D")
    
    plot_2_metric_vs_param(result_filepaths, x_param='efConstruction', y_param='M',
                           metric='throughput_qps', save_dir="plots/3D")

    plot_3_metric_vs_param(result_filepaths, x_param='efSearch', y_param='efConstruction',z_param='M', 
                           metric='build_time_s', save_dir="plots/4D")
    plot_3_metric_vs_param(result_filepaths, x_param='efSearch', y_param='efConstruction',z_param='M', 
                           metric='memory_mb', save_dir="plots/4D")
    plot_3_metric_vs_param(result_filepaths, x_param='efSearch', y_param='efConstruction',z_param='M', 
                           metric='avg_latency_ms', save_dir="plots/4D")
    plot_3_metric_vs_param(result_filepaths, x_param='efSearch', y_param='efConstruction',z_param='M', 
                           metric='throughput_qps', save_dir="plots/4D")
    plot_3_metric_vs_param(result_filepaths, x_param='efSearch', y_param='efConstruction',z_param='M', 
                           metric='recall', save_dir="plots/4D")
