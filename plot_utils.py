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
    return pd.concat(dfs, ignore_index=True)


def plot_metric_vs_param(result_filepaths, x_param, metric, split_by='dataset', hue_param='dimension', save_dir=None):
    results_df = load_results(result_filepaths)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10")
    sns.lineplot(data=results_df, x=x_param, y=metric, hue=hue_param,
                 style=split_by, markers=True, dashes=False, palette=palette)
    plt.xlabel(x_param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {x_param}")
    plt.grid(True)

    # Add text labels
    texts = []
    for line in results_df.groupby([hue_param, split_by]):
        group = line[1].sort_values(by=x_param)
        last = group.iloc[-1]
        label = f"{line[0][0]} - {line[0][1]}"
        texts.append(plt.text(last[x_param], last[metric], label, fontsize=9))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}_vs_{x_param}.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")


def plot_scaling(result_filepaths, metric, save_dir=None):
    results_df = load_results(result_filepaths)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10")
    ax = sns.lineplot(data=results_df, x='train_size', y=metric,
                      hue='dimension', markers=True, dashes=False, palette=palette)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size (log scale)')
    plt.ylabel(metric)
    plt.title(f"{metric} vs Dataset Size")
    plt.grid(True, which="both", ls="--")

    # Add text labels
    texts = []
    for line in results_df.groupby('dimension'):
        group = line[1].sort_values(by='train_size')
        last = group.iloc[-1]
        label = str(line[0])
        texts.append(plt.text(last['train_size'],
                     last[metric], label, fontsize=9))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}_vs_dataset_size.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")


def plot_3d_surface(result_filepaths, x_param, y_param, metric, save_dir=None):
    results_df = load_results(result_filepaths)

    X = results_df[x_param].values
    Y = results_df[y_param].values
    Z = results_df[metric].values
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(14, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, 20, cmap='plasma')
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

def plot_3rd_scatter_color(result_filepaths, x_param, y_param, z_param, metric, save_dir=None):
    """
    Plots a 3D scatter plot with color mapping for a 4th dimension (metric).
    Averages metric values for duplicate (x, y, z) points.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    results_df = load_results(result_filepaths)

    # Aggregate by x, y, z using mean for the metric.
    aggregated = results_df.groupby([x_param, y_param, z_param])[metric].median().reset_index()
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

    plot_scaling(result_filepaths, metric='build_time_s',
                 save_dir="plots/scaling")
    plot_scaling(result_filepaths, metric='memory_mb',
                 save_dir="plots/scaling")

    plot_metric_vs_param(result_filepaths, x_param='dimension',
                         metric='memory_mb', split_by='dataset', save_dir="plots/dimension")
    plot_metric_vs_param(result_filepaths, x_param='dimension',
                         metric='avg_latency_ms', split_by='dataset', save_dir="plots/dimension")

    plot_metric_vs_param(result_filepaths, x_param='efConstruction',
                         metric='build_time_s', split_by='dataset', save_dir="plots/ef_construct")
    plot_metric_vs_param(result_filepaths, x_param='efConstruction',
                         metric='memory_mb', split_by='dataset', save_dir="plots/ef_construct")
    plot_metric_vs_param(result_filepaths, x_param='efConstruction',
                         metric='recall', split_by='dataset', save_dir="plots/ef_construct")

    plot_metric_vs_param(result_filepaths, x_param='efSearch',
                         metric='recall', split_by='dataset', save_dir="plots/ef_search")
    plot_metric_vs_param(result_filepaths, x_param='efSearch',
                         metric='avg_latency_ms', split_by='dataset', save_dir="plots/ef_search")

    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='build_time_s', split_by='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='memory_mb', split_by='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M',
                         metric='recall', split_by='dataset', save_dir="plots/M")

    plot_3d_surface(result_filepaths, x_param='dimension', y_param='M',
                    metric='build_time_s', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='dimension', y_param='efConstruction',
                    metric='build_time_s', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction',
                    y_param='M', metric='build_time_s', save_dir="plots/combined")

    plot_3d_surface(result_filepaths, x_param='dimension',
                    y_param='M', metric='memory_mb', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='dimension',
                    y_param='efConstruction', metric='memory_mb', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction',
                    y_param='M', metric='memory_mb', save_dir="plots/combined")

    plot_3d_surface(result_filepaths, x_param='efSearch',
                    y_param='M', metric='recall', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction',
                    y_param='M', metric='recall', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction',
                    y_param='efSearch', metric='recall', save_dir="plots/combined")

    plot_3d_surface(result_filepaths, x_param='efSearch', y_param='M',
                    metric='avg_latency_ms', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efSearch', y_param='dimension',
                    metric='avg_latency_ms', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='dimension', y_param='M',
                    metric='avg_latency_ms', save_dir="plots/combined")

    plot_3rd_scatter_color(result_filepaths, x_param='efSearch', y_param='efConstruction',
                           z_param='M', metric='avg_latency_ms', save_dir="plots/combined")
    plot_3rd_scatter_color(result_filepaths, x_param='efSearch', y_param='efConstruction',
                           z_param='M', metric='recall', save_dir="plots/combined")
    plot_3rd_scatter_color(result_filepaths, x_param='efSearch', y_param='efConstruction',
                           z_param='M', metric='build_time_s', save_dir="plots/combined")
