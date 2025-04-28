import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def load_results(result_filepaths):
    """
    Helper function to load one or multiple CSV files into a single DataFrame.
    """
    if isinstance(result_filepaths, str):
        result_filepaths = [result_filepaths]  # Convert to list

    dfs = []
    for filepath in result_filepaths:
        df = pd.read_csv(filepath)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df 

def plot_metric_vs_param(result_filepaths, x_param, metric, split_by='dataset', hue_param='dimension', save_dir=None):
    """
    Line plot: metric vs x_param, with lines split by 'split_by' and hue by 'hue_param'.
    """
    results_df = load_results(result_filepaths)

    plt.figure(figsize=(10, 7))
    sns.lineplot(data=results_df, x=x_param, y=metric, hue=hue_param, style=split_by, markers=True, dashes=False)
    plt.xlabel(x_param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {x_param}")
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}_vs_{x_param}.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")

def plot_scaling(result_filepaths, metric, save_dir=None):
    """
    Line plot: metric vs dataset size.
    Assumes a 'train_size' column exists in results_df.
    """
    results_df = load_results(result_filepaths)

    plt.figure(figsize=(10, 7))
    sns.lineplot(data=results_df, x='train_size', y=metric, hue='dimension', markers=True, dashes=False)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size (log scale)')
    plt.ylabel(metric)
    plt.title(f"{metric} vs Dataset Size")
    plt.grid(True, which="both", ls="--")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}_vs_dataset_size.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")

def plot_3d_surface(result_filepaths, x_param, y_param, metric, save_dir=None):
    """
    3D Surface plot: metric over (x_param, y_param).
    """
    results_df = load_results(result_filepaths)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z data
    X = results_df[x_param].values
    Y = results_df[y_param].values
    Z = results_df[metric].values
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(14, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, 20, cmap='viridis')
    
    plt.colorbar(contour, label=metric)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"{metric} over {x_param} and {y_param}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}_over_{x_param}_{y_param}.png")
        plt.savefig(save_path)
        print(f"[Plot] Saved to {save_path}")

if __name__ == "__main__":
    # List of result file(s)
    result_filepaths = ['results/sift-128_results.csv']  # can be list of multiple files

    # --------------------------
    # 1. Dataset size scaling:
    plot_scaling(result_filepaths, metric='build_time_s', save_dir="plots/scaling")
    plot_scaling(result_filepaths, metric='memory_mb', save_dir="plots/scaling")

    # --------------------------
    # 2. Dimension effects:
    plot_metric_vs_param(result_filepaths, x_param='dimension', metric='memory_mb', split_by='dataset', save_dir="plots/dimension")
    plot_metric_vs_param(result_filepaths, x_param='dimension', metric='avg_latency_ms', split_by='dataset', save_dir="plots/dimension")

    # --------------------------
    # 3. efConstruction tuning:
    plot_metric_vs_param(result_filepaths, x_param='efConstruction', metric='build_time_s', split_by='dataset', save_dir="plots/ef_construct")
    plot_metric_vs_param(result_filepaths, x_param='efConstruction', metric='memory_mb', split_by='dataset', save_dir="plots/ef_construct")
    plot_metric_vs_param(result_filepaths, x_param='efConstruction', metric='recall', split_by='dataset', save_dir="plots/ef_construct")

    # --------------------------
    # 4. efSearch tuning:
    plot_metric_vs_param(result_filepaths, x_param='efSearch', metric='recall', split_by='dataset', save_dir="plots/ef_search")
    plot_metric_vs_param(result_filepaths, x_param='efSearch', metric='avg_latency_ms', split_by='dataset', save_dir="plots/ef_search")

    # --------------------------
    # 5. M parameter tuning:
    plot_metric_vs_param(result_filepaths, x_param='M', metric='build_time_s', split_by='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M', metric='memory_mb', split_by='dataset', save_dir="plots/M")
    plot_metric_vs_param(result_filepaths, x_param='M', metric='recall', split_by='dataset', save_dir="plots/M")

    # --------------------------
    # 6. 3D Surface plots for combined effects:
    plot_3d_surface(result_filepaths, x_param='efSearch', y_param='M', metric='recall', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efSearch', y_param='M', metric='avg_latency_ms', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction', y_param='M', metric='build_time_s', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction', y_param='M', metric='memory', save_dir="plots/combined")
    plot_3d_surface(result_filepaths, x_param='efConstruction', y_param='M', metric='recall', save_dir="plots/combined")

