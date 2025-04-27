import faiss
import h5py
import numpy as np
import pandas as pd
import os
import time
import psutil
import matplotlib.pyplot as plt
import requests

def download_data(urls):
    os.makedirs("data", exist_ok=True)

    for name, url in urls.items():
        path = f"data/{name}.hdf5"
        if not os.path.exists(path):
            print(f"[Download] Fetching {name}...")
            r = requests.get(url, allow_redirects=True)
            with open(path, "wb") as f:
                f.write(r.content)
        else:
            print(f"[Download] {name} already exists.")

def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        train = np.array(f['train'])
        test = np.array(f['test'])
        neighbors = np.array(f['neighbors'])
    return train, test, neighbors

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)  # in MB

def build_hnsw_index(train_data, M, efConstruction):
    d = train_data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
   
    t0 = time.time()
    index.add(train_data)
    t1 = time.time()

    build_time = t1 - t0
    mem_usage = measure_memory_usage()

    return index, build_time, mem_usage

def evaluate_index(index, test_data, true_neighbors, efSearch, k=1):
    index.hnsw.efSearch = efSearch
    num_queries = test_data.shape[0]
    
    t0 = time.time()
    distances, indices = index.search(test_data, k)
    t1 = time.time()
    total_latency = t1 - t0
    avg_latency = total_latency / num_queries
    throughput = num_queries / total_latency  # QPS

    correct = np.sum(indices[:, 0] == true_neighbors[:, 0])
    recall = correct / num_queries # Recall @ 1

    return recall, avg_latency, throughput

def experiment(train_data, test_data, neighbors, params_grid):
    results = []
    for M in params_grid['M']:
        for efC in params_grid['efConstruction']:
            for efS in params_grid['efSearch']:
                    print(f"[Running] M={M}, efC={efC}, efS={efS}")
                    
                    index, build_time, mem_usage = build_hnsw_index(train_data, M, efC)
                    recall, avg_latency, throughput = evaluate_index(index, test_data, neighbors, efS)
                    
                    results.append({
                        "M": M,
                        "efConstruction": efC,
                        "efSearch": efS,
                        "recall": recall,
                        "avg_latency_ms": avg_latency * 1000,
                        "throughput_qps": throughput,
                        "build_time_s": build_time,
                        "memory_mb": mem_usage
                    })
    return results

def plot_results(result_filepath, param_x, param_y, metric, save_to=None):
    results = pd.read_csv(result_filepath)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(results[param_x], results[param_y], c=results[metric], cmap='viridis', s=50, edgecolors='k', alpha=0.7)
    plt.colorbar(scatter, label=metric)
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title(f"{metric} for {param_x} vs. {param_y}")
    if save_to:
        plt.savefig(save_to, bbox_inches='tight')
        print(f"[Plot] Saved to {save_to}")
    plt.show()

def run_full_evaluation():
    # Load dataset
    # TODO: Choose more dataset that have what Prof. suggests in the proposal feedback (size of dataset, dimensions)
    urls = {
        "sift-128": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    }
    download_data(urls)

    # Define parameter grid
    # TODO: Change these parameters for our experiment - not sure what are the values we should test
    # TODO: Figure out how to add M0 to our experiment - from my research, FAISS does not allow us to config this parameter (not sure if there's any hack)
    params_grid = {
        "M": [8, 16, 32],
        "efConstruction": [100, 200, 400],
        "efSearch": [10, 50, 100, 200],
        # "M0": [None, 2, 4]
    }

    for dataset_name in urls:
        print(f'[Experiment] Dataset: {dataset_name}')
        train, test, neighbors = load_hdf5(f'data/{dataset_name}.hdf5')

        results = experiment(train, test, neighbors, params_grid)

        # Save results
        os.makedirs("results", exist_ok=True)
        result_filepath = f'results/{dataset_name}_results.csv'

        result_df = pd.DataFrame(results)
        result_df.to_csv(result_filepath, index=False)
        print(result_df)

    # Load result + Plot some example trends
    # TODO: Plot more graphs (Might want to concat results between dataset before plotting)
    # TODO: Customize this function (potentially, showing metrics of adjusting more two parameters - like using 3D graph, or adding more dimensiosn using point size/color/etc.)
    plot_results("results/sift-128_results.csv", "M", "efSearch", "recall")
    
if __name__ == "__main__":
    run_full_evaluation()
