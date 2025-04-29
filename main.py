import faiss
import h5py
import numpy as np
import pandas as pd
import os
import time
import psutil
import requests
import gc

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

def experiment(dataset_name, train_data, test_data, neighbors, params_grid, result_filepath):
    train_size, dimension = train_data.shape
    test_size = test_data.shape[0]

    for M in params_grid['M']:
        for efC in params_grid['efConstruction']:
            for efS in params_grid['efSearch']:
                    print(f"[Running] M={M}, efC={efC}, efS={efS}")
                    
                    index, build_time, mem_usage = build_hnsw_index(train_data, M, efC)
                    recall, avg_latency, throughput = evaluate_index(index, test_data, neighbors, efS)

                    result = {
                        "M": M,
                        "efConstruction": efC,
                        "efSearch": efS,
                        "recall": recall,
                        "avg_latency_ms": avg_latency * 1000,
                        "throughput_qps": throughput,
                        "build_time_s": build_time,
                        "memory_mb": mem_usage,
                        "dataset": dataset_name,
                        "dimension": dimension,
                        "train_size": train_size,
                        "test_size": test_size
                    }
                    res_df = pd.DataFrame([result])
                    res_df.to_csv(result_filepath, mode='a', header=False, index=False)

                    # Clear memory after each evaluation
                    del index, build_time, mem_usage, recall, avg_latency, throughput, result, res_df
                    gc.collect()

def run_full_evaluation():
    # Load dataset
    # TODO: Choose more dataset that have what Prof. suggests in the proposal feedback (size of dataset, dimensions)
    urls = {
        "sift-128": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "mnist": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "last.fm": "http://ann-benchmarks.com/lastfm-64-dot.hdf5",
        "nytimes": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "coco-i2i": "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5"
    }
    download_data(urls)

    # Define parameter grid
    # TODO: Change these parameters for our experiment - not sure what are the values we should test
    # TODO: Figure out how to add M0 to our experiment - from my research, FAISS does not allow us to config this parameter (not sure if there's any hack)
    params_grid = {
        "M": [4, 8, 16, 24, 32],
        "efConstruction": [50, 100, 200],
        "efSearch": [10, 50, 100, 200],
        # "M0": [None, 2, 4]
    }

    for dataset_name in urls:
        print(f'[Experiment] Dataset: {dataset_name}')

        # Split dataset
        train, test, neighbors = load_hdf5(f'data/{dataset_name}.hdf5')

        # Create CSV result file
        os.makedirs("results", exist_ok=True)
        result_filepath = f'results/{dataset_name}_results.csv'

        columns = ['M', 'efConstruction', 'efSearch', 'recall', 'avg_latency_ms', 'throughput_qps', 'build_time_s', 'memory_mb', 'dataset', 'dimension', 'train_size', 'test_size']
        with open(result_filepath, 'w') as f:
            f.write(','.join(columns) + '\n')

        # Run experiment and save result to CSV
        experiment(dataset_name, train, test, neighbors, params_grid, result_filepath)

        # Clean up memory after each dataset
        del train, test, neighbors, result_filepath, columns
        gc.collect()
    
if __name__ == "__main__":
    run_full_evaluation()
    # Load result + Plot some example trends
    # TODO: Plot more graphs (Might want to concat results between dataset before plotting)
    # TODO: Customize plot_utils.py functions (potentially, showing metrics of adjusting more two parameters - like using 3D graph, or adding more dimensiosn using point size/color/etc.)
    
