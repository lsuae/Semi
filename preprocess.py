import numpy as np
import os
from sklearn.manifold import TSNE
import json

def process_data(dataset_name):
    # Path to your features
    feat_path = f"Data/{dataset_name}/{dataset_name}_features.npy"
    label_path = f"Data/{dataset_name}/{dataset_name}_labels.npy"
    
    if not os.path.exists(feat_path):
        print(f"Skipping {dataset_name}, file not found.")
        return

    # Load data
    feats = np.load(feat_path)
    labels = np.load(label_path)
    print(f"Loaded {dataset_name}: {feats.shape}")

    # t-SNE reduction (512 -> 2 dimensions)
    tsne = TSNE(n_components=2, random_state=42)
    low_dim_data = tsne.fit_transform(feats)

    # Save as JSON for web use
    output = []
    for i in range(len(low_dim_data)):
        output.append({
            "x": float(low_dim_data[i, 0]),
            "y": float(low_dim_data[i, 1]),
            "label": int(labels[i])
        })
    
    with open(f"Data/{dataset_name}/{dataset_name}_tsne.json", "w") as f:
        json.dump(output, f)
    print(f"Saved {dataset_name}_tsne.json")

if __name__ == "__main__":
    datasets = ["food101", "eurosat", "stl10", "cifar100"]
    for ds in datasets:
        process_data(ds)