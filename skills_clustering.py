import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import json
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Load skills
with open("skills.txt", "r") as file:
    skills = [line.strip() for line in file if line.strip()]

# Step 2: Load or generate embeddings
embedding_file = "embeddings.pkl"
if os.path.exists(embedding_file):
    print("✅ Embeddings found. Loading from file...")
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
else:
    print("🔄 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # You can change model if needed
    embeddings = model.encode(skills, show_progress_bar=True)
    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)
    print("✅ Embeddings saved.")

# Step 3: Reduce dimensions with UMAP for clustering
print("🔽 Reducing dimensions with UMAP (for clustering)...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, metric="cosine", random_state=42)
reduced_for_clustering = umap_reducer.fit_transform(embeddings)

# Step 4: Cluster using HDBSCAN
print("🔍 Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean")
cluster_labels = clusterer.fit_predict(reduced_for_clustering)

# Save results to CSV
df = pd.DataFrame({"skill": skills, "cluster": cluster_labels})
df.to_csv("clustered_skills.csv", index=False)
print(f"✅ Saved clustered skills to 'clustered_skills.csv' ({df['cluster'].nunique() - (1 if -1 in cluster_labels else 0)} clusters found)")

# Step 5: UMAP for 2D visualization
print("📉 Reducing dimensions with UMAP (2D for visualization)...")
umap_2d = umap.UMAP(n_components=2, metric="cosine", random_state=42)
reduced_2d = umap_2d.fit_transform(embeddings)

# Step 6: Plotting the clusters
print("📊 Plotting clusters...")
unique_clusters = np.unique(cluster_labels)
colors = plt.cm.get_cmap("tab20", len(unique_clusters))

plt.figure(figsize=(12, 8))
for cluster in unique_clusters:
    mask = cluster_labels == cluster
    label = f"Cluster {cluster}" if cluster != -1 else "Noise"
    plt.scatter(
        reduced_2d[mask, 0],
        reduced_2d[mask, 1],
        s=60,
        alpha=0.7,
        label=label,
        c=[colors(cluster % 20)]
    )

plt.title("Skill Clusters (HDBSCAN + UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(loc="best", fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("clustered_skills_plot.png", dpi=300)
print("✅ Plot saved as 'clustered_skills_plot.png'")


# Set how many similar skills you want for each
TOP_N = 5

# Compute cosine similarity matrix (N x N)
similarity_matrix = cosine_similarity(embeddings)

# Store enhanced skills
enhanced_skills = {}

with open("enhanced_skills.txt", "w") as f:
    # Loop through each skill
    for idx, skill in enumerate(skills):
        sim_scores = similarity_matrix[idx]
        # Get top-N similar indices (excluding self)
        top_indices = sim_scores.argsort()[-TOP_N-1:-1][::-1]
        similar = [skills[i] for i in top_indices]
        enhanced_skills[skill] = similar
        f.write(skill + "\n")
        for s in similar:
            f.write(s + "\n")

# Show sample
for skill, similars in list(enhanced_skills.items())[:10]:
    print(f"{skill} ➤ {', '.join(similars)}")


with open("enhanced_skills.json", "w") as f:
    json.dump(enhanced_skills, f, indent=2)
