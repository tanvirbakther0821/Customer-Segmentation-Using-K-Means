import pandas as pd

# Simulate the Mall Customer Segmentation dataset
data = {
    'CustomerID': range(1, 11),
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
    'Annual Income (k$)': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

mall_df = pd.DataFrame(data)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Data Preprocessing
df = mall_df.copy()

# Encode 'Gender' as numerical (Male = 1, Female = 0)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Select features for clustering
features = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize features to standard scale (mean = 0, std = 1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features to a new DataFrame for clarity
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
# Step 1: Cluster interpretation – calculate mean statistics for each cluster
cluster_summary = df.groupby('Cluster')[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_summary['Count'] = df['Cluster'].value_counts().sort_index()
cluster_summary
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: 2D Visualization of clusters (Annual Income vs. Spending Score)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='tab10',
    s=100
)
plt.title("Customer Segments by Income vs. Spending")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
#Try different values of k (e.g. 3 and 5) and compare using the Elbow Method.b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Recreate the dataset
data = {
    'CustomerID': range(1, 11),
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
    'Annual Income (k$)': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}
df = pd.DataFrame(data)

# Preprocessing
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

features = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_df = scaler.fit_transform(features)

# Try different k values and track distortion (inertia) and silhouette scores
k_range = range(1, 10)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_df)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(scaled_df, km.labels_))
    else:
        silhouettes.append(None)

# Plot Elbow and Silhouette
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot (distortion)
axs[0].plot(k_range, inertias, marker='o')
axs[0].set_title("Elbow Method (Inertia)")
axs[0].set_xlabel("Number of clusters (k)")
axs[0].set_ylabel("Inertia (Distortion)")
axs[0].grid(True)

# Silhouette Score plot
axs[1].plot(k_range[1:], silhouettes[1:], marker='o', color='green')
axs[1].set_title("Silhouette Score")
axs[1].set_xlabel("Number of clusters (k)")
axs[1].set_ylabel("Silhouette Score")
axs[1].grid(True)

plt.tight_layout()
plt.show()
