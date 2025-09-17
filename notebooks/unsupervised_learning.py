import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data_path = 'd:/MCsoft Course AI&ML/Heart_Disease_Project/data/HeartDiseaseTrain-Test.csv'
try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file was not found at {data_path}. Please check the file path.")
    exit()

categorical_cols = [
    'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
    'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy',
    'thalassemia'
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


numerical_cols = [
    'age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak'
]
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

X_unsupervised = df_encoded.drop('target', axis=1)

print("\nData is prepared for unsupervised learning.")
print("-" * 50)

print("--- Starting K-Means Clustering ---")
wcss = []  
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_unsupervised)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_unsupervised)

print(f"K-Means clustering complete with {optimal_k} clusters.")
print("-" * 50)

print("--- Starting Hierarchical Clustering ---")

sample_size = 50
df_sample = X_unsupervised.sample(n=sample_size, random_state=42)
linked = linkage(df_sample, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

print(" Hierarchical clustering and dendrogram analysis complete ")