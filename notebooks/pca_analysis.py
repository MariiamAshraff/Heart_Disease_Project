import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

print("\nData is prepared for PCA analysis.")
print("-" * 50)
print("--- Starting PCA Analysis ---")

pca = PCA(n_components=None)
pca.fit(X)


print("Displaying cumulative explained variance plot...")
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


optimal_components = 10
pca_final = PCA(n_components=optimal_components)
X_pca = pca_final.fit_transform(X)

print(f"\nData successfully reduced to {optimal_components} principal components.")
print(f"New dataset shape after PCA: {X_pca.shape}")
print("-" * 50)