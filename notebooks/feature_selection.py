
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

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

print("\n Data is prepared for feature selection.")
print("-" * 50)

print("--- Starting Feature Selection ---")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n Random Forest:")
print(feature_importances)

print("Displaying feature importance plot...")
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('(Feature Importance)')
plt.show()

rfe_selector = RFE(estimator=rf_model, n_features_to_select=10)
rfe_selector.fit(X, y)

selected_features_rfe = X.columns[rfe_selector.support_]
print("\n RFE:")
print(selected_features_rfe)

print("\n Successfully")