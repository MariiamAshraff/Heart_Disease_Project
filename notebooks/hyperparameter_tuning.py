import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

print("\n Data is prepared for hyperparameter tuning ")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("--- Starting GridSearchCV for Hyperparameter Tuning ---")
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]
roc_auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print("\n--- Optimized Random Forest Results ---")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Optimized Accuracy: {accuracy_tuned:.4f}")
print(f"Optimized Precision: {precision_tuned:.4f}")
print(f"Optimized Recall: {recall_tuned:.4f}")
print(f"Optimized F1-Score: {f1_tuned:.4f}")
print(f"Optimized ROC AUC Score: {roc_auc_tuned:.4f}")
print("-" * 50)

print("\n Hyperparameter tuning is complete. The best performing model has been identified and evaluated ")