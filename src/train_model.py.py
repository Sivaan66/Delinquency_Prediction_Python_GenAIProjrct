import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE # This import should now work!
from collections import Counter

# Load the dataset
file_path = r"C:\Users\prave\OneDrive\Desktop\Project TATA IQ.Geldium\PythonEDA\EDACleaned_Delinquency_Dataset.xlsx"
try:
    df = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Initial data inspection
print("\nDataset Head:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDataset Description:")
print(df.describe())

# Corrected target column name based on previous interaction
target_column = 'Delinquent_Account'

if target_column in df.columns:
    print(f"\nDistribution of '{target_column}':")
    print(df[target_column].value_counts())
    print(f"Percentage of delinquent accounts (1): {df[target_column].mean() * 100:.2f}%")
else:
    print(f"Error: Target column '{target_column}' not found in the dataset.")
    exit()

# Separate features (X) and target (y)
# Exclude 'Customer_ID' as it's an identifier, not a feature
X = df.drop(columns=['Customer_ID', target_column])
y = df[target_column]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\nIdentified Numerical Features: {numerical_features}")
print(f"Identified Categorical Features: {categorical_features}")

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- SMOTE for handling imbalanced data ---
# This step should now execute successfully due to imblearn installation
print(f"\nOriginal training target distribution: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print(f"Resampled training target distribution: {Counter(y_train_resampled)}")
# --- End SMOTE ---

# Train the RandomForestClassifier model
# Using the resampled data (X_train_resampled, y_train_resampled) for training
# class_weight='balanced' is generally not needed when using SMOTE, as the data is already balanced.
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

print("\nModel Training Complete.")

# Evaluate the model
y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1] # Probability of being delinquent (class 1)

print("\n--- Model Evaluation ---")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")

pr_auc = average_precision_score(y_test, y_prob)
print(f"Average Precision (AUPRC): {pr_auc:.4f}")

# Plotting Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (AUPRC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('precision_recall_curve.png')

# Plotting ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png')

# Feature Importance
if hasattr(model, 'feature_importances_'):
    full_feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': full_feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\n--- Top 10 Feature Importances (indicating risk sensitivity) ---")
    print(feature_importance_df.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances for Delinquency Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importances.png')

# Flagging Delinquent Customers
results_df = df.loc[X_test.index].copy()
results_df['predicted_is_delinquent'] = y_pred
results_df['delinquency_probability'] = y_prob

flagging_threshold = 0.5 # Default threshold, can be adjusted based on business needs

results_df['flagged_as_risk_sensitive'] = (results_df['delinquency_probability'] >= flagging_threshold).astype(int)

print("\n--- Sample of Customers Flagged as Risk-Sensitive ---")
print(results_df[results_df['flagged_as_risk_sensitive'] == 1].head())

flagged_customers_count = len(results_df[results_df['flagged_as_risk_sensitive'] == 1])
print(f"\nTotal customers flagged as risk-sensitive by the model: {flagged_customers_count}")

results_df.to_csv('predicted_delinquent_accounts_with_flags.csv', index=False)
