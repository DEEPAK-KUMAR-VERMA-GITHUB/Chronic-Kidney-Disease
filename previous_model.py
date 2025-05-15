import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import sys
import joblib
import json
import datetime

# Load the dataset
data_url = "kidney_disease.csv"
try:
    data = pd.read_csv(data_url)
except FileNotFoundError:
    print("Error: Dataset file not found")
    sys.exit(1)

# Display the first few rows and general information about the dataset
print(data.head())
print(data.info())

# Check for missing values in the dataset
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Drop irrelevant columns
data_clean = data.drop(columns=['id'])

# Fill missing values for numerical columns with median
for col in data_clean.select_dtypes(include=['float64', 'int64']).columns:
    data_clean[col] = data_clean[col].fillna(data_clean[col].median())

# Fill missing values for categorical columns with mode
for col in data_clean.select_dtypes(include=['object']).columns:
    data_clean[col] = data_clean[col].fillna(data_clean[col].mode()[0])

# Encode categorical columns using Label Encoding
label_encoders = {}
for col in data_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_clean[col] = le.fit_transform(data_clean[col])
    label_encoders[col] = le

# Verify the cleaned data
missing_values_post = data_clean.isnull().sum()
print("Missing Values After Cleaning:\n", missing_values_post)

# Split data into features and target variable
X = data_clean.drop(columns=['classification'])
y = data_clean['classification']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Add SMOTE for handling imbalanced data
# smote = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")

# Store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    if name == 'Random Forest':
        model = best_rf_model  # Use the best model from grid search
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.2f}")

# Add cross-validation for each model
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} CV Scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Voting Classifier
voting_clf = VotingClassifier(estimators=list(models.items()), voting='soft')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {voting_accuracy:.2f}")

# Create a DataFrame for results
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
results_df.loc['Voting Classifier'] = voting_accuracy
print(results_df)

# Classification report
print(classification_report(y_test, y_pred_voting))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not CKD', 'CKD'], yticklabels =['Not CKD', 'CKD'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plot the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='classification', data=data)
plt.title('Distribution of Target Variable (CKD vs Not CKD)')
plt.xlabel('Classification (0: Not CKD, 1: CKD)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not CKD', 'CKD'])
plt.show()

# Model Accuracy Comparison
plt.figure(figsize=(10, 6))
results_df['Accuracy'].plot(kind='bar', color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.axhline(y=0.93, color='r', linestyle='--', label='Target Accuracy (93%)')
plt.legend()
plt.show()

# Feature Importance
feature_importances = best_rf_model.feature_importances_
features = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Add feature selection using Random Forest
selector = SelectFromModel(best_rf_model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Add probability calibration
calibrated_clf = CalibratedClassifierCV(best_rf_model, cv=2)
calibrated_clf.fit(X_train, y_train)

# Add ROC-AUC score
y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Validate data types and ranges
def validate_data(df):
    required_columns = ['classification']  # Add all required columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns")
    
    if df['classification'].nunique() != 2:
        raise ValueError("Target variable should be binary")

# Save model metadata
model_metadata = {
    'training_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'model_version': '1.0',
    'feature_columns': list(X.columns),
    'performance_metrics': {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'feature_importance': dict(zip(X.columns, best_rf_model.feature_importances_))
    },
    'preprocessing_steps': {
        'scaling': 'StandardScaler',
        'encoding': 'LabelEncoder',
        'resampling': 'SMOTE'
    }
}

# Save everything
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)

joblib.dump(best_rf_model, 'ckd_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

def plot_feature_selection_impact(X_train, X_train_selected, y_train):
    plt.figure(figsize=(15, 5))
    
    # Number of features comparison
    plt.subplot(1, 2, 1)
    features = pd.DataFrame({
        'Original': [X_train.shape[1]],
        'Selected': [X_train_selected.shape[1]]
    }).melt()
    sns.barplot(x='variable', y='value', data=features)
    plt.title('Number of Features')
    plt.ylabel('Count')
    
    # Performance comparison
    plt.subplot(1, 2, 2)
    scores = {
        'Original Features': cross_val_score(
            best_rf_model, X_train, y_train, cv=5
        ).mean(),
        'Selected Features': cross_val_score(
            best_rf_model, X_train_selected, y_train, cv=5
        ).mean()
    }
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    plt.title('Model Performance')
    plt.ylabel('Cross-validation Score')
    
    plt.tight_layout()
    plt.show()

plot_feature_selection_impact(X_train, X_train_selected, y_train)

# After your existing model evaluations, add these visualizations
def plot_model_metrics(y_test, predictions_dict):
    metrics_dict = {}
    
    for model_name, y_pred in predictions_dict.items():
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_dict[model_name] = {
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'Accuracy': accuracy_score(y_test, y_pred)
        }
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16)
    
    # Plot each metric
    metrics_df['Precision'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Precision')
    axes[0,0].set_ylim([0, 1])
    
    metrics_df['Recall'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('Recall')
    axes[0,1].set_ylim([0, 1])
    
    metrics_df['F1-Score'].plot(kind='bar', ax=axes[1,0], color='lightcoral')
    axes[1,0].set_title('F1-Score')
    axes[1,0].set_ylim([0, 1])
    
    metrics_df['Accuracy'].plot(kind='bar', ax=axes[1,1], color='plum')
    axes[1,1].set_title('Accuracy')
    axes[1,1].set_ylim([0, 1])
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return metrics_df

# Collect predictions from all models
predictions_dict = {
    'Logistic Regression': models['Logistic Regression'].predict(X_test),
    'Random Forest': best_rf_model.predict(X_test),
    'Gradient Boosting': models['Gradient Boosting'].predict(X_test),
    'Voting Classifier': voting_clf.predict(X_test)
}

# Plot the metrics
metrics_df = plot_model_metrics(y_test, predictions_dict)
print("\nDetailed Metrics Comparison:")
print(metrics_df)

# Generate support plot
def plot_support_distribution():
    support_data = pd.DataFrame({
        'Class': ['Not CKD', 'CKD'],
        'Support': [
            sum(y_test == 0),
            sum(y_test == 1)
        ]
    })
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Class', y='Support', data=support_data, palette='Set2')
    plt.title('Class Distribution in Test Set')
    plt.ylabel('Number of Samples')
    plt.show()

plot_support_distribution()

# Determine and save the best model based on F1-Score
best_model_name = metrics_df['F1-Score'].idxmax()
best_model = None

if best_model_name == 'Random Forest':
    best_model = best_rf_model
elif best_model_name == 'Voting Classifier':
    best_model = voting_clf
else:
    best_model = models[best_model_name]

print(f"\nBest performing model: {best_model_name}")

# Save the best model for UI usage
model_package = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_columns': list(X.columns),
    'model_name': best_model_name,
    'metrics': metrics_df.loc[best_model_name].to_dict()
}

# Save the complete model package
joblib.dump(model_package, 'best_ckd_model_package.joblib')

print("\nModel package saved as 'best_ckd_model_package.joblib'")
print("This package contains the model, scaler, encoders, and necessary metadata for UI implementation.")

# Print detailed performance report of the best model
print("\nDetailed Performance Report of Best Model:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))