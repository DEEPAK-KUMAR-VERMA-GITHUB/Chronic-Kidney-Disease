"""
Chronic Kidney Disease (CKD) Prediction Model
Features advanced preprocessing, ensemble modeling, and clinical rule integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, precision_recall_curve, f1_score)
from imblearn.over_sampling import SMOTE
import logging
import os
import joblib
from typing import Dict, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CKDPredictor:
    """Main class for CKD prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.clinical_thresholds = {
            'sc': {'threshold': 1.2, 'weight': 2.5},
            'bu': {'threshold': 18.0, 'weight': 2.0},
            'hemo': {'threshold': 13.0, 'weight': 1.5}
        }
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess the input data"""
        try:
            # Drop ID column if exists
            data = data.drop(columns=['id'] if 'id' in data.columns else [])
            
            # Handle classification column
            data['classification'] = data['classification'].str.strip().str.lower()
            data['classification'] = data['classification'].map({'ckd': 1, 'notckd': 0})
            
            # Handle numerical columns
            numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 
                            'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
            for col in numerical_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col].fillna(data[col].median())
            
            # Handle categorical columns
            categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 
                              'dm', 'cad', 'appet', 'pe', 'ane']
            for col in categorical_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].mode()[0])
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
            
            # Scale numerical features
            data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
            
            logging.info("Data preprocessing completed successfully")
            return data, self.label_encoders
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def train_model(self, X: pd.DataFrame, y: pd.Series, use_smote: bool = True) -> None:
        """Train the CKD prediction model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Apply SMOTE for class balancing
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Evaluate model
            self._evaluate_model(X_test, y_test)
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise
            
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate model performance"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Print results
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
    def _plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess input data
        X_processed = X.copy()
        for col in X.columns:
            if col in self.label_encoders:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Make prediction
        return self.model.predict(X_processed)
        
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, path)
        logging.info(f"Model saved to {path}")
        
    @staticmethod
    def load_model(path: str) -> 'CKDPredictor':
        """Load a trained model"""
        model_data = joblib.load(path)
        predictor = CKDPredictor()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.label_encoders = model_data['label_encoders']
        predictor.feature_importance = model_data['feature_importance']
        return predictor

def main():
    """Main execution function"""
    try:
        # Initialize predictor
        predictor = CKDPredictor()
        
        # Load data
        data = pd.read_csv("kidney_disease.csv")
        
        # Preprocess data
        processed_data, _ = predictor.preprocess_data(data)
        
        # Split features and target
        X = processed_data.drop('classification', axis=1)
        y = processed_data['classification']
        
        # Train model
        predictor.train_model(X, y)
        
        # Save model
        predictor.save_model('ckd_model.joblib')
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='importance',
            y='feature',
            data=predictor.feature_importance.head(10)
        )
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()