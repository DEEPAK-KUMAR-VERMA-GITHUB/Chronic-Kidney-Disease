"""
Chronic Kidney Disease (CKD) Prediction Model Training and Evaluation

This script implements a complete pipeline for CKD prediction:
- Data preprocessing and feature engineering
- Model training with multiple algorithms and sampling methods
- Performance evaluation and visualization
- Model persistence with metadata

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, precision_recall_curve, f1_score,
                           precision_score, recall_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
import shap
import joblib
import json
import os
import datetime
import logging
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    filename='ckd_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelConfig:
    """Configuration settings for model training"""
    def __init__(self):
        self.random_state: int = 42
        self.test_size: float = 0.2
        self.cv_folds: int = 5
        self.n_jobs: int = -1
        
        # Directory paths
        self.base_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.metrics_dir: str = os.path.join(self.base_dir, 'metrics')
        self.plots_dir: str = os.path.join(self.base_dir, 'plots')
        self.model_dir: str = os.path.join(self.base_dir, 'models')
        
        # Clinical thresholds
        self.clinical_thresholds: Dict = {
            'sc': {'threshold': 1.2, 'weight': 2.5},
            'bu': {'threshold': 18.0, 'weight': 2.0},
            'egfr': {'threshold': 65.0, 'weight': 2.5},
            'hemo': {'threshold': 13.0, 'weight': 1.5}
        }
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.metrics_dir, self.plots_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)

class EnhancedCKDEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Custom ensemble classifier combining ML predictions with clinical rules"""
    
    def __init__(self, base_model=None, threshold_config=None):
        self.base_model = base_model
        self.threshold_config = threshold_config or {
            'sc': {'threshold': 1.2, 'weight': 2.5},
            'bu': {'threshold': 18.0, 'weight': 2.0},
            'egfr': {'threshold': 65.0, 'weight': 2.5},
            'hemo': {'threshold': 13.0, 'weight': 1.5}
        }
        self.classes_ = None
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'EnhancedCKDEnsembleClassifier':
        """
        Fit the model to training data
        
        Args:
            X: Training features
            y: Target values
        """
        if self.base_model is None:
            raise ValueError("Base model must be specified")
            
        # Store feature names and classes
        self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        self.classes_ = np.unique(y)
        
        # Fit base model
        self.base_model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using both ML model and clinical rules
        
        Args:
            X: Features to predict
        """
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet")
            
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_ is None:
                raise ValueError("Feature names not available for non-DataFrame input")
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        # Get predictions
        ml_predictions = self.base_model.predict(X)
        clinical_predictions = self._apply_clinical_rules(X)
        
        # Combine predictions
        return np.where(clinical_predictions != -1, clinical_predictions, ml_predictions)
        
    def _apply_clinical_rules(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply clinical rules to make predictions
        
        Args:
            X: Features to evaluate
        """
        predictions = np.full(X.shape[0], -1)
        
        for idx, row in X.iterrows():
            score = 0
            critical_indicators = 0
            
            for feature, config in self.threshold_config.items():
                if feature not in row.index:
                    continue
                    
                try:
                    value = float(row[feature])
                    if value > config['threshold']:
                        score += config['weight']
                        critical_indicators += 1
                except (ValueError, TypeError):
                    continue
                    
            if critical_indicators >= 2 or score >= 2.5:
                predictions[idx] = 1
            elif score <= 0.5:
                predictions[idx] = 0
                
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
        """
        if not hasattr(self.base_model, 'predict_proba'):
            raise NotImplementedError("Base model doesn't support probability predictions")
            
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_ is None:
                raise ValueError("Feature names not available for non-DataFrame input")
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        # Get base probabilities
        probas = self.base_model.predict_proba(X)
        clinical_preds = self._apply_clinical_rules(X)
        
        # Adjust probabilities based on clinical rules
        for idx, pred in enumerate(clinical_preds):
            if pred != -1:
                probas[idx] = [0.1, 0.9] if pred == 1 else [0.9, 0.1]
                
        return probas
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        if hasattr(self.base_model, 'feature_importances_'):
            return self.base_model.feature_importances_
        elif hasattr(self.base_model, 'coef_'):
            return self.base_model.coef_[0]
        return None

class DataProcessor:
    """Handles data preprocessing operations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = None
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess the dataset"""
        logging.info(f"Loading data from {data_path}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            data_clean = data.drop(columns=['id'] if 'id' in data.columns else [])
            
            # Handle missing values
            for col in data_clean.select_dtypes(include=['float64', 'int64']).columns:
                data_clean[col] = data_clean[col].fillna(data_clean[col].median())
                
            for col in data_clean.select_dtypes(include=['object']).columns:
                data_clean[col] = data_clean[col].fillna(data_clean[col].mode()[0])
            
            # Encode categorical variables
            for col in data_clean.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data_clean[col] = le.fit_transform(data_clean[col])
                self.label_encoders[col] = le
            
            logging.info("Data preprocessing completed successfully")
            return data_clean, self.label_encoders
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def get_param_grids(self) -> Dict:
        """Define parameter grids for each model"""
        return {
            'KNN': {
                'model': EnhancedCKDEnsembleClassifier(
                    base_model=KNeighborsClassifier(),
                    threshold_config=self.config.clinical_thresholds
                ),
                'params': {
                    'base_model__n_neighbors': [3, 5, 7],
                    'base_model__weights': ['uniform', 'distance'],
                    'base_model__metric': ['euclidean', 'manhattan']
                }
            },
            'Random Forest': {
                'model': EnhancedCKDEnsembleClassifier(
                    base_model=RandomForestClassifier(random_state=self.config.random_state),
                    threshold_config=self.config.clinical_thresholds
                ),
                'params': {
                    'base_model__n_estimators': [100, 200],
                    'base_model__max_depth': [None, 10, 20],
                    'base_model__min_samples_split': [2, 5],
                    'base_model__min_samples_leaf': [1, 2],
                    'base_model__class_weight': ['balanced']
                }
            }
        }
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        """Train all models with different sampling methods"""
        logging.info("Starting model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Get minimum class size for SMOTE
        min_class_size = min(Counter(y_train).values())
        smote_k_neighbors = min(5, min_class_size - 1)
        
        # Prepare sampling methods
        sampling_methods = {
            'Original': (X_train, y_train),
            'SMOTE': SMOTE(
                random_state=self.config.random_state,
                k_neighbors=smote_k_neighbors
            ).fit_resample(X_train, y_train),
            'Random Under': RandomUnderSampler(
                random_state=self.config.random_state
            ).fit_resample(X_train, y_train)
        }
        
        results = {}
        best_models = {}
        
        # Train models with different sampling methods
        for sampling_name, (X_resampled, y_resampled) in sampling_methods.items():
            logging.info(f"Training with {sampling_name} sampling")
            sampling_results = self._train_sampling_method(
                X_resampled, X_test,
                y_resampled, y_test,
                sampling_name
            )
            results[sampling_name] = sampling_results['results']
            best_models[sampling_name] = sampling_results['best_models']
            
        return results, best_models
        
    def _train_sampling_method(self, X_train, X_test, y_train, y_test, sampling_name):
        """Train models for a specific sampling method"""
        results = {}
        best_models = {}
        
        for name, model_info in self.get_param_grids().items():
            try:
                # Prepare cross-validation
                n_splits = min(self.config.cv_folds, min(Counter(y_train).values()))
                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.config.random_state
                )
                
                # Train model
                grid_search = GridSearchCV(
                    estimator=model_info['model'],
                    param_grid=model_info['params'],
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=self.config.n_jobs,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = grid_search.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    **metrics
                }
                
                best_models[name] = grid_search.best_estimator_
                
                # Save plots
                self._save_model_plots(
                    grid_search.best_estimator_,
                    X_test, y_test, y_pred,
                    f"{sampling_name}_{name}"
                )
                
            except Exception as e:
                logging.error(f"Error training {name} with {sampling_name}: {str(e)}")
                continue
                
        return {'results': results, 'best_models': best_models}
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred)
        }
        
    def _save_model_plots(self, model, X_test, y_test, y_pred, model_name):
        """Generate and save visualization plots"""
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.config.plots_dir}/{model_name}_confusion_matrix.png')
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(f'{self.config.plots_dir}/{model_name}_roc_curve.png')
        plt.close()
        
        # Feature Importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance is not None:
                plt.figure(figsize=(12, 6))
                features = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
                importance_df = pd.DataFrame({'feature': features, 'importance': importance})
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                sns.barplot(data=importance_df, x='importance', y='feature')
                plt.title(f'Feature Importance - {model_name}')
                plt.tight_layout()
                plt.savefig(f'{self.config.plots_dir}/{model_name}_feature_importance.png')
                plt.close()

class ModelPersistence:
    """Handles model saving and loading"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def save_model(self, model: Any, metadata: Dict, filename: str):
        """Save model and its metadata"""
        try:
            # Save model
            model_path = os.path.join(self.config.model_dir, f'{filename}.joblib')
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = os.path.join(self.config.model_dir, f'{filename}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logging.info(f"Model saved successfully: {filename}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

def main():
    """Main execution pipeline"""
    try:
        # Initialize components
        config = ModelConfig()
        data_processor = DataProcessor(config)
        model_trainer = ModelTrainer(config)
        model_persistence = ModelPersistence(config)
        
        # Load and prepare data
        data, label_encoders = data_processor.load_and_prepare_data("kidney_disease.csv")
        
        # Split features and target
        X = data.drop(columns=['classification'])
        y = data['classification']
        
        # Train models
        results, best_models = model_trainer.train_models(X, y)
        
        # Find best model
        best_f1 = 0
        best_model_info = None
        
        for sampling_name, sampling_results in results.items():
            for model_name, model_results in sampling_results.items():
                if model_results['f1'] > best_f1:
                    best_f1 = model_results['f1']
                    best_model_info = {
                        'sampling_method': sampling_name,
                        'model_name': model_name,
                        'model': best_models[sampling_name][model_name],
                        'results': model_results
                    }
        
        if best_model_info:
            # Save best model
            metadata = {
                'feature_columns': list(X.columns),
                'sampling_method': best_model_info['sampling_method'],
                'model_name': best_model_info['model_name'],
                'performance': best_model_info['results'],
                'label_encoders': {
                    col: list(le.classes_) 
                    for col, le in label_encoders.items()
                }
            }
            
            model_persistence.save_model(
                best_model_info['model'],
                metadata,
                'best_ckd_model2'
            )
            
            logging.info("Training completed successfully")
            print("\nTraining Summary:")
            print(f"Best Model: {best_model_info['model_name']}")
            print(f"Sampling Method: {best_model_info['sampling_method']}")
            print(f"F1 Score: {best_f1:.4f}")
            
        else:
            logging.error("No valid model found")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()