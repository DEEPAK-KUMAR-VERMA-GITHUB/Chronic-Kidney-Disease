"""
Chronic Kidney Disease (CKD) Prediction Model Training and Evaluation

This script implements a comprehensive pipeline for CKD prediction using multiple models:
- Decision Tree, SVM, Random Forest, XGBoost, KNN, and Gradient Boosting
- Ensemble modeling with clinical thresholds
- Extensive visualization and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, precision_recall_curve, f1_score,
                           precision_score, recall_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
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

# Disable interactive mode
plt.ioff()

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
        
        # Model parameters for grid search
        self.param_grids = {
            'DT': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'RF': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGB': {
                'model': XGBClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'GB': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
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
            
    def get_timestamp(self) -> str:
        """Get current timestamp for file naming"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    

class EnhancedCKDEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Enhanced ensemble classifier that combines multiple models with clinical rules
    Uses weighted voting from all models plus clinical threshold rules
    """
    
    def __init__(self, models: Dict[str, BaseEstimator], threshold_config: Dict):
        self.models = models
        self.threshold_config = threshold_config
        self.classes_ = None
        self.feature_names_ = None
        self.weights = {
            'DT': 1,
            'RF': 2,
            'XGB': 2,
            'GB': 2,
            'KNN': 1,
            'SVM': 1.5,
            'Clinical': 2.5  # Higher weight for clinical rules
        }
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'EnhancedCKDEnsembleClassifier':
        """Fit all models in the ensemble"""
        self.feature_names_ = X.columns.tolist()
        self.classes_ = np.unique(y)
        
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                logging.info(f"Successfully fitted {name} model")
            except Exception as e:
                logging.error(f"Error fitting {name} model: {str(e)}")
                raise
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted voting from all models plus clinical rules"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logging.error(f"Error predicting with {name}: {str(e)}")
                predictions[name] = np.full(X.shape[0], -1)
                
        # Get clinical predictions
        clinical_preds = self._apply_clinical_rules(X)
        predictions['Clinical'] = clinical_preds
        
        # Weighted voting
        final_predictions = np.zeros(X.shape[0])
        for idx in range(X.shape[0]):
            votes = {0: 0, 1: 0}
            for name, preds in predictions.items():
                if preds[idx] != -1:  # Skip if prediction is invalid
                    votes[preds[idx]] += self.weights[name]
            final_predictions[idx] = 1 if votes[1] > votes[0] else 0
            
        return final_predictions
        
    def _apply_clinical_rules(self, X: pd.DataFrame) -> np.ndarray:
        """Apply clinical rules for prediction"""
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        predictions = np.full(X.shape[0], -1)
        
        for idx, row in X.iterrows():
            try:
                score = 0
                critical_indicators = 0
                
                # Process each clinical threshold
                for feature, config in self.threshold_config.items():
                    if feature not in row.index:
                        continue
                        
                    try:
                        value = float(row[feature])
                        threshold = float(config['threshold'])
                        
                        # Apply clinical rules
                        if feature == 'sc' and value > 1.5:
                            critical_indicators += 1
                        elif feature == 'egfr' and value < 60:
                            critical_indicators += 1
                        elif feature == 'bu' and value > 20:
                            critical_indicators += 1
                            
                        # Calculate score
                        if feature == 'egfr':
                            if value < threshold:
                                score += config['weight']
                        else:
                            if value > threshold:
                                score += config['weight']
                                
                    except (ValueError, TypeError):
                        continue
                        
                # Make prediction based on clinical indicators
                if critical_indicators >= 2 or score >= 2.5:
                    predictions[idx] = 1
                elif score <= 0.5:
                    predictions[idx] = 0
                    
            except Exception as e:
                logging.warning(f"Error processing row {idx}: {str(e)}")
                continue
            
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability estimates"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        # Get probabilities from all models
        all_probas = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(X)
                    all_probas.append((probas, self.weights[name]))
                except Exception as e:
                    logging.error(f"Error getting probabilities from {name}: {str(e)}")
                    
        # Add clinical rules probabilities
        clinical_preds = self._apply_clinical_rules(X)
        clinical_probas = np.zeros((X.shape[0], 2))
        for idx, pred in enumerate(clinical_preds):
            if pred == 1:
                clinical_probas[idx] = [0.1, 0.9]
            elif pred == 0:
                clinical_probas[idx] = [0.9, 0.1]
            else:
                clinical_probas[idx] = [0.5, 0.5]
        all_probas.append((clinical_probas, self.weights['Clinical']))
        
        # Weighted average of probabilities
        final_probas = np.zeros((X.shape[0], 2))
        total_weight = sum(weight for _, weight in all_probas)
        
        for probas, weight in all_probas:
            final_probas += (probas * weight)
        final_probas /= total_weight
        
        return final_probas

class DataProcessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess the dataset"""
        try:
            # Load data
            data = pd.read_csv(data_path)
            
            # Handle classification column first
            if 'classification' in data.columns:
                # Fill NaN values in classification before mapping
                data['classification'] = data['classification'].fillna('notckd')  # or 'ckd' depending on your preference
                # Map text labels to binary values
                class_mapping = {'ckd': 1, 'notckd': 0}
                data['classification'] = data['classification'].map(class_mapping).astype(int)
                logging.info(f"Unique classes after encoding: {data['classification'].unique()}")
            
            # Add these lines before the mapping to debug
            logging.info(f"Missing values in classification: {data['classification'].isna().sum()}")
            logging.info(f"Unique values in classification before mapping: {data['classification'].unique()}")
            
            # Remove ID column if exists
            if 'id' in data.columns:
                data = data.drop('id', axis=1)
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Process other columns
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Scale numerical features
            scaler = StandardScaler()
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
            
            # Encode categorical variables
            for col in categorical_cols:
                if col != 'classification':
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
            
            logging.info("Data preprocessing completed successfully")
            return data, self.label_encoders
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Numerical columns
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            data[col] = data[col].fillna(data[col].median())
            
        # Categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
            
        return data
        
    def _encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
        return data
        
    def _scale_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[num_cols] = self.scaler.fit_transform(data[num_cols])
        return data
    
class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.evaluator = ModelEvaluator(config)
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        """Train all models with different sampling methods"""
        logging.info("Starting model training")
        
        # Ensure y is integer type
        y = y.astype(int)
        logging.info(f"Target values: {np.unique(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Convert to integer type again after split
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        
        # Verify data types
        logging.info(f"y_train dtype: {y_train.dtype}")
        logging.info(f"y_train unique values: {np.unique(y_train)}")
        
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
        """Train all models for a specific sampling method"""
        results = {}
        best_models = {}
        
        for name, model_info in self.config.param_grids.items():
            try:
                logging.info(f"Training {name} with {sampling_name} sampling")
                
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
                metrics = self.evaluator.calculate_metrics(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    **metrics
                }
                
                best_models[name] = grid_search.best_estimator_
                
                # Generate and save plots
                self.evaluator.generate_model_plots(
                    grid_search.best_estimator_,
                    X_test, y_test, y_pred,
                    f"{sampling_name}_{name}"
                )
                
            except Exception as e:
                logging.error(f"Error training {name} with {sampling_name}: {str(e)}")
                continue
                
        return {'results': results, 'best_models': best_models}

class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive model performance metrics"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_true, y_pred),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {}
            
    def generate_model_plots(self, model, X_test: pd.DataFrame, y_test: np.ndarray, 
                           y_pred: np.ndarray, model_name: str):
        """Generate and save all visualization plots"""
        try:
            self._plot_confusion_matrix(y_test, y_pred, model_name)
            self._plot_roc_curve(y_test, y_pred, model_name)
            self._plot_feature_importance(model, X_test, model_name)
            
        except Exception as e:
            logging.error(f"Error generating plots for {model_name}: {str(e)}")
            
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot and save confusion matrix"""
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'{self.config.plots_dir}/{model_name}_confusion_matrix.png')
            plt.close('all')
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            plt.close('all')
        
    def _plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot and save ROC curve"""
        try:
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'{self.config.plots_dir}/{model_name}_roc_curve.png')
            plt.close('all')
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {str(e)}")
            plt.close('all')
        
    def _plot_feature_importance(self, model, X_test: pd.DataFrame, model_name: str):
        """Plot and save feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 6))
                importances = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                sns.barplot(data=importances, x='importance', y='feature')
                plt.title(f'Feature Importance - {model_name}')
                plt.tight_layout()
                plt.savefig(f'{self.config.plots_dir}/{model_name}_feature_importance.png')
                plt.close('all')
        except Exception as e:
            logging.error(f"Error plotting feature importance: {str(e)}")
            plt.close('all')

class ModelPersistence:
    """Handles model saving and loading"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def save_model(self, model: Any, metadata: Dict, filename: str):
        """Save model and its metadata"""
        try:
            # Create timestamp for versioning
            timestamp = self.config.get_timestamp()
            model_filename = f"{filename}_{timestamp}"
            
            # Save model
            model_path = os.path.join(self.config.model_dir, f'{model_filename}.joblib')
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata['timestamp'] = timestamp
            metadata['model_filename'] = model_filename
            metadata_path = os.path.join(self.config.model_dir, f'{model_filename}_metadata.json')
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logging.info(f"Model saved successfully: {model_filename}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filename: str) -> Tuple[Any, Dict]:
        """Load model and its metadata"""
        try:
            model_path = os.path.join(self.config.model_dir, f'{filename}.joblib')
            metadata_path = os.path.join(self.config.model_dir, f'{filename}_metadata.json')
            
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            return model, metadata
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

def create_ensemble_model(best_models: Dict) -> EnhancedCKDEnsembleClassifier:
    """Create the final ensemble model from best individual models"""
    config = ModelConfig()
    return EnhancedCKDEnsembleClassifier(
        models=best_models,
        threshold_config=config.clinical_thresholds
    )

def main():
    """Main execution pipeline"""
    try:
        # Initialize components
        config = ModelConfig()
        data_processor = DataProcessor(config)
        model_trainer = ModelTrainer(config)
        model_persistence = ModelPersistence(config)
        
        # Load and prepare data
        logging.info("Loading and preparing data...")
        data, label_encoders = data_processor.load_and_prepare_data("kidney_disease.csv")
        
        # Split features and target
        X = data.drop(columns=['classification'])
        y = data['classification'].astype(int)  # Ensure integer type
        
        logging.info(f"Target variable unique values: {np.unique(y)}")
        logging.info(f"Target variable dtype: {y.dtype}")
        
        # Train models
        results, best_models = model_trainer.train_models(X, y)
        
        # Create and train ensemble model
        logging.info("Creating ensemble model...")
        ensemble_model = create_ensemble_model(best_models['Original'])
        
        # Final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y
        )
        
        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)
        
        # Calculate final metrics
        evaluator = ModelEvaluator(config)
        final_metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        # Generate ensemble model plots
        evaluator.generate_model_plots(
            ensemble_model,
            X_test, y_test, y_pred,
            "ensemble_model"
        )
        
        # Save ensemble model
        metadata = {
            'feature_columns': list(X.columns),
            'model_type': 'ensemble',
            'base_models': list(best_models['Original'].keys()),
            'performance': final_metrics,
            'label_encoders': {
                col: list(le.classes_) 
                for col, le in label_encoders.items()
            }
        }
        
        model_persistence.save_model(
            ensemble_model,
            metadata,
            'ckd_ensemble_model'
        )
        
        # Print summary
        logging.info("Training completed successfully")
        print("\nTraining Summary:")
        print(f"Ensemble Model Performance:")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"F1 Score: {final_metrics['f1']:.4f}")
        print(f"ROC AUC: {final_metrics['roc_auc']:.4f}")
        
        # Save detailed results
        results_path = os.path.join(config.metrics_dir, f'training_results_{config.get_timestamp()}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()