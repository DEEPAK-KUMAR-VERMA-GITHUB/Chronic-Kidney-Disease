from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class EnhancedCKDEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, threshold_config=None):
        self.base_model = base_model
        self.threshold_config = threshold_config or {
            'creatinine': {'threshold': 1.2, 'weight': 2.5},
            'bun': {'threshold': 18.0, 'weight': 2.0},
            'egfr': {'threshold': 65.0, 'weight': 2.5},
            'hemoglobin': {'threshold': 13.0, 'weight': 1.5}
        }
        self.classes_ = None

    def fit(self, X, y):
        if self.base_model is None:
            raise ValueError("Base model must be specified")
        self.classes_ = np.unique(y)
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")
        ml_predictions = self.base_model.predict(X)
        clinical_predictions = self._apply_clinical_rules(X)
        final_predictions = np.where(clinical_predictions != -1, 
                                   clinical_predictions, 
                                   ml_predictions)
        return final_predictions

    def predict_proba(self, X):
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")
        if not hasattr(self.base_model, 'predict_proba'):
            raise NotImplementedError("Base model doesn't support probability predictions")
        
        # Get base model probabilities
        ml_probas = self.base_model.predict_proba(X)
        clinical_preds = self._apply_clinical_rules(X)
        
        # Create a copy to avoid modifying the original array
        final_probas = ml_probas.copy()
        
        # Modify probabilities based on clinical rules
        for idx, clinical_pred in enumerate(clinical_preds):
            if clinical_pred != -1:
                # Get the number of classes from the ml_probas shape
                n_classes = ml_probas.shape[1]
                
                if n_classes == 2:
                    final_probas[idx] = [0.1, 0.9] if clinical_pred == 1 else [0.9, 0.1]
                else:
                    # For multi-class, distribute probabilities
                    probs = np.zeros(n_classes)
                    if clinical_pred == 1:
                        probs[-1] = 0.9  # High probability for positive class
                        probs[:-1] = 0.1 / (n_classes - 1)  # Distribute remaining probability
                    else:
                        probs[0] = 0.9  # High probability for negative class
                        probs[1:] = 0.1 / (n_classes - 1)  # Distribute remaining probability
                    final_probas[idx] = probs
        
        return final_probas

    def _apply_clinical_rules(self, X):
        predictions = np.full(X.shape[0], -1)
        
        # Define feature mapping for clinical rules
        feature_mapping = {
            'creatinine': 'sc',  # serum creatinine
            'bun': 'bu',         # blood urea
            'egfr': 'egfr',      # estimated GFR
            'hemoglobin': 'hemo' # hemoglobin
        }
        
        # Convert X to numeric values if it's a DataFrame
        X_values = X.values if hasattr(X, 'values') else X
        
        for idx, row in enumerate(X_values):
            score = 0
            critical_indicators = 0
            
            for feature_idx, (feature, config) in enumerate(self.threshold_config.items()):
                try:
                    # Convert value to float
                    value = float(row[feature_idx])
                    threshold = float(config['threshold'])
                    
                    # Apply clinical rules
                    if feature == 'creatinine' and value > 1.5:
                        critical_indicators += 1
                    elif feature == 'egfr' and value < 60:
                        critical_indicators += 1
                    elif feature == 'bun' and value > 20:
                        critical_indicators += 1
                    
                    if feature == 'egfr':
                        if value < threshold:
                            score += config['weight']
                    else:
                        if value > threshold:
                            score += config['weight']
                        
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error processing feature {feature} with value {row[feature_idx]}: {str(e)}")
                    continue
            
            if critical_indicators >= 2 or score >= 2.5:
                predictions[idx] = 1
            elif score <= 0.5:
                predictions[idx] = 0
        
        return predictions

    def get_params(self, deep=True):
        params = {
            'base_model': self.base_model,
            'threshold_config': self.threshold_config
        }
        if deep and hasattr(self.base_model, 'get_params'):
            params.update({
                f'base_model__{key}': value 
                for key, value in self.base_model.get_params(deep=True).items()
            })
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'base_model':
                self.base_model = value
            elif key == 'threshold_config':
                self.threshold_config = value
            elif key.startswith('base_model__'):
                self.base_model.set_params(**{key[12:]: value})
            else:
                raise ValueError(f'Invalid parameter {key}')
        return self 