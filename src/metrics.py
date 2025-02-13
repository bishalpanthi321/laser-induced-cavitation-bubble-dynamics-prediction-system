import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple

class ValidationMetrics:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate relative errors
        rel_errors = np.abs(y_true - y_pred) / y_true
        mape = np.mean(rel_errors) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape
        }
