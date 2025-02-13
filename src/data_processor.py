import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional

class DataProcessor:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = RobustScaler()
        
    def process_experimental_data(self, data: np.ndarray, 
                                labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Remove outliers using IQR method
        Q1 = np.percentile(labels, 25)
        Q3 = np.percentile(labels, 75)
        IQR = Q3 - Q1
        mask = (labels >= Q1 - 1.5*IQR) & (labels <= Q3 + 1.5*IQR)
        
        cleaned_data = data[mask]
        cleaned_labels = labels[mask]
        
        # Scale features and targets
        scaled_data = self.feature_scaler.fit_transform(cleaned_data)
        scaled_labels = self.target_scaler.fit_transform(cleaned_labels.reshape(-1, 1))
        
        return scaled_data, scaled_labels.ravel()
