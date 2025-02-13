import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

class DataAugmentation:
    @staticmethod
    def augment_time_series(data: np.ndarray, labels: np.ndarray, 
                           noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        aug_data = []
        aug_labels = []
        
        for i in range(len(data)):
            # Original data
            aug_data.append(data[i])
            aug_labels.append(labels[i])
            
            # Add noise
            noise = np.random.normal(0, noise_level, data[i].shape)
            aug_data.append(data[i] + noise)
            aug_labels.append(labels[i])
            
            # Time warping
            time_warp = np.random.uniform(0.8, 1.2)
            warped_data = np.interp(np.arange(len(data[i])) * time_warp,
                                  np.arange(len(data[i])),
                                  data[i])
            aug_data.append(warped_data)
            aug_labels.append(labels[i])
        
        return np.array(aug_data), np.array(aug_labels)
