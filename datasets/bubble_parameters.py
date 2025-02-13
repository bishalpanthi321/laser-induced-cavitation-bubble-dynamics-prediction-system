import numpy as np
import pandas as pd

def generate_bubble_parameters(n_samples=10000):
    data = {
        'time': np.random.uniform(0, 1e-6, n_samples),
        'pressure': np.random.uniform(1e5, 5e5, n_samples),
        'temperature': np.random.uniform(293, 373, n_samples),
        'laser_power': np.random.uniform(1e6, 5e6, n_samples),
        'initial_radius': np.random.uniform(1e-6, 5e-6, n_samples),
        'viscosity': np.random.uniform(0.8e-3, 1.2e-3, n_samples),
        'surface_tension': np.random.uniform(0.07, 0.073, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('datasets/synthetic/bubble_parameters.csv', index=False)
    return df
