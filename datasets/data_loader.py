import pandas as pd
import numpy as np

class ExperimentalDataLoader:
    @staticmethod
    def load_experimental_data():
        # Simulated experimental data
        n_samples = 1000
        experimental_data = {
            'time': np.linspace(0, 1e-6, n_samples),
            'measured_radius': np.random.normal(3e-6, 0.1e-6, n_samples),
            'laser_power': np.random.uniform(1e6, 5e6, n_samples),
            'temperature': np.random.normal(300, 5, n_samples)
        }
        return pd.DataFrame(experimental_data)

    @staticmethod
    def save_experimental_data(df):
        df.to_csv('datasets/experimental/measured_data.csv', index=False)
