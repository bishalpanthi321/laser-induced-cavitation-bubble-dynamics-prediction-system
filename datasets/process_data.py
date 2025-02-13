
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_data(self, synthetic_df, experimental_df):
        # Merge and process both synthetic and experimental data
        processed_data = pd.merge(
            synthetic_df,
            experimental_df,
            on=['time', 'laser_power'],
            how='inner',
            suffixes=('_syn', '_exp')
        )
        
        # Save processed dataset
        processed_data.to_csv('datasets/processed/combined_data.csv', index=False)
        return processed_data