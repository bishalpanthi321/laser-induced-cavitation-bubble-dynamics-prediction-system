import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class CavitationModel:
    def __init__(self, input_dim=5):
        self.model = self._build_complex_model(input_dim)
        
    def _build_complex_model(self, input_dim):
        inputs = layers.Input(shape=(input_dim,))
        
        # Parallel processing branches
        branch1 = layers.Dense(64, activation='relu')(inputs)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.3)(branch1)
        
        branch2 = layers.Dense(32, activation='tanh')(inputs)
        branch2 = layers.BatchNormalization()(branch2)
        
        # Merge branches
        merged = layers.Concatenate()([branch1, branch2])
        
        # Deep network
        x = layers.Dense(128, activation='elu')(merged)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='selu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model
