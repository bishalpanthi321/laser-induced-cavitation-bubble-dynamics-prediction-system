import tensorflow as tf
from typing import Tuple, Dict
import numpy as np

class ModelTrainer:
    def __init__(self, model, batch_size=32, epochs=100):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        
    def train_with_validation(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        return history.history
