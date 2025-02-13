from bubble_physics import BubblePhysics
from ml_model import CavitationModel
from data_processor import DataProcessor
from metrics import ValidationMetrics
from data_augmentation import DataAugmentation
from trainer import ModelTrainer
from visualizer import ResultVisualizer
from config import ModelConfig, PhysicsConfig, ExperimentConfig

import numpy as np

class ExperimentRunner:
    def __init__(self):
        self.model_config = ModelConfig()
        self.physics_config = PhysicsConfig()
        self.exp_config = ExperimentConfig()
        
    def run_experiment(self):
        # Initialize components
        physics = BubblePhysics(**self.physics_config.__dict__)
        model = CavitationModel(self.model_config.input_dim)
        processor = DataProcessor()
        augmentor = DataAugmentation()
        trainer = ModelTrainer(model.model, self.model_config.batch_size, 
                             self.model_config.epochs)
        
        # Generate and process data
        t = np.linspace(*self.exp_config.time_span, self.exp_config.time_steps)
        R, Rdot = physics.simulate(t, self.exp_config.vapor_pressure)
        
        # Train model and visualize results
        X = np.column_stack([t, R, Rdot])
        y = R
        
        X_processed, y_processed = processor.process_experimental_data(X, y)
        X_aug, y_aug = augmentor.augment_time_series(X_processed, y_processed)
        
        # Split data and train
        split_idx = int(len(X_aug) * (1 - self.model_config.validation_split))
        history = trainer.train_with_validation(
            X_aug[:split_idx], y_aug[:split_idx],
            X_aug[split_idx:], y_aug[split_idx:]
        )
        
        # Visualize results
        visualizer = ResultVisualizer()
        visualizer.plot_training_history(history)
