from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    input_dim: int = 5
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
@dataclass
class PhysicsConfig:
    initial_radius: float = 1e-6
    ambient_pressure: float = 101325
    liquid_density: float = 998
    surface_tension: float = 0.072
    dynamic_viscosity: float = 0.001
    
@dataclass
class ExperimentConfig:
    time_steps: int = 1000
    time_span: tuple = (0, 1e-6)
    vapor_pressure: float = 2300
