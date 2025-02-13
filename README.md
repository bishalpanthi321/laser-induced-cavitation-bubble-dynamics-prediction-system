# laser-induced-cavitation-bubble-dynamics-prediction-system
# Bubble Dynamics Machine Learning Project

## Overview
This project implements an advanced machine learning model to analyze and predict bubble dynamics in fluid systems. It uses TensorFlow and scikit-learn to process complex physical parameters and predict bubble behavior under various conditions.

## Project Structure
```
├── src/
│   ├── bubble_dynamics.py    # Core ML model implementation
│   └── main.py              # Main execution script
├── datasets/
│   ├── synthetic/           # Synthetic dataset generation
│   │   └── bubble_parameters.py
│   ├── experimental/        # Experimental data handling
│   │   └── data_loader.py
│   └── processed/           # Data processing utilities
│       └── data_processor.py
└── README.md
```

## Features
- Advanced neural network architecture for bubble dynamics prediction
- Synthetic data generation for model training
- Experimental data integration capabilities
- Comprehensive data processing pipeline
- Real-time prediction visualization

## Technical Details

### Core Components
1. **BubbleDynamicsModel (src/bubble_dynamics.py)**
   - Neural network architecture with multiple dense layers
   - Dropout layers for regularization
   - Custom data generation methods
   - Model training and prediction utilities

2. **Data Processing (datasets/)**
   - Synthetic data generation with physical parameters
   - Experimental data loading and processing
   - Data merging and standardization

### Model Architecture
- Input Layer: 5 features (time, pressure, temperature, laser power, initial radius)
- Hidden Layers: 128 → 64 → 32 neurons with ReLU activation
- Output Layer: 1 neuron (predicted bubble radius)
- Optimizer: Adam
- Loss Function: Mean Squared Error

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Running the Project
1. Generate synthetic data:
```bash
python datasets/synthetic/bubble_parameters.py
```

2. Run the main script:
```bash
python src/main.py
```

## Model Performance
The model evaluates performance using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Training/Validation Loss Curves
- Prediction Accuracy Visualization

## Output
- Training history plots saved as 'training_history.png'
- Processed data saved in 'datasets/processed/combined_data.csv'
- Model predictions with relative error calculations

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
This project is licensed under the MIT License.
