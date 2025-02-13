
from experiment import ExperimentRunner
import tensorflow as tf
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logging.info("Starting cavitation bubble dynamics experiment")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run experiment
    runner = ExperimentRunner()
    runner.run_experiment()
    
    logging.info("Experiment completed successfully")

if __name__ == "__main__":
    main()