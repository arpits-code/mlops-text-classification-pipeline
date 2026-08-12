"""
Model Promotion Script

This script handles promoting a trained model to production.
It checks model metrics and registers the model if they meet acceptance criteria.
"""

import os
import pickle
import json
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metrics():
    """Load model metrics from metrics.json"""
    metrics_path = Path('reports/metrics.json')
    
    if not metrics_path.exists():
        logger.warning("Metrics file not found at reports/metrics.json")
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Metrics loaded: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to load metrics: {str(e)}")
        return None


def validate_model_quality(metrics):
    """
    Validate that model meets acceptance criteria
    
    Args:
        metrics: Dictionary containing model metrics
        
    Returns:
        bool: True if model passes validation, False otherwise
    """
    if metrics is None:
        logger.warning("No metrics available for validation")
        return False
    
    # Define acceptance criteria
    MIN_ACCURACY = 0.70
    MIN_F1_SCORE = 0.65
    
    try:
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1_score', 0)
        
        if accuracy < MIN_ACCURACY:
            logger.warning(f"Accuracy {accuracy} below minimum {MIN_ACCURACY}")
            return False
        
        if f1_score < MIN_F1_SCORE:
            logger.warning(f"F1 Score {f1_score} below minimum {MIN_F1_SCORE}")
            return False
        
        logger.info("Model passed quality validation")
        return True
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False


def check_model_artifacts():
    """
    Check that required model artifacts exist
    
    Returns:
        bool: True if all artifacts exist, False otherwise
    """
    required_files = [
        'models/model.pkl',
        'models/vectorizer.pkl'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required model artifact missing: {file_path}")
            return False
    
    logger.info("All model artifacts present")
    return True


def promote_model():
    """
    Main promotion logic
    
    Performs the following checks:
    1. Verify model artifacts exist
    2. Load and validate metrics
    3. Log promotion status
    """
    logger.info("Starting model promotion process...")
    
    # Check artifacts
    if not check_model_artifacts():
        logger.error("Model promotion failed: Missing artifacts")
        sys.exit(1)
    
    # Load and validate metrics
    metrics = load_metrics()
    
    if not validate_model_quality(metrics):
        logger.warning("Model promotion skipped: Quality criteria not met")
        logger.info("The model did not pass quality gates. Review metrics and retrain.")
        sys.exit(0)  # Exit successfully but don't promote
    
    # If we reach here, model is ready for promotion
    logger.info("✓ Model promotion successful!")
    logger.info("Model is ready for production deployment")
    
    # Create a promotion record
    promotion_record = {
        'status': 'promoted',
        'metrics': metrics,
        'timestamp': str(Path('reports/metrics.json').stat().st_mtime)
    }
    
    promotion_file = Path('reports/promotion_record.json')
    try:
        with open(promotion_file, 'w') as f:
            json.dump(promotion_record, f, indent=2)
        logger.info(f"Promotion record saved to {promotion_file}")
    except Exception as e:
        logger.warning(f"Could not save promotion record: {str(e)}")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = promote_model()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unexpected error during model promotion: {str(e)}")
        sys.exit(1)
