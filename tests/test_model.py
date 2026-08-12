import unittest
import pickle
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.predict_model import predict_text


class TestModel(unittest.TestCase):
    """Test cases for model prediction functionality"""

    @classmethod
    def setUpClass(cls):
        """Load the model and vectorizer once for all tests"""
        model_path = 'models/model.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                cls.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                cls.vectorizer = pickle.load(f)
            cls.model_loaded = True
        else:
            cls.model_loaded = False

    def test_model_loaded(self):
        """Test that model files are present"""
        self.assertTrue(
            self.model_loaded,
            "Model and vectorizer files should exist in models/ directory"
        )

    def test_prediction_shape(self):
        """Test that predictions have correct shape"""
        if not self.model_loaded:
            self.skipTest("Model not loaded")
        
        test_text = "This is a positive review"
        try:
            # Vectorize the text
            text_vec = self.vectorizer.transform([test_text])
            # Make prediction
            prediction = self.model.predict(text_vec)
            
            self.assertEqual(len(prediction), 1, "Prediction should be 1-dimensional array")
        except Exception as e:
            self.fail(f"Prediction failed: {str(e)}")

    def test_prediction_range(self):
        """Test that predictions are binary (0 or 1)"""
        if not self.model_loaded:
            self.skipTest("Model not loaded")
        
        test_texts = [
            "This movie is amazing",
            "This is terrible",
            "absolutely wonderful"
        ]
        
        try:
            text_vecs = self.vectorizer.transform(test_texts)
            predictions = self.model.predict(text_vecs)
            
            for pred in predictions:
                self.assertIn(pred, [0, 1], "Prediction should be 0 or 1")
        except Exception as e:
            self.fail(f"Prediction failed: {str(e)}")

    def test_empty_text_handling(self):
        """Test that empty text is handled gracefully"""
        if not self.model_loaded:
            self.skipTest("Model not loaded")
        
        try:
            text_vec = self.vectorizer.transform([""])
            prediction = self.model.predict(text_vec)
            self.assertIsNotNone(prediction, "Empty text should produce a prediction")
        except Exception as e:
            self.fail(f"Empty text handling failed: {str(e)}")


if __name__ == '__main__':
    unittest.main()
