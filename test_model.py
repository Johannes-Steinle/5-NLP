import unittest
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_logic import load_data, fit_model, predict_model

class TestNLPModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.X, cls.y = load_data('Yelp.csv')
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=101
        )
        # Norm-Zeit für Pipeline ca. 0.5s
        cls.norm_fit_time = 1.0 

    def test_1_nlp_accuracy(self):
        """
        Ziel: Accuracy > 0.80.
        """
        model, _ = fit_model(self.X_train, self.y_train)
        predictions = predict_model(model, self.X_test)
        
        acc = accuracy_score(self.y_test, predictions)
        
        print(f"\n[Test NLP] Gemessene Accuracy: {acc}")
        self.assertGreater(acc, 0.80, f"Accuracy zu niedrig: {acc} < 0.80")

    def test_2_fit_runtime(self):
        """
        Ziel: Trainingzeit < 120% der Normzeit.
        """
        _, duration = fit_model(self.X_train, self.y_train)
        
        limit = self.norm_fit_time * 1.2
        print(f"\n[Test Fit] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
