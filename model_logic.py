import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging
import time
import os

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_data(filepath):
    """Lädt die Yelp-Daten und filtert nach 1 und 5 Sternen."""
    logger = logging.getLogger()
    try:
        yelp = pd.read_csv(filepath)
        logger.info(f"Daten erfolgreich von {filepath} geladen.")
        
        # Nur 1 und 5 Sterne Bewertungen
        yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
        
        X = yelp_class['text']
        y = yelp_class['stars']
        
        return X, y
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {e}")
        raise

def fit_model(X_train, y_train):
    """Trainiert die NLP Pipeline (CV + TFIDF + NB) und misst die Zeit."""
    logger = logging.getLogger()
    start_time = time.time()
    
    logger.info("Starte Pipeline Training (CV, TFIDF, MultinomialNB)...")
    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),
    ])
    
    pipeline.fit(X_train, y_train)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Training beendet in {duration:.4f} Sekunden.")
    
    return pipeline, duration

def predict_model(model, X_test):
    """Führt Vorhersagen mit der Pipeline durch."""
    logger = logging.getLogger()
    logger.info("Erstelle Vorhersagen...")
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    X, y = load_data('Yelp.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    model, duration = fit_model(X_train, y_train)
    preds = predict_model(model, X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    logging.info(f"NLP Accuracy im Testlauf: {acc}")
