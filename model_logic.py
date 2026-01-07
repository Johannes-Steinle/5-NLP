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
from functools import wraps

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def my_logger(orig_func):
    """Loggt den Funktionsnamen und die übergebenen Argumente."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    """Loggt die Ausführungszeit der Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(f'{orig_func.__name__} ran in: {t2:.4f} sec')
        return result
    return wrapper

@my_logger
@my_timer
def load_data(filepath):
    """Lädt die Yelp-Daten und filtert nach 1 und 5 Sternen."""
    try:
        yelp = pd.read_csv(filepath)
        # Nur 1 und 5 Sterne Bewertungen
        yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
        X = yelp_class['text']
        y = yelp_class['stars']
        return X, y
    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten: {e}")
        raise

@my_logger
@my_timer
def fit_model(X_train, y_train):
    """Trainiert die NLP Pipeline (CV + TFIDF + NB)."""
    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

@my_logger
@my_timer
def predict_model(model, X_test):
    """Führt Vorhersagen mit der Pipeline durch."""
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    X, y = load_data('Yelp.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    model = fit_model(X_train, y_train)
    preds = predict_model(model, X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
