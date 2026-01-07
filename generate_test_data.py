"""
Generiert die Trainings- und Testdaten für den Unit-Test.
Ausführung: python generate_test_data.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

yelp = pd.read_csv('Yelp.csv')
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

train_data = pd.DataFrame({'text': X_train, 'stars': y_train})
train_data.to_csv('train_data.csv', index=False)

test_data = pd.DataFrame({'text': X_test, 'stars': y_test})
test_data.to_csv('test_data.csv', index=False)

print(f"Trainingsdaten gespeichert: {len(train_data)} Zeilen -> train_data.csv")
print(f"Testdaten gespeichert: {len(test_data)} Zeilen -> test_data.csv")
