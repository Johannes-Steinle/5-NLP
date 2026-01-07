# NLP Projekt

Dieses Repository enthält ein Natural Language Processing (NLP) Projekt als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist es, Yelp-Bewertungen basierend auf ihrem Textinhalt in 1-Stern oder 5-Sterne Kategorien zu klassifizieren.

## Inhalt
* `NLP_Solution.ipynb`: Das Haupt-Notebook mit der NLP-Pipeline und Klassifizierung.
* `Yelp.csv`: Der Datensatz, der für Training und Test verwendet wurde.

## Prüfungsaufgabe 2: Automatisierung und Testen

Dieses Projekt wurde gemäß den Anforderungen für Aufgabe 2 refaktoriert und mit automatisierten Tests sowie Logging ausgestattet.

### Struktur
- `model_logic.py`: Enthält die Kernlogik (Pipeline aus CountVectorizer, TfidfTransformer und MultinomialNB) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Modellgüte und der Trainingslaufzeit durch.
- `training.log`: Wird automatisch erstellt und protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt:
```text
[Test NLP] Gemessene Accuracy: 0.8140293637846656
.
[Test Fit] Gemessene Dauer: 0.1485s (Limit: 1.2000s)
.
Ran 2 tests in 0.454s
OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/5-NLP/main?filepath=NLP_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/5-NLP/main?filepath=NLP_Solution.ipynb)
