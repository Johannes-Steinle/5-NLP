# NLP Projekt

Meine Umsetzung der Natural Language Processing Übung aus dem Udemy-Kurs "Python für Data Science, Maschinelles Lernen & Visualization" im Rahmen der Angleichungsleistung.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/5-NLP/main?filepath=NLP_Solution.ipynb)

## Überblick
Klassifizierung von Yelp-Bewertungen anhand ihres Textinhalts in 1-Stern oder 5-Sterne Kategorien. Pipeline: CountVectorizer, TF-IDF Transformer und Naive Bayes.

## Inhalt
* `NLP_Solution.ipynb` - Haupt-Notebook mit der NLP-Pipeline und Klassifizierung
* `Yelp.csv` - Yelp-Bewertungsdaten

## Ausführung

1. Auf den **Binder-Badge** oben klicken, um das Notebook in myBinder zu starten.
2. Warten, bis die Umgebung geladen ist (kann 1-2 Minuten dauern).
3. `NLP_Solution.ipynb` öffnen.
4. Alle Zellen nacheinander ausführen (*Run > Run All Cells*).
5. **Erwartete Ergebnisse:**
   - Analyse der Yelp-Bewertungen (Histogramm der Sterne-Verteilung, Textlängen)
   - Filterung auf 1- und 5-Sterne Bewertungen
   - Aufbau der NLP-Pipeline (Bag of Words -> TF-IDF -> Naive Bayes)
   - Classification Report mit Precision, Recall und F1-Score
   - Accuracy von ca. **0.81 - 0.93** je nach Pipeline-Konfiguration

---

## Prüfungsaufgabe 2: Automatisierung und Testen

Ich habe das Projekt für Aufgabe 2 um Unit-Tests und Logging erweitert, nach dem Ansatz aus dem Artikel "Unit Testing and Logging for Data Science".

### Dateien
| Datei | Beschreibung |
|---|---|
| `model_logic.py` | NLP-Pipeline (CountVectorizer + TfidfTransformer + MultinomialNB) mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Unit-Tests für `predict()` (Accuracy) und `fit()` (Laufzeit) |
| `generate_test_data.py` | Skript zur Erzeugung der Testdaten |
| `train_data.csv` | Trainingsdaten (2860 Bewertungen) |
| `test_data.csv` | Testdaten (1226 Bewertungen) |
| `training.log` | Log-File mit Trainingsereignissen |

### Testfälle

**Testfall 1 - predict():** Die NLP-Pipeline wird auf `train_data.csv` trainiert und die Accuracy auf `test_data.csv` geprüft. Ziel: Accuracy > 0.80.

**Testfall 2 - fit():** Die Laufzeit der Trainingsfunktion wird gemessen und geprüft, ob sie unter 120% der Normzeit (0.8s) bleibt.

### Testergebnisse
```text
[Test predict()] Gemessene Accuracy: 0.8140
.
[Test fit()] Gemessene Dauer: 0.1206s (Limit: 0.9600s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.300s

OK
```

### Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Daten aus `test_data.csv` und `train_data.csv`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testdaten neu zu generieren: `python generate_test_data.py`
