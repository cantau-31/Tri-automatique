# Plan de presentation

## 1. Objectif

Classer automatiquement des messages en anglais selon leur emotion.

## 2. Donnees

Dataset Hugging Face : `dair-ai/emotion`.

Classes principales : sadness, joy, love, anger, fear, surprise.

## 3. Modele

Modele utilise : `distilbert-base-uncased`.

Fine-tuning avec `DistilBertTokenizerFast`, `DistilBertForSequenceClassification` et `Trainer`.

## 4. API

API FastAPI avec un endpoint :

```text
POST /predict
```

Entree :

```json
{ "text": "I am happy today" }
```

Sortie :

```json
{ "emotion": "joy", "confidence": 0.32 }
```

## 5. Front

Interface Gradio simple qui appelle l'API locale.

## 6. Limites

Le modele de demonstration est entraine rapidement. Pour de meilleurs resultats, lancer un entrainement complet avec `--full-data`.
