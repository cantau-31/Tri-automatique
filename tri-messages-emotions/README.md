# tri-messages-emotions

Application de tri automatique de messages par emotion.

Le projet utilise :

- le dataset Hugging Face `dair-ai/emotion`
- DistilBERT avec `transformers`
- une API FastAPI
- un front simple Gradio

## Installation

Depuis le dossier du projet :

```bash
cd tri-messages-emotions
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entrainer le modele

Entrainement rapide pour la demo :

```bash
python data_training/train_model.py
```

Entrainement plus complet :

```bash
python data_training/train_model.py --full-data --epochs 3 --batch-size 16
```

Le modele est sauvegarde dans :

```text
data_training/saved_model/
```

## Evaluer le modele

```bash
python data_training/evaluate_model.py
```

Les fichiers generes sont :

```text
data_training/metrics.json
data_training/confusion_matrix.png
```

## Lancer l'API

```bash
python -m uvicorn api.main:app --reload
```

API locale :

```text
http://127.0.0.1:8000
```

Documentation :

```text
http://127.0.0.1:8000/docs
```

Exemple de requete :

```json
{
  "text": "I am happy today"
}
```

Exemple de reponse :

```json
{
  "emotion": "joy",
  "confidence": 0.98
}
```

## Lancer le front

Dans un deuxieme terminal :

```bash
source .venv/bin/activate
python front/app.py
```

Le front Gradio appelle l'API FastAPI sur :

```text
http://localhost:8000/predict
```

## Tests API

```bash
python -m pytest api/test_api.py
```

## Ordre recommande pour la demo

1. `python data_training/train_model.py`
2. `python data_training/evaluate_model.py`
3. `python -m uvicorn api.main:app --reload`
4. Dans un autre terminal : `python front/app.py`
