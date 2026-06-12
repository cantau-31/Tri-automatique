# Deploiement Hugging Face Spaces

Le projet est pret pour un deploiement Gradio sur Hugging Face Spaces.

## Fichiers importants

- `app.py` : entree Hugging Face Spaces, charge le modele et lance Gradio.
- `requirements.txt` : dependances Python.
- `data_training/saved_model/model.safetensors` : poids du modele.
- `data_training/saved_model/config.json` : configuration du modele.
- `data_training/saved_model/tokenizer.json` : tokenizer.
- `.gitattributes` : indique que `model.safetensors` doit etre gere par Git LFS.

## Pourquoi Git LFS ?

`model.safetensors` fait environ 255 MB. GitHub et Hugging Face refusent souvent les gros fichiers sans Git LFS.

## Commandes

Depuis la racine du projet `tri-messages-emotions/` :

```bash
brew install git-lfs
git lfs install
git lfs track "data_training/saved_model/model.safetensors"
git add .gitattributes
git add app.py requirements.txt DEPLOY.md .gitignore
git add data_training/saved_model/config.json
git add data_training/saved_model/model.safetensors
git add data_training/saved_model/tokenizer.json
git add data_training/saved_model/tokenizer_config.json
git commit -m "Prepare Hugging Face Spaces deployment"
git push
```

## Lancer comme sur Hugging Face Spaces en local

```bash
python3 app.py
```

Cette version ne depend pas de l'API locale. Elle charge directement le modele.

## API FastAPI

L'API dans `api/main.py` reste utile pour le lancement local avec :

```bash
python3 -m uvicorn api.main:app --reload
```

Mais pour Hugging Face Spaces, l'entree principale est `app.py`.
