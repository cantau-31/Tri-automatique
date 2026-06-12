# détecteur d'émotions — M106 projet de fin de module
 
## le projet
 
une appli qui prend un texte en entrée et prédit l'émotion associée parmi 6 catégories :
joy, sadness, anger, fear, love, surprise.
 
secteur visé : support client / modération / RH — analyse automatique de messages
pour identifier rapidement les retours négatifs ou urgents, sans tout lire manuellement.
 
## stack
 
- **dataset** : `dair-ai/emotion` (hugging face)
- **modèle** : DistilBERT fine-tuné avec `transformers`
- **API** : FastAPI (`/predict`)
- **front** : Gradio
- **déploiement** : Hugging Face Spaces
## résultats du modèle
 
- accuracy : **91.8%**
- macro F1 : 0.877
- weighted F1 : 0.918
(voir `data_training/metrics.json` et `data_training/confusion_matrix.png`)
 
## architecture
 
```text
tri-messages-emotions/
├── data_training/      # entraînement + évaluation du modèle
├── api/                 # API FastAPI (/predict)
├── front/               # interface Gradio
├── presentation/        # plan de la présentation
└── demo/                # plan de la vidéo démo
```
 
## installation
 
```bash
cd tri-messages-emotions
python3 -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate sur windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
 
## entraîner le modèle
 
entraînement rapide pour la démo :
 
```bash
python data_training/train_model.py
```
 
entraînement plus complet :
 
```bash
python data_training/train_model.py --full-data --epochs 3 --batch-size 16
```
 
le modèle est sauvegardé dans :
 
```text
data_training/saved_model/
```
 
(déjà inclus dans le repo, donc cette étape est optionnelle)
 
## évaluer le modèle
 
```bash
python data_training/evaluate_model.py
```
 
fichiers générés :
 
```text
data_training/metrics.json
data_training/confusion_matrix.png
```
 
## lancer l'API
 
```bash
python -m uvicorn api.main:app --reload
```
 
API locale : `http://127.0.0.1:8000`
documentation : `http://127.0.0.1:8000/docs`
 
exemple de requête :
 
```json
{ "text": "I am happy today" }
```
 
exemple de réponse :
 
```json
{ "emotion": "joy", "confidence": 0.98 }
```
 
## lancer le front
 
dans un deuxième terminal :
 
```bash
source .venv/bin/activate
python front/app.py
```
 
front Gradio disponible sur `http://127.0.0.1:7860`, il appelle l'API sur
`http://localhost:8000/predict` — l'API doit donc être lancée avant.
 
## tests API
 
```bash
python -m pytest api/test_api.py
```
 
## ordre recommandé pour la démo
 
1. `python data_training/train_model.py` (optionnel, modèle déjà fourni)
2. `python data_training/evaluate_model.py`
3. `python -m uvicorn api.main:app --reload`
4. dans un autre terminal : `python front/app.py`
## choix techniques
 
- **DistilBERT** plutôt que BERT complet : plus léger, plus rapide à fine-tuner,
  adapté à une journée et demie de projet.
- **transfer learning** : on part d'un modèle pré-entraîné et on fine-tune sur
  `dair-ai/emotion` plutôt que d'entraîner depuis zéro.
- **Gradio pour le front** : interface fonctionnelle sans écrire de HTML/JS,
  déployable directement sur Hugging Face Spaces.
- **objectif** : un modèle qui marche réellement et déployé, plutôt qu'un
  modèle "parfait" jamais terminé.
## équipe & répartition
 
- **rayen** — dataset, fine-tuning DistilBERT, évaluation
- **julien** — API FastAPI
- **steven** — front, présentation, README
- **redouane** — logique métier / communication API, vidéo démo

##deploy
https://huggingface.co/spaces/Steevv/Detecteur_emotions
