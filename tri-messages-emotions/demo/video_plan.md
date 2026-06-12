# Plan video demo

1. Montrer rapidement la structure du projet.
2. Expliquer que le dataset utilise est `dair-ai/emotion`.
3. Lancer ou montrer l'entrainement avec `python data_training/train_model.py`.
4. Montrer les fichiers generes : `saved_model/`, `metrics.json`, `confusion_matrix.png`.
5. Lancer l'API avec `python -m uvicorn api.main:app --reload`.
6. Tester `/predict` dans `http://127.0.0.1:8000/docs`.
7. Lancer le front avec `python front/app.py`.
8. Faire une prediction avec un message comme `I am happy today`.
