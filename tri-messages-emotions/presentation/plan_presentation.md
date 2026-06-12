# plan de présentation — détecteur d'émotions

## 1. intro (steven)

- présentation du projet : un détecteur d'émotions sur du texte
- 6 émotions possibles : joy, sadness, anger, fear, love, surprise
- dataset `dair-ai/emotion` + DistilBERT fine-tuné
- annonce du plan : entraînement → API → front → déploiement

## 2. problématique (steven)

- contexte : tri manuel de messages impossible à grande échelle
  (avis clients, tickets support, commentaires)
- pourquoi ce sujet : cas d'usage concret, dataset adapté au format court,
  bon exercice de transfer learning sur du texte
- objectif : un modèle qui marche et qui est déployé, pas un projet
  ambitieux jamais terminé

## 3. données & entraînement (rayen)

- dataset `dair-ai/emotion` (hugging face), ~20k exemples, 6 classes
- modèle de base : `distilbert-base-uncased`
- fine-tuning avec `Trainer` de `transformers`
- résultats :
  - accuracy : 91.8%
  - macro F1 : 0.877
  - weighted F1 : 0.918
- montrer la matrice de confusion (`confusion_matrix.png`)

## 4. API (julien)

- FastAPI, endpoint `POST /predict`
- chargement du modèle sauvegardé par rayen
- entrée : `{ "text": "..." }`
- sortie : `{ "emotion": "...", "confidence": ... }`
- tests automatisés avec pytest

## 5. logique métier (redouane)

- couche entre le front et l'API
- gestion des erreurs et du formatage de la réponse

## 6. front (steven)

- interface Gradio
- champ texte → bouton "Analyser" → résultat affiché dans un bloc coloré
- une couleur par émotion pour une lecture rapide
- choix de Gradio : rapide à mettre en place, déployable directement sur
  Hugging Face Spaces, pas de HTML/JS à écrire

## 7. démo live

- ouvrir l'app déployée sur Hugging Face Spaces
- tester plusieurs phrases (positive, négative, neutre)
- montrer le résultat coloré + score de confiance

## 8. ce qu'on a appris / conclusion

- le transfer learning fonctionne bien même sur un format court
- le déploiement est une étape à part entière, pas un détail
- chaque membre a tenu son rôle : données → modèle → API → front
