"""API FastAPI pour classer un message selon son emotion."""

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


# Chemin vers le dossier ou le script d'entrainement sauvegardera le modele.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data_training" / "saved_model"


class PredictRequest(BaseModel):
    """Donnees envoyees par l'utilisateur pour faire une prediction."""

    text: str = Field(..., min_length=1, examples=["I am happy today"])


class PredictResponse(BaseModel):
    """Format de reponse renvoye par l'API."""

    emotion: str
    confidence: float


app = FastAPI(
    title="Tri Messages Emotions API",
    description="API de prediction d'emotion pour des messages en anglais.",
    version="1.0.0",
)

# Ces variables seront remplies au premier appel de /predict.
tokenizer = None
model = None


def load_saved_model():
    """Charge le tokenizer et le modele depuis data_training/saved_model/."""
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return tokenizer, model

    if not (MODEL_DIR / "config.json").exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Modele non trouve. Lancez d'abord l'entrainement pour remplir "
                "data_training/saved_model/."
            ),
        )

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de charger le modele sauvegarde : {error}",
        ) from error

    return tokenizer, model


@app.get("/")
def home():
    """Endpoint simple pour verifier que l'API est lancee."""
    return {"message": "API de tri automatique des messages par emotion"}


@app.get("/health")
def health():
    """Indique si un modele entraine est disponible."""
    model_available = (MODEL_DIR / "config.json").exists()

    return {
        "api": "ok",
        "model_available": model_available,
        "model_path": str(MODEL_DIR),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predire l'emotion principale d'un message."""
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas etre vide.")

    current_tokenizer, current_model = load_saved_model()

    # Transformation du texte en tenseurs utilisables par DistilBERT.
    inputs = current_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    # Pas de calcul de gradient en prediction.
    with torch.no_grad():
        outputs = current_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_index = predicted_class.item()
    emotion = current_model.config.id2label.get(predicted_index)
    if emotion is None:
        emotion = current_model.config.id2label.get(str(predicted_index), str(predicted_index))

    return {
        "emotion": emotion,
        "confidence": round(confidence.item(), 2),
    }
