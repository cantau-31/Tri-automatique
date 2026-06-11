"""Tests simples pour l'API FastAPI."""

from fastapi.testclient import TestClient

import api.main as api_main


client = TestClient(api_main.app)


def test_home_endpoint():
    """L'endpoint racine doit confirmer que l'API repond."""
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """L'endpoint health doit indiquer l'etat de l'API et du modele."""
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["api"] == "ok"
    assert "model_available" in data


def test_predict_without_saved_model_returns_503(monkeypatch, tmp_path):
    """Sans modele sauvegarde, /predict doit retourner une erreur claire."""
    monkeypatch.setattr(api_main, "MODEL_DIR", tmp_path / "saved_model")
    monkeypatch.setattr(api_main, "tokenizer", None)
    monkeypatch.setattr(api_main, "model", None)

    response = client.post("/predict", json={"text": "I am happy today"})

    assert response.status_code == 503
    assert "Modele non trouve" in response.json()["detail"]
