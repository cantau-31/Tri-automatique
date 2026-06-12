"""Application Gradio deployable sur Hugging Face Spaces.

Cette version charge directement le modele sauvegarde localement dans
data_training/saved_model/. Elle ne depend pas de l'API FastAPI locale.
"""

from pathlib import Path

import gradio as gr
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "data_training" / "saved_model"

EMOTIONS = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

COLORS = {
    "joy": "#f59e0b",
    "sadness": "#3b82f6",
    "anger": "#ef4444",
    "fear": "#8b5cf6",
    "love": "#ec4899",
    "surprise": "#10b981",
}

EMPTY_RESULT = "<div style='height:80px'></div>"

tokenizer = None
model = None


def load_model():
    """Charge le modele une seule fois au premier appel."""
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return tokenizer, model

    if not (MODEL_DIR / "config.json").exists() or not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(
            "Modele introuvable. Verifiez que data_training/saved_model/ contient "
            "config.json, model.safetensors et tokenizer.json."
        )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model


def predict(text):
    """Predire l'emotion d'un texte depuis l'interface Gradio."""
    if not text or not text.strip():
        return EMPTY_RESULT

    current_tokenizer, current_model = load_model()

    inputs = current_tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = current_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_index = predicted_class.item()
    emotion = current_model.config.id2label.get(predicted_index)
    if emotion is None:
        emotion = current_model.config.id2label.get(str(predicted_index), str(predicted_index))

    confidence_value = round(confidence.item(), 2)
    color = COLORS.get(emotion, "#6b7280")
    warning = ""
    if confidence_value < 0.45:
        warning = "<div style='font-size:0.85rem;color:#6b7280;margin-top:6px'>confiance faible</div>"

    return f"""
    <div style='margin-top:12px;padding:20px 24px;border-radius:16px;
                background:rgba(255,255,255,0.05);border:1px solid {color}44;
                display:flex;align-items:center;justify-content:space-between'>
        <span style='font-size:2rem'>{EMOTIONS.get(emotion, "🤔")}</span>
        <span style='font-size:1.3rem;font-weight:700;color:{color}'>{emotion}{warning}</span>
        <span style='background:{color}22;color:{color};padding:6px 14px;
                     border-radius:20px;font-weight:600'>{confidence_value * 100:.1f}%</span>
    </div>"""


def clear_result(_text):
    """Efface l'ancienne prediction quand le message change."""
    return EMPTY_RESULT


CSS = """
body { background: #ffffff !important; }
.gradio-container { max-width: 700px !important; margin: 40px auto !important; }
textarea, input { color: #111827 !important; }
button.primary {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 600 !important;
}
button.secondary {
    background: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
    color: #374151 !important;
}
label { color: #4b5563 !important; font-size: 0.85rem !important; }
"""


with gr.Blocks() as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML(
        "<h1 style='color:#111827;text-align:center;margin:20px 0 4px'>"
        "🎭 Détecteur d'émotions</h1>"
        "<p style='color:#6b7280;text-align:center;margin-bottom:24px'>"
        "Entre un texte — le modèle détecte l'émotion associée</p>"
    )

    inp = gr.Textbox(
        lines=4,
        placeholder="Ex: I'm so happy today!",
        label="Ton message",
    )
    result_box = gr.HTML(EMPTY_RESULT)

    with gr.Row():
        gr.Button("Analyser ✨", variant="primary").click(predict, inp, result_box)
        gr.Button("Effacer", variant="secondary").click(
            lambda: ("", EMPTY_RESULT),
            outputs=[inp, result_box],
        )

    inp.change(clear_result, inp, result_box)
    inp.submit(predict, inp, result_box)


if __name__ == "__main__":
    demo.launch()
