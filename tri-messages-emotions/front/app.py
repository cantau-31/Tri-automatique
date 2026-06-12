import gradio as gr
import requests

# ── Config ──────────────────────────────────────────────
EMOTIONS = {"joy":"😄","sadness":"😢","anger":"😠","fear":"😨","love":"❤️","surprise":"😲"}
EMPTY_RESULT = "<div style='height:80px'></div>"

# ── Modèle ──────────────────────────────────────────────
COLORS = {"joy":"#f59e0b","sadness":"#3b82f6","anger":"#ef4444","fear":"#8b5cf6","love":"#ec4899","surprise":"#10b981"}

def predict(text):
    if not text.strip():
        return EMPTY_RESULT
    
    try:
        res = requests.post("http://localhost:8000/predict", json={"text": text}, timeout=10)
        res.raise_for_status()
    except requests.RequestException as error:
        return f"""
        <div style='margin-top:12px;padding:20px 24px;border-radius:16px;
                    background:rgba(239,68,68,0.08);border:1px solid #ef444466;
                    color:#ef4444;font-weight:600'>
            API indisponible : {error}
        </div>"""

    data = res.json()
    emotion = data["emotion"]
    confidence = data["confidence"]
    
    color = COLORS.get(emotion, "#6b7280")
    warning = ""
    if confidence < 0.45:
        warning = "<div style='font-size:0.85rem;color:#6b7280;margin-top:6px'>confiance faible</div>"

    return f"""
    <div style='margin-top:12px;padding:20px 24px;border-radius:16px;
                background:rgba(255,255,255,0.05);border:1px solid {color}44;
                display:flex;align-items:center;justify-content:space-between'>
        <span style='font-size:2rem'>{EMOTIONS.get(emotion, "🤔")}</span>
        <span style='font-size:1.3rem;font-weight:700;color:{color}'>{emotion}{warning}</span>
        <span style='background:{color}22;color:{color};padding:6px 14px;
                     border-radius:20px;font-weight:600'>{confidence*100:.1f}%</span>
    </div>"""


def clear_result(_text):
    """Efface l'ancienne prediction quand le message change."""
    return EMPTY_RESULT

# ── Style ───────────────────────────────────────────────
css = """
body { background: #ffffff !important; }
.gradio-container { max-width: 700px !important; margin: 40px auto !important; }
button.primary { background: linear-gradient(135deg,#7c3aed,#4f46e5) !important; border: none !important; border-radius: 14px !important; color: white !important; font-weight: 600 !important; }
button.secondary { background: #f9fafb !important; border: 1px solid #d1d5db !important; border-radius: 14px !important; color: #374151 !important; }
label { color: #4b5563 !important; font-size: 0.85rem !important; }
"""

# ── Interface ───────────────────────────────────────────
with gr.Blocks() as demo:
    gr.HTML("<h1 style='color:#111827;text-align:center;margin:20px 0 4px'>🎭 Détecteur d'émotions</h1><p style='color:#6b7280;text-align:center;margin-bottom:24px'>Entre un texte — le modèle détecte l'émotion associée</p>")

    inp = gr.Textbox(lines=4, placeholder="Ex: I'm so happy today!", label="Ton message")
    result_box = gr.HTML(EMPTY_RESULT)
    

    with gr.Row():
        gr.Button("Analyser ✨", variant="primary").click(predict, inp, result_box)
        gr.Button("Effacer", variant="secondary").click(lambda: ("", EMPTY_RESULT), outputs=[inp, result_box])

    inp.change(clear_result, inp, result_box)
    inp.submit(predict, inp, result_box)

# ── Lancement ───────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(css=css, theme=gr.themes.Base())
