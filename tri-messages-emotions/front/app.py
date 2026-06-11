import gradio as gr
import requests

# ── Config ──────────────────────────────────────────────
EMOTIONS = {"joy":"😄","sadness":"😢","anger":"😠","fear":"😨","love":"❤️","surprise":"😲"}

# ── Modèle ──────────────────────────────────────────────
COLORS = {"joy":"#f59e0b","sadness":"#3b82f6","anger":"#ef4444","fear":"#8b5cf6","love":"#ec4899","surprise":"#10b981"}

def predict(text):
    if not text.strip(): return "<div style='height:80px'></div>"
    
    res = requests.post("http://localhost:8000/predict", json={"text": text})
    data = res.json()
    emotion = data["emotion"]
    confidence = data["confidence"]
    
    color = COLORS.get(emotion, "#6b7280")
    return f"""
    <div style='margin-top:12px;padding:20px 24px;border-radius:16px;
                background:rgba(255,255,255,0.05);border:1px solid {color}44;
                display:flex;align-items:center;justify-content:space-between'>
        <span style='font-size:2rem'>{EMOTIONS.get(emotion, "🤔")}</span>
        <span style='font-size:1.3rem;font-weight:700;color:{color}'>{emotion}</span>
        <span style='background:{color}22;color:{color};padding:6px 14px;
                     border-radius:20px;font-weight:600'>{confidence*100:.1f}%</span>
    </div>"""

# ── Style ───────────────────────────────────────────────
css = """
body { background: #0d1117 !important; }
.gradio-container { max-width: 700px !important; margin: 40px auto !important; }
textarea, input { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 14px !important; color: white !important; }
textarea:focus { border-color: rgba(139,92,246,0.6) !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important; }
button.primary { background: linear-gradient(135deg,#7c3aed,#4f46e5) !important; border: none !important; border-radius: 14px !important; color: white !important; font-weight: 600 !important; }
button.secondary { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 14px !important; color: rgba(255,255,255,0.6) !important; }
label { color: rgba(255,255,255,0.6) !important; font-size: 0.85rem !important; }
"""

# ── Interface ───────────────────────────────────────────
with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    gr.HTML("<h1 style='color:white;text-align:center;margin:20px 0 4px'>🎭 Détecteur d'émotions</h1><p style='color:rgba(255,255,255,0.5);text-align:center;margin-bottom:24px'>Entre un texte — le modèle détecte l'émotion associée</p>")

    inp = gr.Textbox(lines=4, placeholder="Ex: I'm so happy today!", label="Ton message")
    result_box = gr.HTML("<div style='height:80px'></div>")
    

    with gr.Row():
        gr.Button("Analyser ✨", variant="primary").click(predict, inp, result_box)
        gr.Button("Effacer", variant="secondary").click(lambda: ("", "<div style='height:80px'></div>"), outputs=[inp, result_box])

# ── Lancement ───────────────────────────────────────────
demo.launch()
