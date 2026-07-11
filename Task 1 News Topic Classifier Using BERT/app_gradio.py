"""
app_gradio.py
-------------
Gradio alternative to app.py for live news headline classification.

Run:
    python app_gradio.py
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./bert-agnews/final"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    return {name: float(p) for name, p in zip(LABEL_NAMES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news headline..."),
    outputs=gr.Label(num_top_classes=4),
    title="News Topic Classifier (BERT fine-tuned on AG News)",
    description="Classifies headlines into World, Sports, Business, or Sci/Tech.",
    examples=[
        "Federal Reserve raises interest rates amid inflation concerns",
        "Local team wins championship in dramatic overtime finish",
        "New exoplanet discovered by space telescope",
        "Tensions rise as peace talks stall in the region",
    ],
)

if __name__ == "__main__":
    demo.launch()
