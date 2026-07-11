"""
app.py
------
Streamlit app for live news headline classification using the
fine-tuned BERT model.

Run:
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./bert-agnews/final"   # change if your saved model lives elsewhere
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1))
    return LABEL_NAMES[pred_idx], probs


def main():
    st.set_page_config(page_title="News Topic Classifier", page_icon="📰")
    st.title("📰 News Topic Classifier (BERT fine-tuned on AG News)")
    st.write(
        "Enter a news headline or short article snippet and the model will "
        "classify it into one of four topics: **World, Sports, Business, Sci/Tech**."
    )

    tokenizer, model = load_model()

    text = st.text_area("Headline / text", placeholder="e.g. Apple unveils new AI chip for data centers")

    if st.button("Classify") and text.strip():
        label, probs = predict(text, tokenizer, model)
        st.subheader(f"Prediction: **{label}**")

        st.write("Class probabilities:")
        for name, p in zip(LABEL_NAMES, probs):
            st.write(f"{name}: {p:.2%}")
            st.progress(p)


if __name__ == "__main__":
    main()
