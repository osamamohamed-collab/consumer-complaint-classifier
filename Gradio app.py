import json, re, pickle
import numpy as np
import gradio as gr
import nltk

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ART = "artifacts"
info = json.load(open(f"{ART}/best_model_info.json"))
CLASSES, MAX_LEN = info["classes"], info["max_len"]
STOP, LEMM = set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"x{2,}", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(LEMM.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2)


def light_clean(t):
    return re.sub(r"\s+", " ", re.sub(r"x{2,}", " ", str(t).lower())).strip()


if info["type"] == "transformer":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(f"{ART}/transformer")
    model = AutoModelForSequenceClassification.from_pretrained(f"{ART}/transformer").eval()

    def get_probs(text):
        enc = tokenizer(light_clean(text), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            return torch.softmax(model(**enc).logits, -1)[0].numpy()
else:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model = tf.keras.models.load_model(f"{ART}/keras_model.keras")
    tok = pickle.load(open(f"{ART}/tokenizer.pkl", "rb"))

    def get_probs(text):
        seq = pad_sequences(tok.texts_to_sequences([clean_text(text)]), maxlen=MAX_LEN,
                            padding="post", truncating="post")
        return model.predict(seq, verbose=0)[0]


def predict(text):
    if not text or not text.strip():
        return "Please enter a complaint.", {}
    probs = get_probs(text)
    top = int(np.argmax(probs))
    summary = f"Predicted category: {CLASSES[top]}\nConfidence: {probs[top]*100:.2f}%"
    return summary, {c: float(p) for c, p in zip(CLASSES, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, label="Complaint narrative", placeholder="Paste the customer complaint here..."),
    outputs=[gr.Textbox(label="Result"), gr.Label(num_top_classes=len(CLASSES), label="Class probabilities")],
    title="Consumer Complaint Classifier",
    description="Enter a complaint narrative and get the predicted category with a confidence score.",
    examples=[["I was charged a late fee on my credit card even though I paid on time."],
              ["A debt collector keeps calling me about a debt that is not mine."]],
)

if __name__ == "__main__":
    demo.launch()
