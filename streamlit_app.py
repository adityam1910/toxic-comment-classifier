import re
import joblib
import streamlit as st

@st.cache_resource
def load_models():
    return (joblib.load("vectorizer.joblib"),
            joblib.load("nb_model.joblib"),
            joblib.load("lr_model.joblib"))

vec, nb, lr = load_models()
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("Toxic Comment Classifier")
st.write("Classical ML models trained on the Jigsaw Toxic Comment dataset.")

model_name = st.radio("Model", ["Logistic Regression", "Naive Bayes"])
comment = st.text_area("Enter a comment")

if st.button("Classify") and comment.strip():
    X = vec.transform([clean_text(comment)])
    model = lr if model_name == "Logistic Regression" else nb
    probs = model.predict_proba(X)[0]
    flagged = [l for l, p in zip(labels, probs) if p >= 0.5]
    if flagged:
        st.error("Flagged as: " + ", ".join(flagged))
    else:
        st.success("No category above 50%")
    for label, p in zip(labels, probs):
        st.write(f"{label}: {p:.0%}")
        st.progress(float(p))
