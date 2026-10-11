import streamlit as st
import tensorflow as tf
import pickle
import numpy as np


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

MAX_LENGTH = 200

LABELS = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate"
]


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "toxic_bilstm.keras"
    )

    return model


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


model = load_model()
tokenizer = load_tokenizer()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_toxicity(comment):

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [comment]
    )

    # Pad sequence
    padded_sequence = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Get probabilities
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    return probabilities


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("🛡️ Toxic Comment Detector")

st.write(
    "Enter a social media comment below to analyze "
    "whether it contains different types of toxic content."
)

st.info(
    "This application uses a Bi-LSTM deep learning model "
    "trained on the Jigsaw Toxic Comment dataset."
)


# --------------------------------------------------
# Text input
# --------------------------------------------------

comment = st.text_area(
    "Enter your comment:",
    height=150,
    placeholder="Example: You are an idiot..."
)


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if st.button(
    "🔍 Analyze Comment",
    use_container_width=True
):

    if comment.strip() == "":
        
        st.warning(
            "Please enter a comment first."
        )

    else:

        probabilities = predict_toxicity(
            comment
        )

        st.subheader(
            "Prediction Results"
        )

        # Display results
        for label, probability in zip(
            LABELS,
            probabilities
        ):

            percentage = probability * 100

            col1, col2 = st.columns(
                [3, 1]
            )

            with col1:
                st.write(label)

            with col2:
                st.write(
                    f"{percentage:.2f}%"
                )

            st.progress(
                float(probability)
            )

        # Overall prediction
        toxic_probability = probabilities[0]

        st.divider()

        if toxic_probability >= 0.5:

            st.error(
                "⚠️ This comment is classified as TOXIC."
            )

        else:

            st.success(
                "✅ This comment is classified as NON-TOXIC."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI/ML Project — Multilabel Toxic Comment Detection"
)

from huggingface_hub import hf_hub_download

MODEL_REPO = "arm19/toxic-comment-bilstm"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="toxic_bilstm.keras"
)

tokenizer_path = hf_hub_download(where to tyo    repo_id=MODEL_REPO,
    filename="tokenizer.pkl"
)
model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
