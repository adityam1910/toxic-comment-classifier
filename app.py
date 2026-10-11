import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO = "arm19/toxic-comment-bilstm"

MAX_LENGTH = 200

LABELS = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="toxic_bilstm.keras"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    tokenizer_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="tokenizer.pkl"
    )

    with open(tokenizer_path, "rb") as file:

        tokenizer = pickle.load(file)

    return tokenizer


# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================

try:

    model = load_model()
    tokenizer = load_tokenizer()

except Exception as error:

    st.error(
        "Unable to load the AI model or tokenizer."
    )

    st.exception(error)

    st.stop()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_toxicity(comment):

    # Convert text into integer sequence
    sequence = tokenizer.texts_to_sequences(
        [comment]
    )

    # Pad the sequence
    padded_sequence = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Get predictions
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    return probabilities


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛡️ Toxic Comment Detector"
)

st.write(
    "Detect different types of toxic content "
    "in social media comments using Deep Learning."
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "This application uses a Bi-LSTM model trained "
    "on the Jigsaw Toxic Comment Classification dataset."
)


# ============================================================
# USER INPUT
# ============================================================

comment = st.text_area(
    "Enter a social media comment:",
    height=150,
    placeholder="Example: You are an idiot..."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Comment",
    use_container_width=True
):

    # Check empty input
    if not comment.strip():

        st.warning(
            "Please enter a comment before analyzing."
        )

    else:

        # Get predictions
        probabilities = predict_toxicity(
            comment
        )

        # Results heading
        st.subheader(
            "Prediction Results"
        )

        # Display each label
        for label, probability in zip(
            LABELS,
            probabilities
        ):

            percentage = probability * 100

            st.write(
                f"**{label}: {percentage:.2f}%**"
            )

            st.progress(
                float(probability)
            )

        # ----------------------------------------------------
        # Overall toxicity
        # ----------------------------------------------------

        st.divider()

        toxic_probability = probabilities[0]

        if toxic_probability >= 0.5:

            st.error(
                "⚠️ This comment is classified as TOXIC."
            )

        else:

            st.success(
                "✅ This comment is classified as NON-TOXIC."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI/ML Project — Multilabel Toxic Comment Detection"
)
