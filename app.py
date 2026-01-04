import streamlit as st
from transformers import pipeline

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Emotion Analysis using BERT",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Movie Review Emotion Analysis")
st.write(
    "This application uses a **Transformer (BERT-based) model** to understand "
    "the **emotion behind movie reviews**, not just positive or negative sentiment."
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_classifier = load_emotion_model()

# ------------------ USER INPUT ------------------
review = st.text_area(
    "✍ Enter your movie review:",
    placeholder="Example: The movie was slow but the climax was brilliant"
)

# ------------------ PREDICTION ------------------
if st.button("🔍 Analyze Emotion"):
    if review.strip():
        predictions = emotion_classifier(review)[0]

        # Sort emotions by confidence
        predictions = sorted(
            predictions,
            key=lambda x: x["score"],
            reverse=True
        )

        top_emotion = predictions[0]

        st.success(
            f"**Emotion:** {top_emotion['label'].capitalize()} \n\n"
            f"**Confidence:** {top_emotion['score']:.2f}"
        )

        st.subheader("🔎 Emotion Breakdown")
        for emo in predictions[:5]:
            st.write(f"{emo['label'].capitalize()} : {emo['score']:.2f}")

    else:
        st.warning("Please enter a review to analyze.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Powered by Transformer (BERT) | HuggingFace | Streamlit")
