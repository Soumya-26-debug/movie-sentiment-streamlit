import streamlit as st
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")

@st.cache_resource
def download_nltk():
    nltk.download('movie_reviews')
    nltk.download('punkt')

download_nltk()

@st.cache_resource
def train_model():
    documents = []
    labels = []

    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(movie_reviews.raw(fileid))
            labels.append(1 if category == "pos" else 0)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(documents)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=50
    )

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model()
st.info(f"Model Accuracy: {accuracy:.2f}")

review = st.text_area("Enter movie review")

if st.button("Predict Sentiment"):
    if review.strip():
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        st.success("Positive 😀" if pred == 1 else "Negative 😞")
    else:
        st.warning("Please enter a review")
