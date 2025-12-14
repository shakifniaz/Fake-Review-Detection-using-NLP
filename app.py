import string
import pickle
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

st.markdown("""
<style>
:root {
    --accent: #f97316;
    --accent-dark: #ea580c;
    --text-main: #111827;
    --text-muted: #374151;
    --border: #e5e7eb;
}

.stApp {
    background-color: #ffffff;
}

html, body, p, span, label, div {
    color: var(--text-main) !important;
}

.top-nav {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 64px;
    background: #ffffff;
    border-bottom: 2px solid var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.top-nav-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--accent) !important;
    letter-spacing: 0.3px;
}

.block-container {
    padding-top: 90px !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4 {
    color: var(--accent) !important;
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 600;
    color: var(--text-main) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    color: var(--text-muted) !important;
}

textarea {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    font-size: 1rem !important;
}

textarea::placeholder {
    color: #9ca3af !important;
}

.stButton > button {
    background-color: var(--accent);
    color: white !important;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    font-weight: 600;
    border: none;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: var(--accent-dark);
    transform: scale(1.02);
}

.fake-bar > div > div > div {
    background-color: var(--accent) !important;
}
.fake-bar > div {
    background-color: #e5e7eb !important;
}

.real-bar > div > div > div {
    background-color: #3b82f6 !important;
}
.real-bar > div {
    background-color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-nav">
    <div class="top-nav-title">🕵️ Fake Review Detector</div>
</div>
""", unsafe_allow_html=True)

with open("model_metrics.json", "r") as f:
    metrics = json.load(f)


col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Model Precision", f"{metrics['precision_CG']*100:.2f}%")


ps = PorterStemmer()

def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)

    y = [t for t in tokens if t.isalnum()]
    y = [t for t in y if t not in stopwords.words('english') and t not in string.punctuation]
    y = [ps.stem(t) for t in y]
    return " ".join(y)

def featurize_single_review(raw_text: str) -> pd.DataFrame:
    raw_text = str(raw_text)
    return pd.DataFrame([{
        'transformed_text': transform_text(raw_text),
        'num_characters': len(raw_text),
        'num_words': len(nltk.word_tokenize(raw_text)),
        'num_sentences': len(nltk.sent_tokenize(raw_text))
    }])

pipe = pickle.load(open('fake_reviews_pipe.pkl', 'rb'))


input_review = st.text_area("Paste a review text")

if st.button("Predict"):
    X_in = featurize_single_review(input_review)

    result = pipe.predict(X_in)[0]

    proba = pipe.predict_proba(X_in)[0]
    classes = pipe.classes_
    idx_cg = list(classes).index('CG')
    p_cg = float(proba[idx_cg])
    p_or = 1.0 - p_cg

    if result == 'CG':
        st.header("Fake Review")
    else:
        st.header("Real Review")

    st.subheader("Confidence meter")

    st.metric("Fake review probability", f"{p_cg * 100:.2f}%")
    st.markdown('<div class="fake-bar">', unsafe_allow_html=True)
    st.progress(int(p_cg * 100))
    st.markdown('</div>', unsafe_allow_html=True)

    st.metric("Real review probability", f"{p_or * 100:.2f}%")
    st.markdown('<div class="real-bar">', unsafe_allow_html=True)
    st.progress(int(p_or * 100))
    st.markdown('</div>', unsafe_allow_html=True)


