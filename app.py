import string
import pickle
import json
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# If you get LookupError on first run, uncomment once:
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")

# -----------------------------
# Page + CSS
# -----------------------------
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")

st.markdown("""
<style>
:root {
    --accent: #f97316;
    --accent-dark: #ea580c;
    --text-main: #111827;
    --text-muted: #374151;
    --border: #e5e7eb;
}

.stApp { background-color: #ffffff; }

html, body, p, span, label, div {
    color: var(--text-main) !important;
}

/* ===== NAVBAR ===== */
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
    font-weight: 900;
    color: var(--accent) !important;
    letter-spacing: 0.3px;
}

.block-container { padding-top: 90px !important; }

header[data-testid="stHeader"] { background: transparent; }

/* ===== HEADINGS ===== */
h1, h2, h3, h4 {
    color: var(--accent) !important;
    font-weight: 800;
}

/* ===== TEXT AREA ===== */
textarea {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    font-size: 1rem !important;
}
textarea::placeholder { color: #9ca3af !important; }

/* ===== BUTTON ===== */
.stButton > button {
    background-color: var(--accent);
    color: white !important;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    font-weight: 700;
    border: none;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: var(--accent-dark);
    transform: scale(1.02);
}

/* ===== PROGRESS BARS ===== */
.fake-bar > div > div > div { background-color: var(--accent) !important; }
.fake-bar > div { background-color: #e5e7eb !important; }

.real-bar > div > div > div { background-color: #3b82f6 !important; }
.real-bar > div { background-color: #e5e7eb !important; }

/* Pills */
.pill {
  display:inline-block;
  padding:.25rem .6rem;
  border-radius:999px;
  font-weight:800;
  font-size:.85rem;
  background:rgba(249,115,22,.1);
  color:var(--accent-dark) !important;
}
.pill-blue {
  background:rgba(59,130,246,.1);
  color:#1d4ed8 !important;
}

/* ===== METRIC CARDS (rectangular, shadow) ===== */
.metric-card {
    background: #ffffff;
    border: 1px solid rgba(229, 231, 235, 0.95);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 10px 26px rgba(17, 24, 39, 0.08);
}
.metric-title {
    font-size: 0.9rem;
    font-weight: 800;
    color: #6b7280 !important;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 900;
    color: #111827 !important;
}

/* ===== SMALL INFO CARDS (Confidence / Risk) ===== */
.info-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 6px 16px rgba(17,24,39,0.06);
}

.info-title {
    font-size: 0.75rem;
    font-weight: 800;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
}

.info-value {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-nav">
    <div class="top-nav-title">🕵️ Fake Review Detector</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Load metrics + model
# -----------------------------
try:
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {"accuracy": None, "precision_CG": None}

pipe = pickle.load(open("fake_reviews_pipe.pkl", "rb"))

# -----------------------------
# Preprocessing
# -----------------------------
ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)

def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    return " ".join(
        ps.stem(t) for t in tokens
        if t.isalnum() and t not in STOPWORDS and t not in PUNCT
    )

def featurize_single_review(raw_text: str) -> pd.DataFrame:
    raw_text = str(raw_text)
    return pd.DataFrame([{
        "transformed_text": transform_text(raw_text),
        "num_characters": len(raw_text),
        "num_words": len(nltk.word_tokenize(raw_text)),
        "num_sentences": len(nltk.sent_tokenize(raw_text))
    }])

# -----------------------------
# Confidence + risk
# -----------------------------
def confidence_label_and_risk(p_cg: float):
    if p_cg >= 0.85:
        return "Very High", "High risk of Fake (CG)"
    elif p_cg >= 0.70:
        return "High", "Moderate–High risk of Fake (CG)"
    elif p_cg >= 0.55:
        return "Medium", "Borderline leaning Fake (CG)"
    elif p_cg >= 0.45:
        return "Low", "Borderline / Uncertain"
    elif p_cg >= 0.30:
        return "Medium", "Borderline leaning Real (OR)"
    else:
        return "High", "Low risk of Fake (likely Real OR)"

# -----------------------------
# Top contributing words
# -----------------------------
def get_top_contributing_words(pipe, transformed_text: str, topn=10):
    try:
        clf = pipe.named_steps["clf"]
        preprocess = pipe.named_steps["preprocess"]
        text_union = preprocess.named_transformers_["text"]

        word_vec = None
        for name, tr in text_union.transformer_list:
            if name == "word":
                word_vec = tr

        x_word = word_vec.transform([transformed_text])
        coef = clf.coef_[0]

        classes = list(clf.classes_)
        if classes[1] == "CG":
            cg_coef = coef
        else:
            cg_coef = -coef

        V = x_word.shape[1]
        cg_coef_word = cg_coef[:V]

        nz = x_word.nonzero()[1]
        inv_vocab = {i: w for w, i in word_vec.vocabulary_.items()}

        contrib = x_word.data * cg_coef_word[nz]
        rows = [(inv_vocab[i], c) for i, c in zip(nz, contrib)]

        pos = sorted([r for r in rows if r[1] > 0], key=lambda x: x[1], reverse=True)[:topn]
        neg = sorted([r for r in rows if r[1] < 0], key=lambda x: x[1])[:topn]
        return pos, neg
    except Exception:
        return [], []

# -----------------------------
# UI
# -----------------------------
col1, col2 = st.columns(2)

acc_text = "N/A" if metrics.get("accuracy") is None else f"{metrics['accuracy']*100:.2f}%"
prec_text = "N/A" if metrics.get("precision_CG") is None else f"{metrics['precision_CG']*100:.2f}%"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Accuracy</div>
        <div class="metric-value">{acc_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Precision (CG)</div>
        <div class="metric-value">{prec_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

input_review = st.text_area(
    "Paste a review text",
    placeholder="Enter a product review here...",
    height=160
)

if st.button("Predict"):
    X_in = featurize_single_review(input_review)

    result = pipe.predict(X_in)[0]
    proba = pipe.predict_proba(X_in)[0]

    classes = list(pipe.classes_)
    p_cg = float(proba[classes.index("CG")])
    p_or = 1.0 - p_cg

    st.header("Fake Review" if result == "CG" else "Real Review")

    conf_label, risk_level = confidence_label_and_risk(p_cg)

    a, b = st.columns(2)
    with a:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Confidence</div>
            <div class="info-value">{conf_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Risk Level</div>
            <div class="info-value">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Confidence meter")

    st.metric("Fake review probability (CG)", f"{p_cg*100:.2f}%")
    st.markdown('<div class="fake-bar">', unsafe_allow_html=True)
    st.progress(int(p_cg * 100))
    st.markdown('</div>', unsafe_allow_html=True)

    st.metric("Real review probability (OR)", f"{p_or*100:.2f}%")
    st.markdown('<div class="real-bar">', unsafe_allow_html=True)
    st.progress(int(p_or * 100))
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Top contributing words")
    pos_words, neg_words = get_top_contributing_words(pipe, X_in["transformed_text"].iloc[0])

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Pushing toward Fake (CG)**")
        for w, s in pos_words:
            st.markdown(f'<span class="pill">{w}</span>', unsafe_allow_html=True)

    with cB:
        st.markdown("**Pushing toward Real (OR)**")
        for w, s in neg_words:
            st.markdown(f'<span class="pill pill-blue">{w}</span>', unsafe_allow_html=True)
