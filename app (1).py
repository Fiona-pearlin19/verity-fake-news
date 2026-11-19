
import streamlit as st
import joblib
import pandas as pd
import re
import unicodedata
import plotly.graph_objects as go

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Verity | Fake News Detection",
    page_icon="🔎",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return joblib.load("fake_news_models.joblib")

bundle = load_models()

STOP = bundle["meta"]["stop_words"]
SHORT_MAX_WORDS = bundle["meta"]["short_max_words"]
BAND = bundle["meta"]["band"]

# ---------------- PREPROCESSING ----------------
def clean_text(text):
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-zà-ÿ\s]", " ", t.lower())
    return " ".join(
        w for w in t.split()
        if len(w) > 1 and w not in STOP
    )

def clean_short(text):
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\d+(?:[.,]\d+)?", " numtok ", t)
    t = t.replace("%", " pcttok ").replace("$", " dollartok ")
    return " ".join(re.sub(r"[^a-z\s]", " ", t).split())

# ---------------- PREDICTION ----------------
def get_probability(item, text):
    pipe = item["pipe"]
    raw = pipe.predict_proba(pd.Series([text]))[:, 1] \
        if hasattr(pipe, "predict_proba") \
        else pipe.decision_function(pd.Series([text]))

    if item["platt"] is not None:
        raw = item["platt"].predict_proba(
            raw.reshape(-1, 1)
        )[:, 1]

    return float(raw[0])

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #17152f);
    color: white;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(120deg, #29215c, #172b50);
    text-align: center;
    border: 1px solid #51458b;
}
.hero h1 {
    color: white;
    font-size: 48px;
    margin-bottom: 5px;
}
.hero p {
    color: #d4d5f5;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🔎 VERITY</h1>
    <p>AI-Powered Fake News Detection & Credibility Analysis</p>
    <p>Analyze information. Understand credibility. Think critically.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.subheader("📰 Analyze a News Claim")
st.write("Paste a news article or a short statement below.")

news = st.text_area(
    "Enter your news text",
    height=220,
    placeholder="Paste your news article or claim here..."
)

word_count = len(news.split())
st.caption(f"Word count: {word_count}")

if st.button("🔍 Analyze News", use_container_width=True):

    if not news.strip():
        st.warning("Please enter some news text first.")

    else:
        track = "short" if word_count <= SHORT_MAX_WORDS else "long"
        scorers = bundle[track]

        processed = clean_short(news) if track == "short" else clean_text(news)

        if not processed.strip():
            st.error("No usable text remains after preprocessing.")
        else:
            results = []

            for name, item in scorers.items():
                probability = get_probability(item, processed)
                threshold = item["threshold"]
                band = BAND[track]

                verdict = (
                    "Likely Real" if probability >= threshold + band
                    else "Likely Fake" if probability < threshold - band
                    else "Uncertain"
                )

                results.append({
                    "Model": name,
                    "Credibility": probability * 100,
                    "Verdict": verdict
                })

            avg = sum(r["Credibility"] for r in results) / len(results)
            avg_verdict = (
                "Likely Real" if avg >= 65
                else "Likely Fake" if avg < 35
                else "Uncertain"
            )

            st.divider()
            st.subheader("📊 Analysis Results")

            c1, c2, c3 = st.columns(3)

            c1.metric("Detection Track", track.upper())
            c2.metric("Credibility Score", f"{avg:.1f}%")
            c3.metric("Final Verdict", avg_verdict)

            st.progress(int(avg))

            st.subheader("🤖 Individual Model Results")

            for r in results:
                st.write(f"**{r['Model']}**")
                st.progress(int(r["Credibility"]))
                st.write(
                    f"{r['Credibility']:.1f}% credibility — {r['Verdict']}"
                )

            fig = go.Figure(go.Pie(
                labels=["Credibility", "Remaining"],
                values=[avg, 100 - avg],
                hole=0.7
            ))

            fig.update_layout(
                title="Credibility Distribution",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "This result is a machine-learning prediction based on text patterns. "
                "It does not independently verify facts. Always consult reliable "
                "news sources and fact-checking organizations."
            )

st.divider()
st.caption("VERITY | Fake News Detection using NLP and Machine Learning")
