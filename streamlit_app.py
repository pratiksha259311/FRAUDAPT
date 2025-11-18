import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Sample fraud cases
sample_data = [
    {"case": "Your bank account is blocked. Click this link to verify your identity.", "label": "Phishing"},
    {"case": "Congratulations! You won 10,00,000 INR. Fill your card details to claim.", "label": "Lottery Scam"},
    {"case": "Your Netflix subscription expired. Pay ₹499 immediately to avoid account suspension.", "label": "Subscription Scam"},
]

# Encode sample cases
sample_vectors = [model.encode(item["case"]) for item in sample_data]

# Search function
def search_case(user_text):
    query_vec = model.encode(user_text).reshape(1, -1)
    similarities = cosine_similarity(query_vec, sample_vectors)[0]
    results = []
    for idx, score in enumerate(similarities):
        results.append({
            "case": sample_data[idx]["case"],
            "label": sample_data[idx]["label"],
            "score": score
        })
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:3]

# Risk calculation
def calculate_risk(similarity_score):
    risk = round(similarity_score * 100, 2)
    if risk < 40:
        return f"{risk}% (Low Risk)"
    elif risk < 75:
        return f"{risk}% (Medium Risk)"
    else:
        return f"{risk}% (HIGH RISK ⚠️)"

# Streamlit UI
st.set_page_config(page_title="FraudAPT Demo", layout="centered")
st.title("🛡️ FraudAPT — Scam Message Detector")
st.write("Paste any suspicious message and get instant fraud detection using AI.")

user_input = st.text_area("Enter suspicious message:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please type something.")
    else:
        st.subheader("🔍 Results:")
        results = search_case(user_input)
        for i, r in enumerate(results):
            st.write(f"**Match {i+1}:**")
            st.write(f"- **Similar Case:** {r['case']}")
            st.write(f"- **Category:** {r['label']}")
            st.write(f"- **Similarity Score:** {calculate_risk(r['score'])}")
            st.markdown("---")

st.info("Model: MiniLM-L6-v2 • Vector DB: Removed, using local embeddings only")
