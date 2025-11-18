import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
import numpy as np
import uuid

# -------------------------------
# 1) Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# 2) Connect to Remote Qdrant
# -------------------------------
client = QdrantClient(
    url="YOUR_CLUSTER_URL",      # replace with your remote cluster URL
    api_key="YOUR_API_KEY"       # replace with your remote API key
)

COLLECTION_NAME = "fraudapt_cases"

# -------------------------------
# 3) Ensure Collection Exists
# -------------------------------
try:
    client.get_collection(COLLECTION_NAME)
except Exception as e:
    st.warning(f"Collection might already exist or cannot be created: {e}")

# -------------------------------
# 4) Seed Sample Fraud Cases (if empty)
# -------------------------------
sample_data = [
    {"case": "Your bank account is blocked. Click this link to verify your identity.", "label": "Phishing"},
    {"case": "Congratulations! You won 10,00,000 INR. Fill your card details to claim.", "label": "Lottery Scam"},
    {"case": "Your Netflix subscription expired. Pay ₹499 immediately to avoid account suspension.", "label": "Subscription Scam"},
]

def seed_sample_cases():
    try:
        existing = client.count(COLLECTION_NAME).result.count
    except:
        existing = 0

    if existing == 0:
        vectors = []
        payloads = []
        ids = []

        for item in sample_data:
            vec = model.encode(item["case"]).tolist()
            vectors.append(vec)
            payloads.append(item)
            ids.append(str(uuid.uuid4()))

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )

seed_sample_cases()

# -------------------------------
# 5) Search Function
# -------------------------------
def search_case(user_text):
    query_vec = model.encode(user_text).tolist()

    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=3
        )
    except Exception as e:
        st.error(f"Error searching Qdrant: {e}")
        results = []

    return results

# -------------------------------
# 6) Risk Score Logic
# -------------------------------
def calculate_risk(similarity_score):
    risk = round(similarity_score * 100, 2)
    if risk < 40:
        return f"{risk}% (Low Risk)"
    elif risk < 75:
        return f"{risk}% (Medium Risk)"
    else:
        return f"{risk}% (HIGH RISK ⚠️)"

# -------------------------------
# 7) Streamlit UI
# -------------------------------
st.set_page_config(page_title="FraudAPT Demo", layout="centered")
st.title("🛡️ FraudAPT — Scam Message Detector")
st.write("Paste any suspicious message and get instant fraud detection using AI + Vector Database.")

user_input = st.text_area("Enter suspicious message:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please type something.")
    else:
        st.subheader("🔍 Results:")
        results = search_case(user_input)

        if not results:
            st.info("No similar cases found in the database.")
        else:
            for i, r in enumerate(results):
                st.write(f"**Match {i+1}:**")
                st.write(f"- **Similar Case:** {r.payload['case']}")
                st.write(f"- **Category:** {r.payload['label']}")
                st.write(f"- **Similarity Score:** {calculate_risk(r.score)}")
                st.markdown("---")

st.info("Model: MiniLM-L6-v2 • Vector DB: Qdrant (Remote)")
