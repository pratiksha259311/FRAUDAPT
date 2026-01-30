🛡️ FRAUDAPT — Scam Message Risk Analyzer
📌 Overview

FRAUDAPT is an AI-powered demo application that helps assess the risk level of suspicious messages such as phishing texts, lottery scams, and fake subscription alerts.

It uses semantic similarity instead of simple keywords, making it more flexible against new or slightly modified scam messages.

🚨 Problem Statement

Scam and phishing messages are increasing rapidly, but most users lack a quick way to judge whether a message is dangerous before clicking links or sharing sensitive information.

Traditional rule-based systems often fail when scam text is reworded.

💡 Solution

FRAUDAPT compares a user-provided message against known scam patterns using vector embeddings and calculates a risk score based on semantic similarity.

Instead of exact text matching, it understands meaning.

⚙️ How It Works

User pastes a suspicious message

Message is converted into embeddings using MiniLM-L6-v2

Vectors are searched in a Qdrant vector database

Top similar scam cases are returned

A risk score is generated based on similarity

🧠 Tech Stack

Python

Sentence Transformers (MiniLM-L6-v2)

Qdrant (Vector Database – Remote)

Streamlit (UI)

🎯 Features

Semantic scam detection (not keyword-based)

Risk classification: Low / Medium / High

Real-time similarity search

Clean and simple Streamlit interface

📊 Sample Use Cases

Phishing SMS detection

Lottery & prize scams

Fake subscription renewal alerts

Suspicious payment requests

⚠️ Limitations

Uses a small demo dataset for illustration

Not a replacement for production-grade fraud detection systems

Risk score is indicative, not definitive

🚀 Future Improvements

Larger labeled scam dataset

Multilingual message support

Threshold-based automated alerts

Integration with SMS / email pipelines

🔐 Security Note

Sensitive credentials (API keys, URLs) are managed via environment variables and are not hard-coded in production setups.

🧪 Demo

This project is intended for educational and demonstration purposes.

Paste a message → analyze similarity → understand fraud risk.

👤 Author

Built with ❤️ to explore applied NLP, vector databases, and real-world fraud detection concepts.
