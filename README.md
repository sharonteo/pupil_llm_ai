# Pupillometry Clinical Dashboard + LLM Narratives

# Pupillometry Clinical Dashboard + LLM Narratives  
🔗 **Live Demo:** https://pupillometer.streamlit.app

A lightweight Streamlit app for exploring synthetic pupillometry data, running ML models, and generating FDA‑style narratives using Claude.

---

## Overview

This project includes:

- Synthetic pupillometry dataset (GCS, NPi, pupil metrics, diagnosis labels)
- Basic ML models (Logistic Regression, Random Forest, XGBoost)
- Interactive Streamlit visualizations
- Optional LLM‑powered narrative generation (Anthropic Claude)

The app is designed for quick clinical analytics demos and regulatory‑style summaries.

---
## Project Structure
```
pupil_llm_ai/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   ├── models/
│   └── data/
│
├── requirements.txt
└── .streamlit/runtime.txt
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Launch the app
streamlit run app/app.py

Anthropic API Key (Optional)
To enable FDA‑style narrative generation:
- In Streamlit Cloud, open Manage App → Secrets
- Add: ANTHROPIC_API_KEY = "your-key-here"


pandas==2.2.3
numpy==1.26.4
scikit-learn==1.8.0
xgboost==3.2.0
matplotlib==3.9.2
seaborn==0.13.2
plotly==6.6.0
streamlit==1.32.0
altair==4.2.2
anthropic==0.85.0
statsmodels==0.14.0   # Not used in app (trendlines removed)