# Fake-news-detection

# 🧠 AI Fact-Check & News Verification Dashboard (2025)

### 📊 A Hybrid LLM + ML Powered Fake News Detection & Evidence Analysis System

---

## 🚀 Project Overview

The **AI Fact-Check Dashboard** is a next-generation data analytics project designed to automatically **verify the truthfulness of news claims, articles, and URLs** using a combination of:

- **Google Gemini 1.5 / 2.0 Pro (via Vertex API)**  
- **OpenAI GPT-5 reasoning model**  
- **Custom Machine Learning classifier (XGBoost / Logistic Regression hybrid)**  
- **Real-time evidence retrieval via Google Custom Search, NewsAPI, and GNews**  

The system serves as a **research-grade capstone project for Data Analysts** exploring hybrid AI–ML systems, retrieval-augmented generation (RAG), and media verification pipelines.

---

## 🧩 Core Features

✅ **LLM + ML Hybrid Reasoning**  
> Combines Gemini’s structured analysis and GPT-5’s logical reasoning with a machine learning truth score trained on recent news datasets.

✅ **Real-Time Evidence Retrieval**  
> Integrates Google Custom Search, GNews, and RSS pipelines to fetch live, verifiable sources supporting or refuting the input claim.

✅ **Automatic Claim Type Routing**  
> Detects whether the input is historical (Gemini), recent (GPT-5), or predictive (ML model fallback).

✅ **Dual Verdict Logic**  
> Provides both a *reasoned verdict* (LLM) and a *numerical truth score* (ML model) with visual indicators.

✅ **Dynamic Dashboard (Streamlit)**  
> Beautifully designed Streamlit UI for interactive exploration of claims, verdicts, evidence, and analytics.

✅ **Explainable Results**  
> Every result includes:  
> - ✅ Verdict  
> - 💬 Reasoning  
> - 🔍 Evidence sources  
> - 📈 Truth score

---

## 🧠 System Architecture



User Input → Data Pipeline → Evidence Retrieval
↓
Hybrid Reasoning Layer
(Gemini + GPT-5 + ML Model)
↓
Verdict + Explanation + Truth Score
↓
Streamlit Dashboard


---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit (Python) |
| **Backend Reasoning** | Google Gemini 1.5 Pro, OpenAI GPT-5 |
| **ML Model** | Scikit-learn, XGBoost |
| **Retrieval APIs** | Google Custom Search JSON API, NewsAPI, GNews |
| **Scripting / Integration** | Python 3.10+, Requests, dotenv |
| **Visualization** | Streamlit charts, Matplotlib |

---

## 📦 Project Structure




---

## 🔑 Environment Variables

Create a `.env` file in your project root with the following keys:

```bash
# Google APIs
GOOGLE_API_KEY=your_google_api_key_here
SEARCH_ENGINE_ID=your_cse_id_here

# News APIs (optional)
NEWS_API_KEY=your_newsapi_key_here
GNEWS_API_KEY=your_gnews_key_here

# LLM APIs
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_gpt5_key_here



---

## 🔑 Environment Variables

Create a `.env` file in your project root with the following keys:

```bash
# Google APIs
GOOGLE_API_KEY=your_google_api_key_here
SEARCH_ENGINE_ID=your_cse_id_here

# News APIs (optional)
NEWS_API_KEY=your_newsapi_key_here
GNEWS_API_KEY=your_gnews_key_here

# LLM APIs
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_gpt5_key_here


⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/ai-factcheck-dashboard.git
cd ai-factcheck-dashboard
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Configure environment
Create .env file with your API keys (see above).
4️⃣ Run the Streamlit dashboard
streamlit run dashboard.py
