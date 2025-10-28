"""
dashboard.py
-------------
Interactive dashboard for Dual-LLM Hybrid Fact-Checker (GPT-5 + Gemini)
"""

import streamlit as st
import plotly.graph_objects as go
from ai_factcheck import hybrid_fact_check
from ai_factcheck import MODEL_STATUS


st.set_page_config(page_title="AI Fact-Checker Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 Dual-LLM Fact-Checker")
st.caption("Powered by GPT-5 + Gemini 2.5 Flash | Adaptive factual reasoning")

# --- Model Connectivity Banner ---
colA, colB = st.columns(2)
with colA:
    if MODEL_STATUS.get("gemini"):
        st.success("✅ Gemini Connected")
    else:
        st.error("❌ Gemini Not Connected")

with colB:
    if MODEL_STATUS.get("gpt5"):
        st.success("✅ GPT-5 Connected")
    else:
        st.warning("⚠️ GPT-5 Not Connected (Gemini fallback active)")


st.sidebar.header("⚙️ Model Settings")
st.sidebar.info("In Auto mode, GPT-5 handles recent events and Gemini handles historical facts.")

user_input = st.text_area("Enter a claim or article URL:", placeholder="Example: 'Global warming intensified in 2025 reports.'", height=120)

if st.button("🔍 Analyze Claim", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a claim to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = hybrid_fact_check(user_input)

        st.markdown("## 🧩 Summary")
        st.markdown(f"**Routing:** {result.summary}")

        st.markdown("### 🧠 Final Verdict")
        col1, col2 = st.columns(2)
        col1.metric("Verdict", result.gemini.verdict)
        col2.metric("Truth Score", result.gemini.truth_score or "N/A")

        with st.expander("💬 Explanation"):
            st.write(result.gemini.explanation)

        if result.ml_score is not None:
            st.markdown("---")
            st.markdown("### 🤖 ML Model Confidence (Secondary)")
            st.write(f"Estimated truth likelihood: **{result.ml_score * 100:.1f}%**")

        st.markdown("---")
        st.markdown("### 📚 Evidence Sources")
        if result.fact_sources:
            for ev in result.fact_sources:
                st.markdown(f"**{ev.title or 'Untitled'}**  \n{ev.snippet or ''}  \n🔗 [{ev.source}]({ev.url})")
        else:
            st.warning("No evidence sources found.")

        if result.gemini.truth_score is not None:
            st.markdown("---")
            st.markdown("### 🎯 Confidence Visualization")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.gemini.truth_score,
                title={'text': "Truth Score"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
