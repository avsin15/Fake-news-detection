"""
dashboard.py
-------------
Interactive dashboard for Dual-LLM Hybrid Fact-Checker (GPT-5 + Gemini + XGBoost ML)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from ai_factcheck import hybrid_fact_check, MODEL_STATUS, ML_LOADED

st.set_page_config(
    page_title="AI Fact-Checker Dashboard", 
    page_icon="🧠", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .verdict-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #0e1117;
        color: white;
    }
    .verdict-true {
        border-left: 8px solid #28a745;
    }
    .verdict-false {
        border-left: 8px solid #dc3545;
    }
    .verdict-uncertain {
        border-left: 8px solid #ffc107;
    }
    .verdict-error {
        border-left: 8px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "check_count" not in st.session_state:
    st.session_state.check_count = 0
if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""
if "user_claim" not in st.session_state:
    st.session_state.user_claim = ""

# Header
st.markdown('<div class="main-header">🧠 AI Fact-Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by GPT-5 + Gemini 2.0 Flash + XGBoost ML | Adaptive Evidence-Based Verification</div>', 
    unsafe_allow_html=True
)

# --- Model Connectivity Banner ---
st.markdown("### 🔌 Model Status")
col1, col2, col3 = st.columns(3)

with col1:
    if MODEL_STATUS.get("gemini"):
        st.success("✅ **Gemini 2.0 Flash** Connected")
    else:
        st.error("❌ **Gemini** Not Connected")

with col2:
    if MODEL_STATUS.get("gpt5"):
        st.success("✅ **GPT-5** Connected")
    else:
        st.warning("⚠️ **GPT-5** Not Connected")

with col3:
    if ML_LOADED:
        st.success("✅ **XGBoost ML** Loaded")
    else:
        st.info("ℹ️ **ML Model** Not Available")

# Show warning if no models are available
if not MODEL_STATUS.get("gemini") and not MODEL_STATUS.get("gpt5"):
    st.error("🚨 **Critical**: No AI models are connected! Please check your API keys.")

st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.info("""
    **How it works:**
    
    1️⃣ **Recent events** (2025, breaking news) → GPT-5 priority
    
    2️⃣ **Historical facts** → Gemini priority
    
    3️⃣ **Consensus mode**: Both models analyze and cross-check
    
    4️⃣ **Evidence**: Pulled from Google Fact Check, News API, and web search
    
    5️⃣ **ML Analysis**: XGBoost model provides independent truth score
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Current Session")
    st.metric("Fact-Checks Performed", st.session_state.check_count)
    
    if st.button("Reset Counter"):
        st.session_state.check_count = 0
        st.rerun()

# --- Main Input Area ---
st.markdown("### 📝 Enter Your Claim")

# Example claims buttons BEFORE input widgets
with st.expander("💡 Try Example Claims"):
    examples = [
        "The Earth is flat",
        "COVID-19 vaccines contain microchips",
        "Climate change is causing more extreme weather events",
        "Water boils at 100 degrees Celsius at sea level",
        "The 2024 Olympics were held in Paris"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"📌 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.selected_example = example
                st.session_state.user_claim = example

# Use selected example as default value
default_text = st.session_state.get("selected_example", "")

# Single unified input
user_input = st.text_area(
    "Enter a claim or paste a URL:",
    value=default_text,
    placeholder="Example: 'Climate change caused record temperatures in 2024' or 'https://example.com/article'",
    height=120,
    key="user_input_field"
)

# Clear the selected example after using it
if st.session_state.get("selected_example"):
    st.session_state.selected_example = ""

st.markdown("---")

# --- Analysis Button ---
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    analyze_button = st.button("🔍 Analyze Claim", type="primary", use_container_width=True)

# --- Analysis Logic ---
if analyze_button:
    # Get the actual input value
    claim_text = user_input.strip() if user_input else ""
    
    if not claim_text:
        st.warning("⚠️ Please enter a claim or URL to analyze.")
    else:
        st.session_state.check_count += 1
        
        # Show what we're analyzing
        st.info(f"🔍 Analyzing: {claim_text[:100]}{'...' if len(claim_text) > 100 else ''}")
        
        with st.spinner("🔄 Analyzing claim and gathering evidence..."):
            try:
                result = hybrid_fact_check(claim_text)
                
                # --- Display Results ---
                st.markdown("## 📊 Analysis Results")
                
                # Summary Box
                st.info(f"**Routing Strategy**: {result.summary}")
                
                # Verdict Display
                verdict_lower = result.gemini.verdict.lower()
                if verdict_lower == "true":
                    verdict_class = "verdict-true"
                    verdict_icon = "✅"
                    verdict_color = "#28a745"
                elif verdict_lower == "false":
                    verdict_class = "verdict-false"
                    verdict_icon = "❌"
                    verdict_color = "#dc3545"
                elif verdict_lower == "uncertain":
                    verdict_class = "verdict-uncertain"
                    verdict_icon = "⚠️"
                    verdict_color = "#ffc107"
                else:
                    verdict_class = "verdict-error"
                    verdict_icon = "🚫"
                    verdict_color = "#dc3545"
                
                st.markdown(
                    f'<div class="verdict-box {verdict_class}">'
                    f'<h2>{verdict_icon} Verdict: {result.gemini.verdict}</h2>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # LLM Truth Score
                col1.metric(
                    "LLM Truth Score", 
                    f"{result.gemini.truth_score}%" if result.gemini.truth_score is not None else "N/A",
                    help="Truth score from AI language models"
                )
                
                # ML Truth Score (if available)
                if result.ml_score is not None:
                    ml_percentage = int(result.ml_score * 100)
                    col2.metric(
                        "🤖 ML Truth Score", 
                        f"{ml_percentage}%",
                        help="Independent ML model prediction (XGBoost)"
                    )
                else:
                    col2.metric("🤖 ML Truth Score", "N/A", help="ML model not available")
                
                col3.metric("Evidence Sources", len(result.fact_sources))
                col4.metric(
                    "Confidence", 
                    "High" if (result.gemini.truth_score or 50) > 75 else 
                    "Medium" if (result.gemini.truth_score or 50) > 40 else "Low"
                )
                
                # Explanation
                st.markdown("### 💬 Detailed Explanation")
                st.markdown(result.gemini.explanation)
                
                # ML Score Analysis (if available)
                if result.ml_score is not None:
                    st.markdown("---")
                    st.markdown("### 🤖 ML Model Analysis")
                    
                    ml_percentage = int(result.ml_score * 100)
                    
                    # Interpretation
                    if ml_percentage >= 75:
                        ml_verdict = "Likely True"
                        ml_color = "green"
                        ml_icon = "✅"
                    elif ml_percentage >= 50:
                        ml_verdict = "Uncertain/Mixed"
                        ml_color = "orange"
                        ml_icon = "⚠️"
                    else:
                        ml_verdict = "Likely False"
                        ml_color = "red"
                        ml_icon = "❌"
                    
                    col_ml1, col_ml2 = st.columns([1, 2])
                    with col_ml1:
                        st.metric("ML Verdict", f"{ml_icon} {ml_verdict}")
                    with col_ml2:
                        st.metric("ML Confidence", f"{ml_percentage}%")
                    
                    st.info(f"""
                    **Note:** The ML model (XGBoost trained on {'>50k'} claims) provides an independent 
                    statistical analysis based on linguistic patterns. This is a supplementary metric 
                    and should be considered alongside the LLM analysis and evidence.
                    """)
                    
                    # Comparison with LLM
                    if result.gemini.truth_score is not None:
                        llm_score = result.gemini.truth_score
                        score_diff = abs(llm_score - ml_percentage)
                        
                        if score_diff < 20:
                            st.success(f"✅ **Agreement**: LLM ({llm_score}%) and ML ({ml_percentage}%) scores are closely aligned (difference: {score_diff}%)")
                        else:
                            st.warning(f"⚠️ **Divergence**: LLM ({llm_score}%) and ML ({ml_percentage}%) scores differ significantly (difference: {score_diff}%). Consider the evidence carefully.")
                
                # Evidence Sources
                st.markdown("---")
                st.markdown("### 📚 Evidence Sources")
                
                if result.fact_sources:
                    has_real_sources = any(e.source != "system" for e in result.fact_sources)
                    
                    if not has_real_sources:
                        st.warning("⚠️ No external evidence found. Verdict based on AI knowledge.")
                    
                    for i, ev in enumerate(result.fact_sources, 1):
                        with st.expander(f"Source {i}: {ev.title}", expanded=(i <= 2)):
                            st.markdown(f"**Source Type**: {ev.source.upper()}")
                            if ev.snippet:
                                st.markdown(f"**Summary**: {ev.snippet}")
                            if ev.url:
                                st.markdown(f"**Link**: [{ev.url}]({ev.url})")
                            else:
                                st.caption("_No URL available_")
                else:
                    st.warning("No evidence sources available.")
                
                # Visualization
                if result.gemini.truth_score is not None or result.ml_score is not None:
                    st.markdown("---")
                    st.markdown("### 🎯 Truth Score Comparison")
                    
                    # Create dual gauge if both scores available
                    if result.gemini.truth_score is not None and result.ml_score is not None:
                        col_g1, col_g2 = st.columns(2)
                        
                        with col_g1:
                            st.markdown("#### 🧠 LLM Truth Score")
                            fig1 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=result.gemini.truth_score,
                                title={'text': "LLM Score", 'font': {'size': 20}},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 2},
                                    'bar': {'color': verdict_color},
                                    'steps': [
                                        {'range': [0, 40], 'color': '#ffebee'},
                                        {'range': [40, 75], 'color': '#fff9e6'},
                                        {'range': [75, 100], 'color': '#e8f5e9'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': result.gemini.truth_score
                                    }
                                }
                            ))
                            fig1.update_layout(height=250)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col_g2:
                            st.markdown("#### 🤖 ML Truth Score")
                            ml_percentage = int(result.ml_score * 100)
                            ml_color_gauge = "#28a745" if ml_percentage > 75 else "#ffc107" if ml_percentage > 40 else "#dc3545"
                            
                            fig2 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=ml_percentage,
                                title={'text': "ML Score", 'font': {'size': 20}},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 2},
                                    'bar': {'color': ml_color_gauge},
                                    'steps': [
                                        {'range': [0, 40], 'color': '#ffebee'},
                                        {'range': [40, 75], 'color': '#fff9e6'},
                                        {'range': [75, 100], 'color': '#e8f5e9'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': ml_percentage
                                    }
                                }
                            ))
                            fig2.update_layout(height=250)
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    # Single gauge if only one score available
                    elif result.gemini.truth_score is not None:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result.gemini.truth_score,
                            title={'text': "LLM Truth Score (0-100)", 'font': {'size': 24}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 2},
                                'bar': {'color': verdict_color},
                                'steps': [
                                    {'range': [0, 40], 'color': '#ffebee'},
                                    {'range': [40, 75], 'color': '#fff9e6'},
                                    {'range': [75, 100], 'color': '#e8f5e9'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': result.gemini.truth_score
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download Results
                st.markdown("---")
                result_json = {
                    "claim": claim_text,
                    "verdict": result.gemini.verdict,
                    "llm_truth_score": result.gemini.truth_score,
                    "ml_truth_score": int(result.ml_score * 100) if result.ml_score else None,
                    "explanation": result.gemini.explanation,
                    "evidence_count": len(result.fact_sources),
                    "summary": result.summary,
                    "timestamp": str(pd.Timestamp.now())
                }
                
                import json
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(result_json, indent=2),
                    file_name="fact_check_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"🚨 **Error during analysis**: {str(e)}")
                with st.expander("Show Full Error Details"):
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
    Built with ❤️ using Streamlit | Powered by GPT-5, Gemini & XGBoost<br>
    <small>⚠️ This tool is for educational purposes. Always verify critical information from multiple sources.</small>
    </div>
    """,
    unsafe_allow_html=True
)