"""
ai_factcheck.py
---------------
Dual-LLM Hybrid Fact-Checker:
✅ GPT-5 for recent and real-time events
✅ Gemini 2.5 Flash for older or general reasoning
✅ Consensus logic between models
✅ Fallback reasoning and evidence aggregation
"""

import os, re, json, logging, numpy as np
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

# Import from evidence_pipeline (no duplicate definitions)
from evidence_pipeline import (
    gather_evidence,
    extract_article,
    build_evidence_context,
    EvidenceItem,
)

# ML Model imports
try:
    import joblib
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    log.warning("⚠️ ML dependencies not available (joblib, sentence-transformers)")

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hybrid_factcheck")

# --- Load environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug: Check environment
log.info("🔍 Environment check:")
log.info(f"   OPENAI_API_KEY: {'✅ Set' if OPENAI_API_KEY else '❌ Missing'}")
log.info(f"   GEMINI_API_KEY: {'✅ Set' if GEMINI_API_KEY else '❌ Missing'}")

# --- Optional GPT/OpenAI import ---
client = None
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("✅ OpenAI client initialized")
except ImportError:
    log.warning("⚠️ OpenAI library not installed (pip install openai)")
except Exception as e:
    log.warning(f"⚠️ OpenAI client initialization failed: {e}")

# --- Gemini setup ---
genai = None
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        log.info("✅ Gemini configured")
    else:
        log.error("❌ GEMINI_API_KEY not set!")
        genai = None
except ImportError:
    log.error("❌ google-generativeai not installed (pip install google-generativeai)")
    genai = None
except Exception as e:
    log.error(f"❌ Gemini configuration failed: {e}")
    genai = None


# ------------------------------------------------------------
# ✅ MODEL CONNECTIVITY CHECK
# ------------------------------------------------------------
def check_model_connectivity():
    """Test if models are actually reachable"""
    status = {"gemini": False, "gpt5": False}
    
    # --- Gemini test ---
    if genai and GEMINI_MODEL_NAME:
        try:
            test_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            resp = test_model.generate_content("ping", 
                generation_config={"max_output_tokens": 10})
            if resp and hasattr(resp, "text"):
                status["gemini"] = True
                log.info("✅ Gemini connection verified")
            elif hasattr(resp, "candidates") and resp.candidates:
                status["gemini"] = True
                log.info("✅ Gemini connection verified")
        except Exception as e:
            log.warning(f"⚠️ Gemini connection failed: {e}")
    else:
        log.warning("⚠️ Gemini not configured")

    # --- GPT-5 test ---
    if client:
        try:
            log.info("🔍 Testing GPT-5 connection...")
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=10
            )
            # Check if we got a valid response
            if resp and resp.choices and len(resp.choices) > 0:
                content = resp.choices[0].message.content
                log.info(f"✅ GPT-5 connection verified (response: {content[:20] if content else 'empty'})")
                status["gpt5"] = True
            else:
                log.warning("⚠️ GPT-5 returned empty response")
        except Exception as e:
            log.warning(f"⚠️ GPT-5 connection failed: {e}")
            log.warning(f"   Error details: {type(e).__name__}: {str(e)}")
    else:
        log.warning("⚠️ OpenAI client not initialized")

    return status

# Run connectivity check once on startup
MODEL_STATUS = check_model_connectivity()


# ------------------------------------------------------------
# ✅ ML MODEL LOADING (XGBoost)
# ------------------------------------------------------------
ML_MODEL = None
ML_ENCODER = None
ML_MODEL_NAME = "all-MiniLM-L6-v2"

def load_ml_model():
    """Load the trained XGBoost model and SentenceTransformer encoder"""
    global ML_MODEL, ML_ENCODER
    
    if not ML_AVAILABLE:
        log.warning("⚠️ ML dependencies not available")
        return False
    
    model_path = "models/xgboost_model.pkl"
    
    try:
        # Load XGBoost model
        if os.path.exists(model_path):
            ML_MODEL = joblib.load(model_path)
            log.info("✅ XGBoost model loaded successfully")
        else:
            log.warning(f"⚠️ ML model not found at {model_path}")
            return False
        
        # Load sentence encoder
        ML_ENCODER = SentenceTransformer(ML_MODEL_NAME)
        log.info(f"✅ Sentence encoder loaded: {ML_MODEL_NAME}")
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to load ML model: {e}")
        return False

# Try to load ML model on startup
ML_LOADED = load_ml_model()

# Make ML_LOADED available for import
__all__ = ['hybrid_fact_check', 'MODEL_STATUS', 'ML_LOADED']


def get_ml_prediction(text: str) -> Optional[float]:
    """
    Get ML model prediction (truth probability).
    Returns probability score between 0-1 (higher = more likely real/true).
    """
    if not ML_LOADED or ML_MODEL is None or ML_ENCODER is None:
        return None
    
    try:
        # Generate embedding
        embedding = ML_ENCODER.encode([text], convert_to_numpy=True)
        
        # Get prediction probabilities
        # Assuming model outputs: [prob_real, prob_fake, prob_partially_true]
        proba = ML_MODEL.predict_proba(embedding)[0]
        
        # Calculate truth score
        # Real = 1.0, Partially True = 0.5, Fake = 0.0
        if len(proba) == 3:
            truth_score = proba[0] * 1.0 + proba[2] * 0.5 + proba[1] * 0.0
        elif len(proba) == 2:
            # Binary: [real, fake]
            truth_score = proba[0]
        else:
            truth_score = proba[0]
        
        log.info(f"🤖 ML truth score: {truth_score:.3f}")
        return float(truth_score)
        
    except Exception as e:
        log.error(f"❌ ML prediction failed: {e}")
        return None


# --- Data structures ---
@dataclass
class GeminiResult:
    verdict: str
    explanation: str
    truth_score: Optional[int] = None

@dataclass
class HybridResult:
    gemini: GeminiResult
    fact_sources: List[EvidenceItem]
    ml_score: Optional[float]
    summary: str


# ------------------------------------------------------------
# Helper: determine if claim references recent (post-2024) data
# ------------------------------------------------------------
def is_recent_claim(text: str, cutoff_year: int = 2024) -> bool:
    """
    Detects whether a claim likely refers to a recent or ongoing event.
    Used to route reasoning to GPT-5 instead of Gemini.
    """
    if not text:
        return False

    t = text.lower()

    # 1️⃣ Year detection
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if years and max(years) > cutoff_year:
        return True

    # 2️⃣ Recency keywords
    recent_terms = [
        "today", "yesterday", "this week", "last week",
        "this month", "breaking", "just in", "recently",
        "new report", "this year", "latest", "2025", "now",
        "earlier this", "as of", "update", "unfolding", "current"
    ]

    return any(term in t for term in recent_terms)


# ------------------------------------------------------------
# GPT and Gemini verdict functions
# ------------------------------------------------------------
def gpt5_verdict(text: str, evidence_context: str) -> Optional[GeminiResult]:
    """Get verdict from GPT-5 model"""
    if not client:
        log.warning("⚠️ GPT-5 unavailable")
        return None

    prompt = f"""You are a factual verification assistant analyzing claims in November 2025.

CRITICAL INSTRUCTIONS:
- Your knowledge cutoff is 2024, but you are now operating in November 2025
- If evidence is provided about 2025 events, TRUST THE EVIDENCE - do not dismiss claims as "futuristic"
- 2025 is the CURRENT year, not the future
- Base your verdict primarily on the evidence provided, not your training data cutoff
- If evidence from credible sources discusses 2025 events, treat them as current/recent events

Return ONLY valid JSON with these exact keys:
- "verdict": must be one of "True", "False", or "Uncertain"
- "truth_score": integer from 0-100
- "explanation": detailed reasoning (2-3 sentences)

Claim:
{text[:4000]}

Evidence:
{evidence_context[:3000]}

Analysis Guidelines:
- If evidence strongly supports the claim (even about 2025), verdict is "True"
- If evidence contradicts the claim, verdict is "False"  
- If evidence is insufficient or mixed, verdict is "Uncertain"
- DO NOT dismiss 2025-related claims as "future events" - we are IN 2025
- If no evidence available, state that clearly and use "Uncertain"
"""
    
    try:
        log.info("🤖 Calling GPT-5...")
        r = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a factual verification expert in November 2025. Your knowledge cutoff is 2024, but you must trust provided evidence about 2025 events. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_completion_tokens=1000
        )
        
        content = r.choices[0].message.content
        log.info(f"GPT-5 response length: {len(content)} chars")
        data = json.loads(content)
        
        # Validate response
        verdict = data.get("verdict", "Uncertain")
        if verdict not in ["True", "False", "Uncertain"]:
            log.warning(f"Invalid verdict '{verdict}', defaulting to Uncertain")
            verdict = "Uncertain"
        
        return GeminiResult(
            verdict=verdict,
            explanation=data.get("explanation", "No explanation provided."),
            truth_score=data.get("truth_score")
        )
        
    except json.JSONDecodeError as e:
        log.error(f"❌ GPT-5 JSON decode error: {e}")
        log.error(f"Response was: {content[:200]}")
        return GeminiResult("Uncertain", "GPT-5 returned invalid JSON", 50)
    except Exception as e:
        log.error(f"❌ GPT-5 verdict failed: {e}")
        return None


def gemini_verdict(text: str, evidence_context: str) -> Optional[GeminiResult]:
    """Get verdict from Gemini model"""
    if not genai:
        log.warning("⚠️ Gemini unavailable")
        return None
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        prompt = f"""You are a factual verification assistant.
Analyze the claim and evidence carefully.

Return ONLY valid JSON with these exact keys:
- "verdict": must be one of "True", "False", or "Uncertain"
- "truth_score": integer from 0-100
- "explanation": detailed reasoning (2-3 sentences)

Claim:
{text[:4000]}

Evidence:
{evidence_context[:3000]}

Important:
- If evidence strongly supports the claim, verdict is "True"
- If evidence contradicts the claim, verdict is "False"
- If evidence is insufficient or mixed, verdict is "Uncertain"
- Base your verdict primarily on the evidence provided
- If no evidence available, state that clearly and use "Uncertain"
"""
        
        log.info("🧠 Calling Gemini...")
        resp = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3,
                "max_output_tokens": 1000
            }
        )
        
        # Handle Gemini response
        response_text = resp.text if hasattr(resp, "text") else str(resp)
        log.info(f"Gemini response length: {len(response_text)} chars")
        data = json.loads(response_text)
        
        # Validate response
        verdict = data.get("verdict", "Uncertain")
        if verdict not in ["True", "False", "Uncertain"]:
            log.warning(f"Invalid verdict '{verdict}', defaulting to Uncertain")
            verdict = "Uncertain"
        
        return GeminiResult(
            verdict=verdict,
            explanation=data.get("explanation", "No explanation provided."),
            truth_score=data.get("truth_score")
        )
        
    except json.JSONDecodeError as e:
        log.error(f"❌ Gemini JSON decode error: {e}")
        log.error(f"Response was: {response_text[:200]}")
        return GeminiResult("Uncertain", "Gemini returned invalid JSON", 50)
    except Exception as e:
        log.error(f"❌ Gemini verdict failed: {e}")
        return None


# ------------------------------------------------------------
# Fallback reasoning
# ------------------------------------------------------------
def fallback_reasoning(text: str) -> GeminiResult:
    """Fallback when no evidence is found and models fail"""
    log.info("🔄 Using fallback reasoning...")
    
    if not genai:
        return GeminiResult(
            "Uncertain", 
            "Unable to verify claim: No evidence found and verification models unavailable.",
            50
        )
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        prompt = f"""The following claim could not be verified due to lack of evidence.
Provide a brief explanation of why this claim is difficult to verify.

Claim:
{text[:600]}

Return JSON with keys: "verdict" (use "Uncertain"), "explanation", "truth_score" (use 50).
"""
        resp = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.5
            }
        )
        data = json.loads(resp.text if hasattr(resp, "text") else str(resp))
        return GeminiResult(
            data.get("verdict", "Uncertain"),
            data.get("explanation", "No information available to verify this claim."),
            data.get("truth_score", 50)
        )
    except Exception as e:
        log.error(f"❌ Fallback reasoning failed: {e}")
        return GeminiResult(
            "Uncertain", 
            "Unable to verify claim: No evidence found and verification process encountered errors.",
            50
        )


# ------------------------------------------------------------
# Dual-LLM hybrid controller
# ------------------------------------------------------------
def hybrid_fact_check(user_input: str) -> HybridResult:
    """
    Main fact-checking function.
    Routes to appropriate LLM based on claim recency and availability.
    """
    log.info("="*60)
    log.info("🚀 Starting fact-check process")
    log.info("="*60)
    
    text = user_input.strip()
    
    # Handle URL input
    url_mode = bool(re.match(r"^https?://", text))
    if url_mode:
        log.info(f"📄 URL detected: {text[:80]}")
        extracted = extract_article(text)
        if extracted and len(extracted) > 100:
            text = extracted
            log.info(f"✅ Extracted {len(text)} characters from article")
        else:
            log.warning("⚠️ Article extraction failed or content too short")
            text = user_input  # Fall back to URL as text

    # --- Evidence retrieval ---
    log.info("🔍 Gathering evidence...")
    evidence = gather_evidence(text)
    evidence_context = build_evidence_context(evidence)
    
    log.info(f"📚 Evidence context: {len(evidence_context)} chars")
    log.info(f"📊 Evidence items collected: {len(evidence)}")
    
    # Check if we have real evidence
    has_real_evidence = any(e.source != "system" for e in evidence)
    log.info(f"🎯 Real evidence found: {has_real_evidence}")

    # --- Get ML prediction (independent analysis) ---
    ml_score = get_ml_prediction(text)
    if ml_score is not None:
        log.info(f"🤖 ML Model Truth Score: {ml_score*100:.1f}%")
    else:
        log.info("⚠️ ML prediction not available")

    # --- Decide routing strategy ---
    recent_mode = is_recent_claim(text)
    log.info(f"🕐 Recent claim detected: {recent_mode}")
    
    gpt_res = None
    gem_res = None

    # Route based on recency and availability
    if recent_mode and MODEL_STATUS.get("gpt5"):
        log.info("📍 ROUTING: GPT-5 (recent event)")
        gpt_res = gpt5_verdict(text, evidence_context)
        # Also get Gemini for consensus if available
        if MODEL_STATUS.get("gemini"):
            gem_res = gemini_verdict(text, evidence_context)
    else:
        log.info("📍 ROUTING: Gemini first (historical/general)")
        if MODEL_STATUS.get("gemini"):
            gem_res = gemini_verdict(text, evidence_context)
        # Get GPT-5 for consensus if available
        if MODEL_STATUS.get("gpt5"):
            gpt_res = gpt5_verdict(text, evidence_context)

    # --- Consensus logic ---
    if gem_res and gpt_res:
        log.info(f"🤝 Both models responded: Gemini={gem_res.verdict}, GPT-5={gpt_res.verdict}")
        
        if gem_res.verdict == gpt_res.verdict:
            # Agreement - use higher scoring one
            final = gem_res if (gem_res.truth_score or 0) >= (gpt_res.truth_score or 0) else gpt_res
            final.explanation += f"\n\n✅ **Consensus**: Both models agree on '{final.verdict}'."
            log.info(f"✅ Models agree: {final.verdict}")
        else:
            # Disagreement - mark as uncertain with both perspectives
            explanation = (
                f"⚖️ **Models Disagree**:\n\n"
                f"**Gemini verdict**: {gem_res.verdict} (Score: {gem_res.truth_score})\n"
                f"{gem_res.explanation}\n\n"
                f"**GPT-5 verdict**: {gpt_res.verdict} (Score: {gpt_res.truth_score})\n"
                f"{gpt_res.explanation}\n\n"
                f"Due to this disagreement, we mark the claim as 'Uncertain' and recommend further investigation."
            )
            score = int(np.mean([(gem_res.truth_score or 50), (gpt_res.truth_score or 50)]))
            final = GeminiResult("Uncertain", explanation, score)
            log.warning(f"⚖️ Models disagree: Gemini={gem_res.verdict}, GPT-5={gpt_res.verdict}")
    
    elif gem_res or gpt_res:
        # Only one model responded
        final = gpt_res or gem_res
        model_name = "GPT-5" if gpt_res else "Gemini"
        log.info(f"✅ Single model verdict: {model_name} = {final.verdict}")
    
    else:
        # No models available - critical failure
        log.error("❌ No models available for verdict!")
        final = GeminiResult(
            "Error",
            "Unable to process claim: No AI models are currently available.",
            0
        )

    # --- Fallback reasoning for uncertain cases with no evidence ---
    if (final.verdict.lower() == "uncertain" and not has_real_evidence and 
        "error" not in final.explanation.lower()):
        log.info("🔄 Triggering fallback reasoning (uncertain + no evidence)...")
        fallback = fallback_reasoning(text)
        # Append fallback explanation
        final.explanation += f"\n\n**Additional Context**: {fallback.explanation}"

    # --- Build summary ---
    summary_parts = [
        f"Verdict: {final.verdict}",
        f"Score: {final.truth_score if final.truth_score is not None else 'N/A'}",
        f"Evidence: {len(evidence)} items"
    ]
    
    if gpt_res and gem_res:
        summary_parts.append("Mode: Dual-LLM (GPT-5 + Gemini)")
    elif gpt_res:
        summary_parts.append("Mode: GPT-5 Only")
    elif gem_res:
        summary_parts.append("Mode: Gemini Only")
    else:
        summary_parts.append("Mode: Error - No Models")
    
    summary = " | ".join(summary_parts)

    # --- Ensure evidence is never completely empty ---
    if not evidence or all(e.source == "system" for e in evidence):
        evidence = [EvidenceItem(
            title="No external evidence found",
            snippet="No relevant fact-checks or sources were located. The verdict is based on AI analysis using available knowledge.",
            url="",
            source="system"
        )]

    log.info("="*60)
    log.info(f"✅ Fact-check complete: {final.verdict}")
    log.info("="*60)

    return HybridResult(final, evidence, ml_score, summary)