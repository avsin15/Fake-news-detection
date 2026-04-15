"""
ai_factcheck.py
---------------
Dual-LLM Hybrid Fact-Checker:
GPT-5 for recent and real-time events
Gemini 2.5 Flash for older or general reasoning
Consensus logic between models
Fallback reasoning and evidence aggregation
"""

"""
ai_factcheck.py
---------------
Dual-LLM Hybrid Fact-Checker
GPT-5 + Gemini + ML model

Cleaned stable version for capstone demo
"""

"""
ai_factcheck.py
---------------
Dual-LLM Hybrid Fact-Checker

GPT-5 + Gemini + XGBoost ML
Evidence-based claim verification

Final stabilized version for capstone demo
"""

import os
import re
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

from openai import OpenAI

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from evidence_pipeline import (
    gather_evidence,
    extract_article,
    build_evidence_context,
    EvidenceItem,
)

# ML imports
try:
    import joblib
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def parse_truth_score(value):
    """
    Safely convert truth score to integer 0-100
    """
    try:
        value = float(value)

        # handle 0-1 scale
        if value <= 1:
            value = value * 100

        value = int(value)

        if value > 100:
            value = 100
        if value < 0:
            value = 0

        return value

    except Exception:
        return 50


def normalize_verdict(verdict):
    """
    Convert any LLM verdict to standard format:
    True / False / Uncertain
    """

    if isinstance(verdict, bool):
        return "True" if verdict else "False"

    verdict = str(verdict).strip().lower()

    true_terms = [
        "true", "correct", "supports", "supported",
        "accurate", "confirmed", "yes"
    ]

    false_terms = [
        "false", "incorrect", "refutes", "refuted",
        "wrong", "misleading", "no"
    ]

    uncertain_terms = [
        "uncertain", "unknown", "mixed",
        "inconclusive", "partially true"
    ]

    if verdict in true_terms:
        return "True"

    if verdict in false_terms:
        return "False"

    if verdict in uncertain_terms:
        return "Uncertain"

    return "Uncertain"
# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hybrid_factcheck")


# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.5-flash"

log.info("Environment check:")
log.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Missing'}")
log.info(f"GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Missing'}")


# ------------------------------------------------------------
# MODEL STATUS
# ------------------------------------------------------------

MODEL_STATUS = {
    "gemini": False,
    "gpt5": False
}


# ------------------------------------------------------------
# OPENAI INITIALIZATION
# ------------------------------------------------------------

openai_client = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        MODEL_STATUS["gpt5"] = True
        log.info("OpenAI initialized")
    except Exception as e:
        log.warning(f"OpenAI init failed: {e}")


# ------------------------------------------------------------
# GEMINI INITIALIZATION
# ------------------------------------------------------------

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        MODEL_STATUS["gemini"] = True
        log.info("Gemini initialized")
    except Exception as e:
        log.warning(f"Gemini init failed: {e}")


# ------------------------------------------------------------
# CONNECTIVITY TEST
# ------------------------------------------------------------

def check_model_connectivity():

    status = {"gemini": False, "gpt5": False}

    if genai and GEMINI_API_KEY:
        try:
            # Use REST-based list_models instead of gRPC generate_content ping
            # This avoids DNS/gRPC failures on cloud servers
            models = list(genai.list_models())
            if models:
                status["gemini"] = True
                log.info("Gemini connection verified via REST")

        except Exception as e:
            log.warning(f"Gemini connection failed: {e}")

    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5
            )

            if resp:
                status["gpt5"] = True
                log.info("GPT-5 connection verified")

        except Exception as e:
            log.warning(f"GPT connection failed: {e}")

    return status


MODEL_STATUS = check_model_connectivity()


# ------------------------------------------------------------
# ML MODEL LOADING
# ------------------------------------------------------------

ML_MODEL = None
ML_ENCODER = None
ML_LOADED = False


# Resolve model path relative to this script file, not the working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "xgboost_model.pkl")


def load_ml_model():

    global ML_MODEL, ML_ENCODER, ML_LOADED

    if not ML_AVAILABLE:
        return False

    try:

        if os.path.exists(_MODEL_PATH):
            ML_MODEL = joblib.load(_MODEL_PATH)
            ML_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")

            ML_LOADED = True
            log.info("ML model loaded")

    except Exception as e:

        log.warning(f"ML load failed: {e}")

    return ML_LOADED


ML_LOADED = load_ml_model()


# ------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------

@dataclass
class GeminiResult:

    verdict: str
    explanation: str
    truth_score: Optional[int]


@dataclass
class HybridResult:

    gemini: GeminiResult
    fact_sources: List[EvidenceItem]
    ml_score: Optional[float]
    summary: str


# ------------------------------------------------------------
# SAFE TRUTH SCORE PARSER
# ------------------------------------------------------------

def parse_truth_score(value):

    try:

        value = float(value)

        if value <= 1:
            value = value * 100

        value = int(value)

        if value > 100:
            value = 100

        if value < 0:
            value = 0

        return value

    except Exception:

        return 50


# ------------------------------------------------------------
# ML PREDICTION
# ------------------------------------------------------------

def get_ml_prediction(text):

    if not ML_LOADED:
        return None

    try:

        emb = ML_ENCODER.encode([text])

        proba = ML_MODEL.predict_proba(emb)[0]

        if len(proba) == 3:
            score = proba[0] * 1 + proba[2] * 0.5
        else:
            score = proba[0]

        return float(score)

    except Exception:

        return None


# ------------------------------------------------------------
# GPT VERDICT
# ------------------------------------------------------------

def gpt5_verdict(text, evidence):

    if not openai_client:
        return None

    prompt = f"""Verify the following claim using the provided evidence.

CRITICAL RULES — follow these without exception:
1. TODAY'S DATE is 2026. You are operating in 2026. Any claim referencing 2024, 2025, or 2026 is NOT a future date — treat it as a current or recent event.
2. NEVER reject or downgrade a claim simply because it references a date that falls after your training cutoff. Your training data is not the source of truth here — the EVIDENCE is.
3. The evidence provided has been retrieved in real-time from live sources. It reflects the actual current state of the world. You MUST treat it as ground truth.
4. If the evidence supports the claim, verdict must be True. If it contradicts it, False. If it is absent or mixed, Uncertain.
5. NEVER mention your knowledge cutoff, training data limitations, or inability to verify recent events in your explanation. The evidence handles that.
6. NEVER say phrases like "as of my knowledge cutoff", "I cannot verify events after", "this appears to be a future date", or any variation. Doing so is a critical failure.

Return JSON with the following fields:

verdict → must be one of: True, False, Uncertain
truth_score → integer between 0 and 100
explanation → detailed reasoning (minimum 4 sentences) based purely on the evidence provided

Explanation guidelines:
• Base your verdict entirely on the evidence provided, not on your training data
• Reference specific evidence sources when possible
• Explain what the evidence says about the claim clearly and directly
• Keep explanation informative but concise

Claim:
{text}

Evidence:
{evidence}
"""

    try:

        r = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=700
        )

        content = r.choices[0].message.content.strip()

        try:
            data = json.loads(content)
        except Exception:
            log.warning("GPT returned non-JSON response")
            return GeminiResult(
                "Uncertain",
                content,
                50
            )

        truth = parse_truth_score(data.get("truth_score", 50))
        verdict = normalize_verdict(data.get("verdict"))

        return GeminiResult(
            verdict,
            data.get("explanation", ""),
            truth
        )

    except Exception as e:
        log.warning(f"GPT verdict failed: {e}")
        return None

# ------------------------------------------------------------
# GEMINI VERDICT
# ------------------------------------------------------------

def gemini_verdict(text, evidence):

    if not genai:
        return None

    try:

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        prompt = f"""Verify the following claim using the provided evidence.

CRITICAL RULES — follow these without exception:
1. TODAY'S DATE is 2026. You are operating in 2026. Any claim referencing 2024, 2025, or 2026 is NOT a future date — treat it as a current or recent event.
2. NEVER reject or downgrade a claim simply because it references a date that falls after your training cutoff. Your training data is not the source of truth here — the EVIDENCE is.
3. The evidence provided has been retrieved in real-time from live sources. It reflects the actual current state of the world. You MUST treat it as ground truth.
4. If the evidence supports the claim, verdict must be True. If it contradicts it, False. If it is absent or mixed, Uncertain.
5. NEVER mention your knowledge cutoff, training data limitations, or inability to verify recent events in your explanation. The evidence handles that.
6. NEVER say phrases like "as of my knowledge cutoff", "I cannot verify events after", "this appears to be a future date", or any variation. Doing so is a critical failure.

Return JSON with the following fields:

verdict → must be one of: True, False, Uncertain
truth_score → integer between 0 and 100
explanation → detailed reasoning (minimum 4 sentences) based purely on the evidence provided

Explanation guidelines:
• Base your verdict entirely on the evidence provided, not on your training data
• Reference specific evidence sources when possible
• Explain what the evidence says about the claim clearly and directly
• Keep explanation informative but concise

Claim:
{text}

Evidence:
{evidence}
"""

        resp = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json"
            }
        )

        content = resp.text.strip()

        try:
            data = json.loads(content)
        except Exception:
            log.warning("Gemini returned non-JSON response")
            return GeminiResult(
                "Uncertain",
                content,
                50
            )

        truth = parse_truth_score(data.get("truth_score", 50))
        verdict = normalize_verdict(data.get("verdict"))

        return GeminiResult(
            verdict,
            data.get("explanation", ""),
            truth
        )

    except Exception as e:
        log.warning(f"Gemini verdict failed: {e}")
        return None

# ------------------------------------------------------------
# MAIN FACT CHECK
# ------------------------------------------------------------

def hybrid_fact_check(user_input):

    text = user_input.strip()

    # URL extraction
    if text.startswith("http"):

        extracted = extract_article(text)

        if extracted:
            text = extracted

    evidence = gather_evidence(text)

    evidence_context = build_evidence_context(evidence)

    gem_res = gemini_verdict(text, evidence_context)

    gpt_res = gpt5_verdict(text, evidence_context)

    final = gem_res or gpt_res

    if not final:

        final = GeminiResult(
            "Uncertain",
            "No model response available.",
            50
        )

    ml_score = get_ml_prediction(text)

    summary = f"Verdict: {final.verdict} | Evidence: {len(evidence)} sources"

    return HybridResult(final, evidence, ml_score, summary)


# ------------------------------------------------------------
# DASHBOARD ACCESSOR
# ------------------------------------------------------------

def get_model_status():

    return MODEL_STATUS, ML_LOADED