"""
ai_factcheck.py
---------------
Dual-LLM Hybrid Fact-Checker:
✅ GPT-5 for recent and real-time events
✅ Gemini 2.5 Flash for older or general reasoning
✅ Consensus logic between models
✅ Fallback reasoning and evidence aggregation
"""

import os, re, json, logging, requests, datetime, numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from evidence_pipeline import gather_evidence, extract_article, build_evidence_context, EvidenceItem
from evidence_pipeline import (
    gather_evidence,
    extract_article,
    build_evidence_context,
    EvidenceItem,
)



# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hybrid_factcheck")

# --- Load environment ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Optional GPT-5 import ---
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    client = None
    OPENAI_API_KEY = None

# --- Gemini setup ---
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"


# ------------------------------------------------------------
# ✅ MODEL CONNECTIVITY CHECK
# ------------------------------------------------------------
def check_model_connectivity():
    status = {"gemini": False, "gpt5": False}
    # --- Gemini test ---
       # --- Gemini test (fixed) ---
    try:
        test_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # Simple ping — Gemini sometimes returns no text for short prompts, so we only check candidate presence
        resp = test_model.generate_content("ping")
        if hasattr(resp, "candidates") and resp.candidates:
            status["gemini"] = True
            log.info("✅ Gemini connection verified.")
        else:
            log.warning("⚠️ Gemini reachable but returned empty candidate list.")
    except Exception as e:
        log.warning(f"⚠️ Gemini connection failed: {e}")


    # --- GPT-5 test ---
    if client and OPENAI_API_KEY:
        try:
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "ping"}]
            )
            if resp.choices[0].message.content:
                status["gpt5"] = True
                log.info("✅ GPT-5 connection verified.")
        except Exception as e:
            log.warning(f"⚠️ GPT-5 connection failed: {e}")
    else:
        log.warning("⚠️ GPT-5 client not initialized or API key missing.")

    return status

# Run connectivity check once on startup
MODEL_STATUS = check_model_connectivity()


# --- Data structures ---
@dataclass
class EvidenceItem:
    title: str
    snippet: str
    url: str
    source: str

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
        "earlier this", "as of", "update", "unfolding"
    ]

    return any(term in t for term in recent_terms)


# ------------------------------------------------------------
# Evidence gathering (placeholder – use your existing one)
# ------------------------------------------------------------
def gather_evidence(query: str) -> List[EvidenceItem]:
    # Placeholder evidence retrieval stub
    return [EvidenceItem("No evidence found", "Evidence system placeholder.", "", "none")]

def get_article_text(url: str) -> str:
    return url  # placeholder for your extractor logic

# ------------------------------------------------------------
# GPT-5 and Gemini verdict functions
# ------------------------------------------------------------
def gpt5_verdict(text: str, evidence_context: str) -> GeminiResult:
    if not client:
        return GeminiResult("Uncertain", "GPT-5 unavailable.", None)

    prompt = f"""
You are a factual verification assistant.
Return ONLY valid JSON with keys: verdict ("True"|"False"|"Uncertain"), truth_score (0–100), explanation.

Claim:
{text[:4000]}

Evidence (summarized):
{evidence_context}
"""
    try:
        r = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a truth verification assistant. Reply only in JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        data = json.loads(r.choices[0].message.content)
        return GeminiResult(
            data.get("verdict", "Uncertain"),
            data.get("explanation", "No explanation."),
            data.get("truth_score")
        )
    except Exception as e:
        log.warning(f"GPT-5 verdict failed: {e}")
        return GeminiResult("Uncertain", "GPT-5 unavailable or unstructured.", None)

def gemini_verdict(text: str, evidence_context: str) -> GeminiResult:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    prompt = f"""
Return ONLY JSON with keys: verdict ("True"|"False"|"Uncertain"), truth_score (0–100), explanation.

Claim:
{text[:4000]}

Evidence (summarized):
{evidence_context}
"""
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(resp.text if hasattr(resp, "text") else str(resp))
        return GeminiResult(
            data.get("verdict", "Uncertain"),
            data.get("explanation", "No explanation."),
            data.get("truth_score")
        )
    except Exception as e:
        log.warning(f"Gemini verdict failed: {e}")
        return GeminiResult("Uncertain", "Gemini response unstructured.", None)


# ------------------------------------------------------------
# Fallback reasoning
# ------------------------------------------------------------
def fallback_reasoning(text: str) -> GeminiResult:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    prompt = f"""
The claim below returned no clear factual evidence.
Provide a brief, educational explanation of why this may be the case.

Claim:
{text[:600]}

Return JSON with keys: verdict, explanation, truth_score.
"""
    try:
        resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        data = json.loads(resp.text if hasattr(resp,"text") else str(resp))
        return GeminiResult(data.get("verdict","Uncertain"), data.get("explanation","No info."), data.get("truth_score",50))
    except Exception:
        return GeminiResult("Uncertain", "No factual context found.", 50)

# ------------------------------------------------------------
# Dual-LLM hybrid controller
# ------------------------------------------------------------
def hybrid_fact_check(user_input: str) -> HybridResult:
    text = user_input.strip()
    url_mode = bool(re.match(r"^https?://", text))
    if url_mode:
        text = get_article_text(text)

    # --- Evidence retrieval + LLM-ready summary context ---
    evidence = gather_evidence(text)
    evidence_context = build_evidence_context(evidence)

    # --- Decide recent vs historical and call LLMs with evidence_context ---
    recent_mode = is_recent_claim(text)
    gpt_res = None
    gem_res = None

    if recent_mode and OPENAI_API_KEY:
       log.info("🕒 Recent event detected — prioritizing GPT-5.")
       gpt_res = gpt5_verdict(text, evidence_context)
       # also get Gemini for cross-check if you want consensus:
       gem_res = gemini_verdict(text, evidence_context)
    else:
       log.info("📘 Historical or Gemini-only mode.")
       gem_res = gemini_verdict(text, evidence_context)
    if OPENAI_API_KEY:
        gpt_res = gpt5_verdict(text, evidence_context)



    # --- consensus logic ---
    if gem_res and gpt_res:
        if gem_res.verdict == gpt_res.verdict:
            final = gem_res if (gem_res.truth_score or 0) >= (gpt_res.truth_score or 0) else gpt_res
            final.explanation += f"\n\n✅ Both models agree ({final.verdict})."
        else:
            explanation = f"⚖️ Gemini says {gem_res.verdict}, GPT-5 says {gpt_res.verdict}. Defaulting to 'Uncertain'."
            score = int(np.mean([(gem_res.truth_score or 50), (gpt_res.truth_score or 50)]))
            final = GeminiResult("Uncertain", explanation, score)
    else:
        final = gpt_res or gem_res

    gem = final
    ml_score = None

    # --- fallback reasoning ---
    if (gem.verdict.lower() == "uncertain" or not gem.explanation.strip()) and len(evidence) <= 1:
        gem = fallback_reasoning(text)

    # --- summary ---
    summary_parts = [
        f"Verdict: {gem.verdict}",
        f"Truth Score: {gem.truth_score if gem.truth_score is not None else 'N/A'}",
        f"Evidence: {len(evidence)}"
    ]
    if gpt_res and gem_res:
        summary_parts.append("Dual-LLM Mode: GPT-5 + Gemini")
    elif gpt_res:
        summary_parts.append("Primary Model: GPT-5")
    elif gem_res:
        summary_parts.append("Primary Model: Gemini")
    summary = " | ".join(summary_parts)

    if not evidence or all("No external" in e.title for e in evidence):
     evidence = [EvidenceItem(
        title="No external evidence found.",
        snippet="No relevant fact-checks or sources were located.",
        url="",
        source="system"
    )]


    return HybridResult(gem, evidence, ml_score, summary)
