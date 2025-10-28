"""
evidence_pipeline.py
--------------------
Evidence & article retrieval for the Dual-LLM Hybrid Fact-Checker.

Features:
✅ Google Fact Check API
✅ Google Custom Search (CSE)
✅ NewsAPI (live headlines, 2025 support)
✅ Google RSS fallback
✅ Article extraction (Trafilatura / Newspaper3k / Playwright)
✅ Credibility ranking + safe fallback text
"""

import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("evidence")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------
@dataclass
class EvidenceItem:
    title: str
    snippet: str
    url: str
    source: str


# ------------------------------------------------------------
# Credibility lookup
# ------------------------------------------------------------
CREDIBLE_SOURCES = {
    "reuters.com": "Highly credible (Reuters)",
    "bbc.com": "Highly credible (BBC)",
    "apnews.com": "Highly credible (AP)",
    "nytimes.com": "Credible (NYT)",
    "theguardian.com": "Credible (Guardian)",
    "cnn.com": "Generally credible (CNN)",
    "washingtonpost.com": "Credible (WP)",
    "aljazeera.com": "Credible (Al Jazeera)",
    "npr.org": "Highly credible (NPR)",
    "dw.com": "Credible (DW)",
    "forbes.com": "Credible (Forbes)",
    "bloomberg.com": "Highly credible (Bloomberg)"
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _http_get(url: str, timeout: int = 10):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (CapstoneBot/1.0)"})
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


# ------------------------------------------------------------
# 1️⃣ Google Fact Check API
# ------------------------------------------------------------
def fetch_factcheck(query: str, max_results: int = 5) -> List[EvidenceItem]:
    if not GOOGLE_API_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": GOOGLE_API_KEY, "query": query, "languageCode": "en", "pageSize": max_results}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for c in data.get("claims", []):
            for review in c.get("claimReview", []):
                results.append(EvidenceItem(
                    title=review.get("title", "Fact Check"),
                    snippet=f"{review.get('publisher', {}).get('name', '')}: {review.get('textualRating', '')}",
                    url=review.get("url", ""),
                    source="factcheck"
                ))
        return results
    except Exception as e:
        log.warning(f"Fact Check API error: {e}")
        return []


# ------------------------------------------------------------
# 2️⃣ Google Custom Search (CSE)
# ------------------------------------------------------------
import os, requests, logging
from typing import List

log = logging.getLogger(__name__)

class EvidenceItem:
    def __init__(self, title, snippet, url, source="google"):
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source

    def __repr__(self):
        return f"[{self.source.upper()}] {self.title} → {self.url}"

def fetch_cse(query: str, max_results: int = 6) -> List[EvidenceItem]:
    """
    Fetch evidence via Google Custom Search JSON API.
    Returns list of EvidenceItem objects.
    """
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

        if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
            log.warning("⚠️ Missing GOOGLE_API_KEY or SEARCH_ENGINE_ID")
            return []

        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
            f"&q={requests.utils.quote(query)}"
            f"&num={max_results}&lr=en&safe=off"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            log.warning(f"❌ CSE request failed ({r.status_code}): {r.text}")
            return []

        data = r.json()
        items = data.get("items", [])
        if not items:
            log.warning("⚠️ No results found in CSE response.")
            return []

        results = []
        for item in items[:max_results]:
            title = item.get("title")
            snippet = item.get("snippet") or "No description available."
            link = item.get("link")
            if title and link:
                results.append(EvidenceItem(title, snippet, link, "google"))

        log.info(f"✅ Retrieved {len(results)} CSE evidence items.")
        return results

    except Exception as e:
        log.error(f"❌ Error in fetch_cse: {e}")
        return []

def gather_evidence(claim: str) -> List[EvidenceItem]:
    """
    Unified evidence retrieval — merges multiple sources.
    """
    evidence = []
    evidence += fetch_cse(claim)
    return evidence



# ------------------------------------------------------------
# 3️⃣ NewsAPI (live headlines, 2025 support)
# ------------------------------------------------------------
def fetch_newsapi(query: str, max_results: int = 6) -> List[EvidenceItem]:
    if not NEWS_API_KEY:
        return []
    try:
        url = (
            "https://newsapi.org/v2/top-headlines?"
            f"q={requests.utils.quote(query)}&language=en&pageSize={max_results}&apiKey={NEWS_API_KEY}"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            log.warning(f"NewsAPI HTTP {r.status_code}")
            return []
        data = r.json()
        articles = data.get("articles", [])
        results = [
            EvidenceItem(
                title=a.get("title", ""),
                snippet=a.get("description", ""),
                url=a.get("url", ""),
                source="newsapi"
            )
            for a in articles if a.get("title")
        ]
        return results
    except Exception as e:
        log.warning(f"NewsAPI error: {e}")
        return []

# ------------------------------------------------------------
# 4️⃣ RSS fallback
# ------------------------------------------------------------
def fetch_rss(query: str, max_results: int = 6) -> List[EvidenceItem]:
    import feedparser
    rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(rss_url)
        results = []
        for entry in feed.entries[:max_results]:
            results.append(EvidenceItem(
                title=entry.get("title", "Untitled"),
                snippet=entry.get("summary", "No summary."),
                url=entry.get("link", ""),
                source="rss"
            ))
        return results
    except Exception as e:
        log.warning(f"RSS fallback error: {e}")
        return []


# ------------------------------------------------------------
# 5️⃣ Article Extraction (multi-method)
# ------------------------------------------------------------
def extract_article(url: str) -> str:
    text = ""

    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        if len(art.text.strip()) > 300:
            log.info("Extracted via Newspaper3k")
            return art.text.strip()
    except Exception:
        pass

    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.strip()) > 300:
                log.info("Extracted via Trafilatura")
                return text.strip()
    except Exception:
        pass

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            content = page.inner_text("body")
            browser.close()
            if len(content.strip()) > 300:
                log.info("Extracted via Playwright")
                return content.strip()
    except Exception:
        pass

    return text


# ------------------------------------------------------------
# Master evidence gatherer
# ------------------------------------------------------------
def gather_evidence(query: str, max_total: int = 10) -> List[EvidenceItem]:
    log.info(f"Gathering evidence for: {query[:80]}...")
    all_evidence = []

    # 1. Fact Check
    fc = fetch_factcheck(query)
    all_evidence.extend(fc)

    # 2. Google CSE
    if len(all_evidence) < max_total:
        cs = fetch_cse(query)
        all_evidence.extend(cs)

    # 3. NewsAPI (live)
    if len(all_evidence) < max_total:
        na = fetch_newsapi(query)
        all_evidence.extend(na)

    # 4. RSS fallback
    if len(all_evidence) < max_total:
        rf = fetch_rss(query)
        all_evidence.extend(rf)

    # Fallback
    if not all_evidence:
        all_evidence.append(EvidenceItem(
            title="No external evidence found.",
            snippet="No relevant fact-checks or reports could be located for this claim.",
            url="",
            source="system"
        ))

    # Remove duplicates and limit
    seen = set()
    unique = []
    for ev in all_evidence:
        if ev.url not in seen and ev.url:
            unique.append(ev)
            seen.add(ev.url)

    log.info(f"✅ {len(unique)} evidence items collected.")
    return unique[:max_total]


# ------------------------------------------------------------
# Utility: Build readable context for LLMs
# ------------------------------------------------------------
def build_evidence_context(evidence_items: List[EvidenceItem]) -> str:
    """
    Builds a readable context string for LLM input or dashboard display.
    """
    if not evidence_items:
        return "No external evidence found."
    context = "\n".join(
        [f"- {e.title} ({e.source}) → {e.url}\n  {e.snippet}" for e in evidence_items]
    )
    return f"Evidence gathered:\n{context}"
