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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    """Fetch evidence from Google Fact Check Tools API"""
    if not GOOGLE_API_KEY:
        log.warning("⚠️ GOOGLE_API_KEY not set for Fact Check API")
        return []
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": GOOGLE_API_KEY,
        "query": query,
        "languageCode": "en",
        "pageSize": max_results
    }
    
    try:
        log.info(f"🔍 Querying Fact Check API: {query[:50]}...")
        r = requests.get(url, params=params, timeout=10)
        
        if r.status_code != 200:
            log.warning(f"❌ Fact Check API returned {r.status_code}: {r.text[:200]}")
            return []
        
        data = r.json()
        results = []
        
        for c in data.get("claims", []):
            for review in c.get("claimReview", []):
                results.append(EvidenceItem(
                    title=review.get("title", "Fact Check"),
                    snippet=f"{review.get('publisher', {}).get('name', 'Unknown')}: {review.get('textualRating', 'No rating')}",
                    url=review.get("url", ""),
                    source="factcheck"
                ))
        
        log.info(f"✅ Fact Check API returned {len(results)} items")
        return results
        
    except Exception as e:
        log.error(f"❌ Fact Check API error: {e}")
        return []


# ------------------------------------------------------------
# 2️⃣ Google Custom Search (CSE)
# ------------------------------------------------------------
def fetch_cse(query: str, max_results: int = 6) -> List[EvidenceItem]:
    """
    Fetch evidence via Google Custom Search JSON API.
    Returns list of EvidenceItem objects.
    """
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        log.warning("⚠️ Missing GOOGLE_API_KEY or SEARCH_ENGINE_ID for CSE")
        return []
    
    try:
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
            f"&q={requests.utils.quote(query)}"
            f"&num={max_results}&lr=lang_en&safe=off"
        )
        
        log.info(f"🔍 Querying Google CSE: {query[:50]}...")
        r = requests.get(url, timeout=10)
        
        if r.status_code != 200:
            log.warning(f"❌ CSE request failed ({r.status_code}): {r.text[:200]}")
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


# ------------------------------------------------------------
# 3️⃣ NewsAPI (live headlines, 2025 support)
# ------------------------------------------------------------
def fetch_newsapi(query: str, max_results: int = 6) -> List[EvidenceItem]:
    """Fetch live news from NewsAPI"""
    if not NEWS_API_KEY:
        log.warning("⚠️ NEWS_API_KEY not set")
        return []
    
    try:
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={requests.utils.quote(query)}&language=en&pageSize={max_results}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        
        log.info(f"🔍 Querying NewsAPI: {query[:50]}...")
        r = requests.get(url, timeout=10)
        
        if r.status_code != 200:
            log.warning(f"❌ NewsAPI HTTP {r.status_code}: {r.text[:200]}")
            return []
        
        data = r.json()
        articles = data.get("articles", [])
        
        results = [
            EvidenceItem(
                title=a.get("title", "Untitled"),
                snippet=a.get("description", "No description"),
                url=a.get("url", ""),
                source="newsapi"
            )
            for a in articles if a.get("title") and a.get("url")
        ]
        
        log.info(f"✅ NewsAPI returned {len(results)} articles")
        return results
        
    except Exception as e:
        log.error(f"❌ NewsAPI error: {e}")
        return []


# ------------------------------------------------------------
# 4️⃣ RSS fallback
# ------------------------------------------------------------
def fetch_rss(query: str, max_results: int = 6) -> List[EvidenceItem]:
    """Fallback to Google News RSS feed"""
    try:
        import feedparser
    except ImportError:
        log.warning("⚠️ feedparser not installed, skipping RSS")
        return []
    
    rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        log.info(f"🔍 Querying Google News RSS: {query[:50]}...")
        feed = feedparser.parse(rss_url)
        results = []
        
        for entry in feed.entries[:max_results]:
            results.append(EvidenceItem(
                title=entry.get("title", "Untitled"),
                snippet=entry.get("summary", "No summary."),
                url=entry.get("link", ""),
                source="rss"
            ))
        
        log.info(f"✅ RSS returned {len(results)} items")
        return results
        
    except Exception as e:
        log.error(f"❌ RSS fallback error: {e}")
        return []


# ------------------------------------------------------------
# 5️⃣ Article Extraction (multi-method)
# ------------------------------------------------------------
def extract_article(url: str) -> str:
    """Extract article text from URL using multiple methods"""
    text = ""

    # Method 1: Newspaper3k
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        if len(art.text.strip()) > 300:
            log.info("✅ Extracted via Newspaper3k")
            return art.text.strip()
    except Exception as e:
        log.debug(f"Newspaper3k failed: {e}")

    # Method 2: Trafilatura
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.strip()) > 300:
                log.info("✅ Extracted via Trafilatura")
                return text.strip()
    except Exception as e:
        log.debug(f"Trafilatura failed: {e}")

    # Method 3: Playwright (slower, last resort)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            content = page.inner_text("body")
            browser.close()
            if len(content.strip()) > 300:
                log.info("✅ Extracted via Playwright")
                return content.strip()
    except Exception as e:
        log.debug(f"Playwright failed: {e}")

    log.warning(f"⚠️ Could not extract article from {url}")
    return text


# ------------------------------------------------------------
# Master evidence gatherer
# ------------------------------------------------------------
def gather_evidence(query: str, max_total: int = 10) -> List[EvidenceItem]:
    """
    Main evidence gathering function - tries multiple sources.
    Returns list of EvidenceItem objects.
    """
    log.info(f"🔎 Gathering evidence for: '{query[:80]}...'")
    all_evidence = []

    # 1. Fact Check API (most reliable for fact-checking)
    fc = fetch_factcheck(query)
    all_evidence.extend(fc)
    log.info(f"After Fact Check: {len(all_evidence)} items")

    # 2. Google CSE (general web search)
    if len(all_evidence) < max_total:
        cs = fetch_cse(query)
        all_evidence.extend(cs)
        log.info(f"After CSE: {len(all_evidence)} items")

    # 3. NewsAPI (live news)
    if len(all_evidence) < max_total:
        na = fetch_newsapi(query)
        all_evidence.extend(na)
        log.info(f"After NewsAPI: {len(all_evidence)} items")

    # 4. RSS fallback
    if len(all_evidence) < max_total:
        rf = fetch_rss(query)
        all_evidence.extend(rf)
        log.info(f"After RSS: {len(all_evidence)} items")

    # Remove duplicates based on URL
    seen = set()
    unique = []
    for ev in all_evidence:
        if ev.url and ev.url not in seen:
            unique.append(ev)
            seen.add(ev.url)
        elif not ev.url:  # Keep items without URL (like system messages)
            unique.append(ev)

    # Fallback message if nothing found
    if not unique:
        log.warning("⚠️ No evidence found from any source!")
        unique.append(EvidenceItem(
            title="No external evidence found.",
            snippet="No relevant fact-checks or reports could be located for this claim.",
            url="",
            source="system"
        ))

    log.info(f"✅ Final evidence count: {len(unique)} unique items")
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
    
    if len(evidence_items) == 1 and evidence_items[0].source == "system":
        return "No external evidence found."
    
    context_parts = []
    for i, e in enumerate(evidence_items, 1):
        source_label = f"[{e.source.upper()}]" if e.source != "system" else ""
        context_parts.append(
            f"{i}. {source_label} {e.title}\n"
            f"   {e.snippet}\n"
            f"   URL: {e.url if e.url else 'N/A'}"
        )
    
    return "Evidence gathered:\n" + "\n\n".join(context_parts)