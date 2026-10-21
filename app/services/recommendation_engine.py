"""Hybrid semantic ranking for OmniStream-AI search candidates.

The embedding model runs locally. If model loading fails, ranking falls back to
TF-IDF cosine similarity so recommendation endpoints remain available.
"""
import ipaddress
import logging
import os
import re
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OMNISTREAM_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_model = None
_tokenizer = None
_model_lock = threading.Lock()


def clean_text(value):
    """Normalize text without discarding non-English characters."""
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_http_url(value):
    """Only accept absolute HTTP(S) links with a hostname."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.hostname.lower() in {"localhost"}
        ):
            return False
        try:
            address = ipaddress.ip_address(parsed.hostname)
            if not address.is_global:
                return False
        except ValueError:
            # Ordinary DNS hostnames are valid; only literal IPs need this check.
            pass
        return True
    except (TypeError, ValueError):
        return False


def canonical_url(value):
    """Drop fragments and normalize host/default ports for deduplication."""
    if not is_valid_http_url(value):
        return ""
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
        and not key.lower().startswith("utm_")
    ]
    query = urlencode(query_items, doseq=True)
    return urlunsplit((scheme, host, path, query, ""))


def _history_text(item):
    if isinstance(item, str):
        return clean_text(item)
    if not isinstance(item, dict):
        return ""
    parts = [
        item.get("title", ""),
        item.get("content", ""),
        item.get("snippet", ""),
        item.get("native_genre", item.get("genre", "")),
        item.get("subreddit", ""),
        item.get("channel", ""),
    ]
    return clean_text(" ".join(str(part) for part in parts if part))


def _candidate_text(item):
    return clean_text(" ".join(
        str(item.get(key) or "")
        for key in ("title", "snippet", "description", "content", "source_domain")
    ))


def _load_embedding_model():
    """Load the multilingual encoder once, only when semantic ranking is used."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model
        from transformers import AutoModel, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(DEFAULT_EMBEDDING_MODEL)
        model.eval()
        _tokenizer, _model = tokenizer, model
        return tokenizer, model


def _embed_texts(texts, batch_size=16):
    import torch
    tokenizer, model = _load_embedding_model()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors) if vectors else np.empty((0, 0))


def _lexical_similarities(profile_text, candidate_texts):
    corpus = [profile_text] + candidate_texts
    if not profile_text.strip() or not any(text.strip() for text in candidate_texts):
        return np.zeros(len(candidate_texts), dtype=float)
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=5000).fit_transform(corpus)
        return cosine_similarity(matrix[1:], matrix[0]).ravel()
    except ValueError:
        # Empty vocabulary is possible for punctuation-only or very short text.
        return np.zeros(len(candidate_texts), dtype=float)


def _parse_timestamp(value):
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class RecommendationEngine:
    """Validate, deduplicate, and rank search candidates against user interests."""

    @staticmethod
    def rank_candidates(history_items, candidates, limit=None, now=None):
        if not candidates:
            return []

        # Reject malformed links and remove duplicate URLs before scoring.
        unique = []
        seen = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link")
            normalized = canonical_url(url)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidate = dict(item)
            candidate["url"] = url
            unique.append(candidate)

        if not unique:
            return []

        history_items = history_items or []
        history_texts = [_history_text(item) for item in history_items]
        profile_text = clean_text(" ".join(text for text in history_texts if text))
        candidate_texts = [_candidate_text(item) for item in unique]
        relevance = np.zeros(len(unique), dtype=float)

        if profile_text:
            lexical = _lexical_similarities(profile_text, candidate_texts)
            try:
                vectors = _embed_texts([profile_text] + candidate_texts)
                semantic = np.clip(vectors[1:] @ vectors[0], 0.0, 1.0)
                # Semantic meaning leads; lexical overlap stabilizes short queries.
                relevance = 0.75 * semantic + 0.25 * lexical
            except Exception as exc:
                logger.warning(
                    "Multilingual embedding ranking unavailable; using TF-IDF fallback: %s",
                    exc,
                )
                relevance = lexical

        preferred_genres = {
            str(item.get("native_genre") or item.get("genre") or "").strip().casefold()
            for item in history_items
            if isinstance(item, dict)
            and str(item.get("native_genre") or item.get("genre") or "").strip()
            and str(item.get("native_genre") or item.get("genre") or "").strip().casefold()
            not in {"general", "unknown"}
        }
        now_utc = _parse_timestamp(now or datetime.now(timezone.utc)) or datetime.now(timezone.utc)

        ranked = []
        for index, item in enumerate(unique):
            # Query labels are not verified candidate genres. Only independently
            # supplied candidate metadata may earn a genre-match bonus.
            genre = str(
                item.get("candidate_genre")
                or item.get("native_genre")
                or item.get("category")
                or ""
            ).strip().casefold()
            genre_match = 1.0 if genre and genre in preferred_genres else 0.0

            published = _parse_timestamp(
                item.get("published_at") or item.get("timestamp") or item.get("date")
            )
            if published is None:
                freshness = 0.5
            else:
                age_days = max(0.0, (now_utc - published).total_seconds() / 86400.0)
                freshness = float(np.exp(-np.log(2) * age_days / 30.0))

            try:
                quality = float(item.get("quality_score", item.get("search_score", 0.5)))
            except (TypeError, ValueError):
                quality = 0.5
            quality = min(1.0, max(0.0, quality))

            # These transparent weights are initial heuristics, not trained values.
            # Tune only after evaluating reviewed examples with Precision@K/nDCG@K.
            score = (
                0.65 * float(relevance[index])
                + 0.15 * genre_match
                + 0.10 * freshness
                + 0.10 * quality
            )
            item["relevance_score"] = round(float(relevance[index]), 4)
            item["recommendation_score"] = round(score, 4)
            ranked.append((score, index, item))

        ranked.sort(key=lambda row: (-row[0], row[1]))
        results = [row[2] for row in ranked]
        return results[:max(0, limit)] if limit is not None else results
