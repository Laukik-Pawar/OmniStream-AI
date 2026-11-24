
import re

import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity

from app.services.recommendation_engine import _embed_texts


CUSTOM_STOPWORDS = {
    "guys", "finally", "just", "really",
    "viral", "trending", "video", "shorts",
    "youtubeshorts", "official", "please",
    "ama", "think", "hope", "today",
    "thing", "things", "damn", "available"
}

STOPWORDS = sorted(
    set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS
)


def clean_interest_text(text):
    """Clean text while preserving meaningful phrases."""

    text = str(text or "")

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(?=\w)", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_interests(topic_groups, top_n=5):
    """
    Extract representative interest phrases
    from semantic topic groups.
    """

    documents = []
    valid_groups = []

    # 1. Combine interactions within each group.
    for group in topic_groups:

        parts = []

        for item in group["interactions"]:

            title = clean_interest_text(
                item.get("title")
            )

            content = clean_interest_text(
                item.get("content")
            )[:300]

            if content.lower() in {
                "no description",
                "no description available",
                "link post - no description"
            }:
                content = ""

            parts.append(
                f"{title}. {title}. {content}"
            )

        document = " ".join(parts).strip()

        if document:
            documents.append(document)
            valid_groups.append(group)

    if not documents:
        return []

    # 2. Extract candidate phrases using TF-IDF.
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w][\w-]*\b"
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return []

    terms = vectorizer.get_feature_names_out()

    profiles = []

    # 3. Process each semantic group.
    for index, group in enumerate(valid_groups):

        row = matrix.getrow(index)

        ranked = sorted(
            zip(row.indices, row.data),
            key=lambda pair: pair[1],
            reverse=True
        )

        candidates = []

        for term_index, score in ranked[:30]:

            phrase = terms[term_index]

            if len(phrase) < 3 or phrase.isnumeric():
                continue

            candidates.append({
                "phrase": phrase,
                "tfidf": float(score)
            })

        if not candidates:
            continue

        # 4. Compare phrases with the group's meaning.
        try:
            texts = [documents[index]] + [
                item["phrase"]
                for item in candidates
            ]

            embeddings = _embed_texts(texts)

            similarities = cosine_similarity(
                embeddings[1:],
                embeddings[:1]
            ).ravel()

            max_tfidf = max(
                item["tfidf"]
                for item in candidates
            )

            for i, item in enumerate(candidates):

                lexical = item["tfidf"] / max_tfidf

                semantic = float(
                    np.clip(similarities[i], 0, 1)
                )

                item["score"] = (
                    0.7 * semantic +
                    0.3 * lexical
                )

        except Exception:

            for item in candidates:
                item["score"] = item["tfidf"]

        # 5. Rank candidate interest phrases.
        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected = []

        for item in candidates:

            phrase = item["phrase"]

            # Avoid redundant overlapping phrases.
            phrase_words = set(phrase.split())

            redundant = any(
                phrase_words.issubset(
                    set(existing.split())
                )
                or set(existing.split()).issubset(
                    phrase_words
                )
                for existing in selected
            )

            if redundant:
                continue

            selected.append(phrase)

            if len(selected) >= top_n:
                break

        # 6. Build the interest profile.
        profiles.append({
            "topic": selected[0] if selected else group["topic"],
            "interests": selected,
            "count": group["count"],
            "sources": group["sources"],
            "confidence": group.get("confidence"),
            "titles": [
                item.get("title", "")
                for item in group["interactions"]
            ]
        })

    profiles.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    return profiles
