
import re
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.recommendation_engine import (
    _embed_texts,
    clean_text
)

logger = logging.getLogger(__name__)


def prepare_interaction(item):
    """Preserve title and supporting context."""

    title = clean_text(item.get("title"))
    content = clean_text(item.get("content"))[:400]
    subreddit = clean_text(item.get("subreddit"))
    genre = clean_text(item.get("native_genre"))

    if content.lower() in {
        "no description",
        "no description available",
        "link post - no description"
    }:
        content = ""

    if not title and not content:
        return ""

    return " ".join(
        part for part in [
            title,
            title,
            content,
            subreddit,
            genre
        ]
        if part
    )


def group_interactions(history, threshold=0.55):
    """
    Group interactions using semantic similarity.

    Uses complete-link grouping to avoid merging
    unrelated items through intermediate connections.
    """

    items = []
    documents = []

    # 1. Prepare interaction text.
    for item in history:
        text = prepare_interaction(item)

        if not text:
            continue

        items.append(item)
        documents.append(text)

    if not items:
        return []

    # 2. Generate semantic embeddings.
    try:
        embeddings = _embed_texts(documents)

        similarities = cosine_similarity(embeddings)
        grouping_method = "semantic_minilm"
        active_threshold = threshold

    except Exception:
        logger.exception(
            "Semantic grouping failed. Using TF-IDF fallback."
        )

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )

            matrix = vectorizer.fit_transform(documents)
            similarities = cosine_similarity(matrix)

        except ValueError:
            similarities = np.eye(len(items))

        grouping_method = "tfidf_fallback"
        active_threshold = 0.25

    # 3. Initially, every interaction is its own group.
    clusters = [[i] for i in range(len(items))]

    # 4. Merge the most similar compatible groups.
    while True:
        best_pair = None
        best_score = active_threshold

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                pair_scores = similarities[
                    np.ix_(clusters[i], clusters[j])
                ]

                # Require every cross-group pair
                # to meet the similarity threshold.
                score = float(pair_scores.min())

                if score >= best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair

        clusters[i].extend(clusters[j])
        clusters.pop(j)

    # 5. Build topic groups.
    groups = []

    for cluster in clusters:
        group_items = [items[i] for i in cluster]

        group_similarity = similarities[
            np.ix_(cluster, cluster)
        ]

        # Find the most representative interaction.
        representative_index = cluster[
            int(np.argmax(group_similarity.mean(axis=1)))
        ]

        representative_title = (
            items[representative_index].get("title")
            or documents[representative_index]
        )

        if len(cluster) > 1:
            upper = group_similarity[
                np.triu_indices(len(cluster), k=1)
            ]

            confidence = float(upper.mean())

        else:
            confidence = None

        groups.append({
            "topic": representative_title,
            "count": len(group_items),
            "sources": sorted(set(
                str(item.get("source") or "unknown")
                for item in group_items
            )),
            "interactions": group_items,
            "confidence": (
                round(confidence, 3)
                if confidence is not None
                else None
            ),
            "method": grouping_method
        })

    # 6. Prioritize recurring interests.
    groups.sort(
        key=lambda group: group["count"],
        reverse=True
    )

    return groups
