
import logging
import numpy as np

from app.services.recommendation_engine import (
    _embed_texts,
    clean_text
)


logger = logging.getLogger(__name__)


def rerank_candidates(candidates):
    """
    Rank retrieved content using semantic relevance,
    user preference strength and Google search rank.
    """

    if not candidates:
        return []

    candidates = [dict(item) for item in candidates]

    # 1. Prepare candidate documents.
    candidate_texts = []

    for item in candidates:

        title = clean_text(item.get("title"))
        snippet = clean_text(item.get("snippet"))

        text = f"{title} {title} {snippet}"

        candidate_texts.append(text.strip())

    # 2. Prepare unique interest queries.
    query_texts = {}

    for item in candidates:

        query = clean_text(
            item.get("matched_query")
        )

        if query not in query_texts:

            keywords = item.get("keywords") or []

            keyword_text = " ".join(
                str(word) for word in keywords
            )

            query_texts[query] = (
                f"{query} {keyword_text}"
            ).strip()

    unique_queries = list(query_texts.keys())

    # 3. Generate semantic embeddings in one batch.
    try:

        documents = (
            list(query_texts.values())
            + candidate_texts
        )

        embeddings = _embed_texts(documents)

        query_embeddings = embeddings[
            :len(unique_queries)
        ]

        candidate_embeddings = embeddings[
            len(unique_queries):
        ]

        query_indices = {
            query: index
            for index, query in enumerate(unique_queries)
        }

    except Exception:

        logger.exception(
            "Experimental semantic reranking failed"
        )

        for item in candidates:
            item["semantic_score"] = None
            item["final_score"] = None
            item["ranking_method"] = "search_order_fallback"

        return candidates

    # 4. Normalize preference scores.
    max_preference = max(
        float(item.get("preference_score") or 0)
        for item in candidates
    )

    # 5. Calculate personalized ranking scores.
    for index, item in enumerate(candidates):

        query = clean_text(
            item.get("matched_query")
        )

        query_index = query_indices[query]

        query_vector = query_embeddings[query_index]

        candidate_vector = candidate_embeddings[index]

        # MiniLM embeddings are already L2 normalized.
        semantic_score = float(
            np.dot(query_vector, candidate_vector)
        )

        semantic_score = float(
            np.clip(semantic_score, 0.0, 1.0)
        )

        preference = float(
            item.get("preference_score") or 0
        )

        preference_score = (
            preference / max_preference
            if max_preference > 0
            else 0.0
        )

        search_rank = max(
            1, int(item.get("search_rank") or 1)
        )

        search_score = 1.0 / search_rank

        final_score = (
            0.75 * semantic_score
            + 0.20 * preference_score
            + 0.05 * search_score
        )

        item["semantic_score"] = round(
            semantic_score, 4
        )

        item["normalized_preference_score"] = round(
            preference_score, 4
        )

        item["search_score"] = round(
            search_score, 4
        )

        item["final_score"] = round(
            final_score, 4
        )

        item["ranking_method"] = "semantic_minilm"

    # 6. Sort candidates by personalized relevance.
    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    # 7. Assign personalized ranking positions.
    for rank, item in enumerate(candidates, start=1):

        item["personalized_rank"] = rank

    return candidates
