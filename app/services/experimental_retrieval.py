
import logging

from app.services.content_service import ContentService


logger = logging.getLogger(__name__)


def retrieve_candidates(queries, results_per_query=10):
    """
    Retrieve candidate recommendations using
    personalized search queries.
    """

    content_service = ContentService()

    candidates = []
    search_summary = []

    results_per_query = max(
        1, min(int(results_per_query), 10)
    )

    if content_service.service is None:
        return {
            "candidates": [],
            "summary": {
                "status": "search_unavailable",
                "queries_searched": 0,
                "total_candidates": 0,
                "searches": []
            }
        }

    # 1. Execute each personalized search query.
    for query_info in queries:

        query = str(
            query_info.get("query") or ""
        ).strip()

        if not query:
            continue

        try:

            results = content_service.scrape_content(
                search_term=query,
                num_results=results_per_query
            )

            retrieved_count = 0

            # 2. Standardize candidate metadata.
            for rank, item in enumerate(
                results, start=1
            ):

                url = item.get("url") or item.get("link")

                if not url:
                    continue

                candidates.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "source_domain": item.get(
                        "source_domain", "Web"
                    ),
                    "image_url": item.get(
                        "image_url", ""
                    ),

                    # Personalized query information.
                    "matched_query": query,
                    "topic": query_info.get("topic", ""),
                    "interests": query_info.get(
                        "interests", []
                    ),
                    "keywords": query_info.get(
                        "keywords", []
                    ),
                    "interest_source": query_info.get(
                        "source", "unknown"
                    ),
                    "preference_score": query_info.get(
                        "preference_score", 0
                    ),
                    "group_id": query_info.get("group_id"),

                    # Retrieval information.
                    "search_rank": rank,
                    "retrieval_method": "google_cse"
                })

                retrieved_count += 1

            search_summary.append({
                "query": query,
                "requested_results": results_per_query,
                "retrieved_results": retrieved_count,
                "status": (
                    "success"
                    if retrieved_count
                    else "empty_or_failed"
                )
            })

        except Exception:

            logger.exception(
                "Experimental retrieval failed for: %s",
                query
            )

            search_summary.append({
                "query": query,
                "requested_results": results_per_query,
                "retrieved_results": 0,
                "status": "failed"
            })

    # 3. Build retrieval summary.
    return {
        "candidates": candidates,
        "summary": {
            "status": (
                "completed"
                if candidates
                else "no_candidates"
            ),
            "queries_searched": len(search_summary),
            "results_per_query": results_per_query,
            "total_candidates": len(candidates),
            "successful_searches": sum(
                item["status"] == "success"
                for item in search_summary
            ),
            "searches": search_summary
        }
    }
