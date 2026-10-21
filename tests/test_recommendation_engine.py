import unittest
from unittest.mock import patch

from app.services.recommendation_engine import RecommendationEngine, canonical_url


class RecommendationEngineTests(unittest.TestCase):
    def test_rejects_invalid_urls_and_deduplicates_fragments(self):
        candidates = [
            {"title": "Python tutorial", "url": "https://example.com/python#section"},
            {"title": "Duplicate page", "url": "https://example.com/python#other"},
            {"title": "Unsafe link", "url": "javascript:alert(1)"},
            {"title": "Relative link", "url": "/relative/path"},
        ]
        with patch(
            "app.services.recommendation_engine._embed_texts",
            side_effect=RuntimeError("model not installed"),
        ):
            results = RecommendationEngine.rank_candidates(
                [{"title": "Python programming"}], candidates
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Python tutorial")
        self.assertTrue(results[0]["url"].startswith("https://"))
        self.assertIn("recommendation_score", results[0])

    def test_lexical_fallback_ranks_relevant_content_first(self):
        candidates = [
            {"title": "Celebrity gossip", "url": "https://example.com/celeb"},
            {"title": "Python data engineering tutorial", "url": "https://example.com/python"},
        ]
        with patch(
            "app.services.recommendation_engine._embed_texts",
            side_effect=RuntimeError("offline"),
        ):
            results = RecommendationEngine.rank_candidates(
                [{"title": "Python and data engineering", "content": "Spark pipelines"}],
                candidates,
            )

        self.assertEqual(results[0]["title"], "Python data engineering tutorial")

    def test_query_label_does_not_count_as_verified_candidate_genre(self):
        candidates = [
            {
                "title": "Same candidate text",
                "url": "https://example.com/query-label",
                "genre": "Technology",
            },
            {
                "title": "Same candidate text",
                "url": "https://example.com/verified-label",
                "candidate_genre": "Technology",
            },
        ]
        with patch(
            "app.services.recommendation_engine._embed_texts",
            side_effect=RuntimeError("offline"),
        ):
            results = RecommendationEngine.rank_candidates(
                [{"title": "General interests", "native_genre": "Technology"}],
                candidates,
                now="2026-10-21T12:00:00Z",
            )

        scores = {item["url"]: item["recommendation_score"] for item in results}
        self.assertGreater(
            scores["https://example.com/verified-label"],
            scores["https://example.com/query-label"],
        )

    def test_empty_history_still_returns_valid_unique_candidates(self):
        candidates = [
            {"title": "First", "url": "https://example.com/1"},
            {"title": "Second", "url": "https://example.com/2"},
        ]
        results = RecommendationEngine.rank_candidates([], candidates, limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "First")

    def test_canonical_url_removes_fragment_and_default_port(self):
        self.assertEqual(
            canonical_url("https://EXAMPLE.com:443/path/#part"),
            "https://example.com/path",
        )


if __name__ == "__main__":
    unittest.main()
