import logging
import os

from googleapiclient.discovery import build

from app.services.recommendation_engine import RecommendationEngine, canonical_url

logger = logging.getLogger(__name__)


class ContentService:
    """Google Custom Search adapter with consistent, safe result normalization."""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.service = None

        if self.api_key and self.cse_id:
            try:
                self.service = build(
                    "customsearch",
                    "v1",
                    developerKey=self.api_key,
                    cache_discovery=False,
                )
            except Exception:
                logger.exception("Failed to initialize Custom Search Service")
        else:
            logger.warning("Google API credentials missing. Search will be disabled.")

    def scrape_content(self, search_term, num_results=5, start=1):
        """Return only unique HTTP(S) results; an unavailable search returns []."""
        if not self.service or not isinstance(search_term, str) or not search_term.strip():
            return []

        try:
            response = self.service.cse().list(
                q=search_term.strip(),
                cx=self.cse_id,
                num=max(1, min(int(num_results), 10)),
                start=max(1, int(start)),
                safe="active",
            ).execute()
        except Exception:
            logger.exception("Error fetching Google Custom Search results")
            return []

        results = []
        seen = set()
        for item in response.get("items", []):
            url = item.get("link")
            normalized_url = canonical_url(url)
            if not normalized_url or normalized_url in seen:
                continue
            seen.add(normalized_url)

            pagemap = item.get("pagemap") or {}
            image_url = ""
            for image_type in ("cse_thumbnail", "cse_image"):
                images = pagemap.get(image_type) or []
                if images and isinstance(images[0], dict):
                    candidate_image = images[0].get("src", "")
                    if canonical_url(candidate_image):
                        image_url = candidate_image
                        break

            results.append({
                "title": item.get("title") or "Untitled result",
                "link": url,
                "url": url,
                "snippet": item.get("snippet") or "",
                "source_domain": item.get("displayLink") or "Web",
                "image_url": image_url,
            })
        return results
