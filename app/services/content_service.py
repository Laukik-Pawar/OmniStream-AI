import os
import logging
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

class ContentService:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if self.api_key and self.cse_id:
            try:
                self.service = build("customsearch", "v1", developerKey=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Custom Search Service: {e}")
                self.service = None
        else:
            self.service = None
            logger.warning("Google API credentials missing. Search will be disabled.")

    def scrape_content(self, search_term, num_results=5, start=1):
        """Fetch fresh content recommendations from the web."""
        if not self.service:
            return [("Search disabled: Missing credentials", "#")]

        try:
            res = self.service.cse().list(
                q=search_term, 
                cx=self.cse_id, 
                num=num_results, 
                start=start
            ).execute()

            results = []
            for item in res.get('items', []):
                results.append({
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'snippet': item.get('snippet', '')
                })
            
            return results if results else [{"title": "No relevant results found", "link": "#"}]
        
        except Exception as e:
            logger.error(f"Error fetching search results: {e}")
            return [{"title": "Unable to fetch results at the moment.", "link": "#"}]