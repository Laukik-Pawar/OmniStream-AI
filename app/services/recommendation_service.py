import os
import logging
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

class RecommendationService:
    @staticmethod
    def fetch_google_cse_results(search_queries, num_results_per_query=3):
        """Fetches live web content and normalizes the metadata for the frontend."""
        api_key = os.environ.get('GOOGLE_API_KEY')
        cse_id = os.environ.get('GOOGLE_CSE_ID')
        
        if not api_key or not cse_id:
            logger.error("Google API Key or CSE ID is missing from environment variables.")
            return []

        recommendations = []
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            
            for query_obj in search_queries:
                query_text = query_obj.get('query')
                genre = query_obj.get('genre', 'General')
                
                if not query_text or query_text.lower() == 'news':
                    continue
                    
                res = service.cse().list(
                    q=query_text, 
                    cx=cse_id, 
                    num=num_results_per_query,
                    safe="off"
                ).execute()
                
                for item in res.get('items', []):
                    # Safely extract a thumbnail image from Google's pagemap metadata
                    pagemap = item.get('pagemap', {})
                    thumbnail_url = ''
                    if 'cse_thumbnail' in pagemap and len(pagemap['cse_thumbnail']) > 0:
                        thumbnail_url = pagemap['cse_thumbnail'][0].get('src', '')

                    recommendations.append({
                        'title': item.get('title', 'No Title'),
                        'url': item.get('link', '#'),
                        'snippet': item.get('snippet', ''),
                        'source_domain': item.get('displayLink', 'Web'),
                        'genre': genre,
                        'matched_query': query_text,  # <-- THIS IS THE FIX
                        'image_url': thumbnail_url
                    })
                    
        except Exception as e:
            logger.error(f"Error fetching Google CSE recommendations: {e}")
            
        return recommendations