import praw
from datetime import datetime, timezone
import logging

# Set up basic logging for the service
logger = logging.getLogger(__name__)

class RedditService:
    def __init__(self, credentials: dict):
        """Initialize the Reddit client with OAuth 2.0 Web App credentials."""
        try:
            # Reconstruct the PRAW client using the OAuth refresh_token
            self.reddit = praw.Reddit(
                client_id=credentials.get('client_id'),
                client_secret=credentials.get('client_secret'),
                refresh_token=credentials.get('refresh_token'),
                user_agent=credentials.get('user_agent', 'web:omnistream-ai:v1.0')
            )
            
            # Verify the credentials by fetching the authenticated user
            self.user = self.reddit.user.me()
            if not self.user:
                raise ValueError("Reddit authentication failed: Invalid credentials.")
        except Exception as e:
            logger.error(f"Failed to initialize RedditService: {e}")
            raise

    def _format_item(self, item, signal_type):
        """Standardize Reddit models into our unified data format."""
        # Using timezone.utc to avoid Python 3.12 deprecation warnings on utcfromtimestamp
        timestamp = datetime.fromtimestamp(item.created_utc, timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Differentiate between a Comment and a Submission (Post)
        is_comment = isinstance(item, praw.models.Comment)
        
        return {
            'source': 'reddit', # Lowercase ensures the Jinja template applies the 'warning' badge
            'type': 'Comment' if is_comment else 'Submission',
            'title': item.submission.title if is_comment else item.title,
            'content': item.body if is_comment else getattr(item, 'selftext', 'Link Post - No Description'),
            'url': f"https://reddit.com{item.permalink}" if is_comment else item.url,
            'subreddit': item.subreddit.display_name,
            'timestamp': timestamp,
            'signal': signal_type
        }

    def fetch_history(self, limit=10):
        """Fetch user's saved and upvoted content."""
        history = []
        try:
            # 1. Fetch saved posts and comments
            for item in self.user.saved(limit=limit):
                history.append(self._format_item(item, signal_type="saved"))

            # 2. Fetch upvoted posts and comments
            for item in self.user.upvoted(limit=limit):
                history.append(self._format_item(item, signal_type="upvoted"))

        except Exception as e:
            logger.error(f"Error fetching Reddit history: {e}")
        
        return history

    def fetch_upvoted_posts(self, limit=25):
        """Dedicated method called by views.py to feed the ML pipeline."""
        content_items = []
        try:
            for item in self.user.upvoted(limit=limit):
                content_items.append(self._format_item(item, signal_type="upvoted"))
        except Exception as e:
            logger.error(f"Error fetching exclusively upvoted posts: {e}")
            
        return content_items