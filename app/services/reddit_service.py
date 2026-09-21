import praw
from datetime import datetime
import logging

# Set up basic logging for the service
logger = logging.getLogger(__name__)

class RedditService:
    def __init__(self, credentials: dict):
        """Initialize the Reddit client with OAuth credentials."""
        try:
            self.reddit = praw.Reddit(
                client_id=credentials.get('client_id'),
                client_secret=credentials.get('client_secret'),
                user_agent=credentials.get('user_agent', 'omnistream-ai:v1.0'),
                username=credentials.get('username'),
                password=credentials.get('password')
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
        timestamp = datetime.utcfromtimestamp(item.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        
        # Differentiate between a Comment and a Submission (Post)
        is_comment = isinstance(item, praw.models.Comment)
        
        return {
            'source': 'Reddit',
            'type': 'Comment' if is_comment else 'Submission',
            'title': item.submission.title if is_comment else item.title,
            'content': item.body if is_comment else item.selftext,
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