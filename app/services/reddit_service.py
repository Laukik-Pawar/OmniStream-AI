import os
import logging
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Map popular subreddits to your dashboard's master candidate labels.
REDDIT_GENRE_MAP = {
    "technology": "Technology", "programming": "Technology", "dataengineering": "Technology", "python": "Technology",
    "movies": "Entertainment", "television": "Entertainment", "bollywood": "Entertainment",
    "personalfinance": "Finance", "wallstreetbets": "Finance", "investing": "Finance",
    "gaming": "Gaming", "pcgaming": "Gaming", "boardgames": "Gaming",
    "funny": "Comedy", "jokes": "Comedy", "memes": "Comedy",
    "science": "Science", "askscience": "Education",
    "nba": "Sports", "soccer": "Sports", "cricket": "Sports"
}

class RedditService:
    def __init__(self, credentials: dict):
        """Initialize RedditService using Neon PostgreSQL as the primary data store."""
        self.db_url = os.environ.get('DATABASE_URL')
        
        # Map the exact keys your Flask session is currently sending
        self.user_id = credentials.get('db_user_id') or credentials.get('user_id')
        self.reddit_username = credentials.get('username') or credentials.get('reddit_username')
        self.refresh_token = credentials.get('refresh_token')

        # If user_id is not passed directly, look it up via username or token
        if not self.user_id and (self.reddit_username or self.refresh_token):
            self.user_id = self._resolve_user_id()

    def _get_connection(self):
        """Create a fresh connection to the Neon database."""
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is missing or empty.")
        return psycopg2.connect(self.db_url)

    def _resolve_user_id(self):
        """Resolve the Neon user ID using credentials stored in session."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if self.reddit_username:
                        cursor.execute("SELECT id FROM users WHERE reddit_username = %s LIMIT 1;", (self.reddit_username,))
                    else:
                        cursor.execute("SELECT id FROM users WHERE reddit_refresh_token = %s LIMIT 1;", (self.refresh_token,))
                    
                    row = cursor.fetchone()
                    return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to resolve user_id from database: {e}")
            return None

    def _format_db_record(self, row, signal_type="upvoted"):
        """Standardize database records into the unified template and ML dictionary schema."""
        raw_ts = row.get('interaction_timestamp')
        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                raw_ts = raw_ts.replace(tzinfo=timezone.utc)
            timestamp = raw_ts.isoformat().replace('+00:00', 'Z')
        else:
            timestamp = str(raw_ts)

        # EXTRACT NATIVE GENRE FROM SUBREDDIT
        raw_subreddit = row.get('subreddit') or ''
        # Clean the string in case it includes "r/" prefix
        clean_sub = raw_subreddit.lower().replace('r/', '') 
        
        official_genre = REDDIT_GENRE_MAP.get(clean_sub)
        
        # Fallback: Capitalize the subreddit name if it isn't in our hardcoded list
        if not official_genre and clean_sub:
            official_genre = clean_sub.capitalize()
        elif not official_genre:
            official_genre = "General"

        return {
            'source': 'reddit',
            'type': 'Submission',
            'title': row.get('title') or '',
            'content': row.get('content') or 'Link Post - No Description',
            'url': row.get('url') or '',
            'subreddit': raw_subreddit,
            'timestamp': timestamp,
            'signal': signal_type,
            'native_genre': official_genre  # Expose to ML pipeline
        }

    def fetch_upvoted_posts(self, limit=25):
        """Fetch ingested upvoted posts and join with known post-level genres."""
        if not self.user_id:
            logger.warning("fetch_upvoted_posts called without an identifiable user_id.")
            return []

        content_items = []
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # LEFT JOIN now checks if we have classified this exact post before
                    query = """
                        SELECT r.reddit_post_id, r.title, r.content, r.url, 
                               r.subreddit, r.interaction_timestamp, pg.genre as known_genre
                        FROM reddit_interactions r
                        LEFT JOIN post_genres pg ON r.url = pg.post_url
                        WHERE r.user_id = %s
                        ORDER BY r.interaction_timestamp DESC
                        LIMIT %s;
                    """
                    cursor.execute(query, (self.user_id, limit))
                    rows = cursor.fetchall()
                    for row in rows:
                        item = self._format_db_record(row, signal_type="upvoted")
                        item['native_genre'] = row.get('known_genre') 
                        content_items.append(item)
        except Exception as e:
            logger.error(f"Error fetching upvoted posts from Neon: {e}")

        return content_items
    
    def fetch_history(self, limit=10):
        """Fetch recent interactions from the database for the history view."""
        return self.fetch_upvoted_posts(limit=limit)