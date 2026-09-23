from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

class YouTubeService:
    @staticmethod
    def fetch_liked_videos(session_credentials, limit=25):
        """Fetches the user's liked videos with the timestamp of the actual 'like'."""
        content_items = []
        try:
            # Reconstruct Google credentials from the session dictionary
            creds = Credentials(
                token=session_credentials['token'],
                refresh_token=session_credentials.get('refresh_token'),
                token_uri=session_credentials.get('token_uri'),
                client_id=session_credentials.get('client_id'),
                client_secret=session_credentials.get('client_secret'),
                scopes=session_credentials.get('scopes')
            )
            
            youtube = build('youtube', 'v3', credentials=creds)

            # 1. Fetch the user's channel details to find the exact ID of their "Likes" playlist
            channel_request = youtube.channels().list(
                part="contentDetails",
                mine=True
            )
            channel_response = channel_request.execute()
            
            if not channel_response.get("items"):
                raise ValueError("Could not locate YouTube channel for authenticated user.")
                
            likes_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["likes"]

            # 2. Fetch the items from the "Likes" playlist
            # The 'snippet.publishedAt' on a PlaylistItem is the date it was added to the playlist!
            playlist_request = youtube.playlistItems().list(
                part="snippet",
                playlistId=likes_playlist_id,
                maxResults=limit
            )
            playlist_response = playlist_request.execute()

            for item in playlist_response.get('items', []):
                snippet = item['snippet']
                content_items.append({
                    'source': 'youtube', # Lowercase for Jinja badge styling
                    'type': 'Video',
                    'title': snippet.get('title'),
                    'content': snippet.get('description', 'No Description'),
                    'url': f"https://youtube.com/watch?v={snippet['resourceId']['videoId']}",
                    'channel': snippet.get('videoOwnerChannelTitle', 'Unknown Channel'),
                    # This is now the exact time the user hit the Like button
                    'timestamp': snippet.get('publishedAt'), 
                    'signal': 'liked'
                })

        except Exception as e:
            logger.error(f"Error fetching YouTube history: {e}")
            raise
            
        return content_items

# from googleapiclient.discovery import build
# import logging

# logger = logging.getLogger(__name__)

# class YouTubeService:
#     def __init__(self, credentials):
#         """Initialize the YouTube API client with OAuth credentials."""
#         try:
#             self.youtube = build('youtube', 'v3', credentials=credentials)
#         except Exception as e:
#             logger.error(f"Failed to initialize YouTubeService: {e}")
#             raise

#     def fetch_liked_videos(self, max_results=10):
#         """Fetch videos the authenticated user has liked."""
#         liked_videos = []
#         try:
#             request = self.youtube.videos().list(
#                 part="snippet",
#                 myRating="like",
#                 maxResults=max_results
#             )
#             response = request.execute()
            
#             for item in response.get('items', []):
#                 liked_videos.append({
#                     'source': 'YouTube',
#                     'type': 'Video',
#                     'title': item['snippet']['title'],
#                     'content': item['snippet'].get('description', '')[:200], # truncate long descriptions
#                     'url': f"https://www.youtube.com/watch?v={item['id']}",
#                     'channel': item['snippet']['channelTitle'],
#                     'timestamp': item['snippet']['publishedAt'],
#                     'signal': 'liked'
#                 })
#         except Exception as e:
#             logger.error(f"Error fetching YouTube liked videos: {e}")
            
#         return liked_videos

#     def fetch_playlists(self, max_playlists=5, max_items_per_playlist=5):
#         """Fetch videos from the user's created playlists."""
#         playlist_data = []
#         try:
#             playlists = self.youtube.playlists().list(
#                 part="snippet",
#                 mine=True,
#                 maxResults=max_playlists
#             ).execute()

#             for playlist in playlists.get("items", []):
#                 items = self.youtube.playlistItems().list(
#                     part="snippet",
#                     playlistId=playlist["id"],
#                     maxResults=max_items_per_playlist
#                 ).execute()

#                 for video in items.get("items", []):
#                     playlist_data.append({
#                         'source': 'YouTube',
#                         'type': 'Playlist_Video',
#                         'title': video['snippet']['title'],
#                         'content': '',
#                         'url': f"https://www.youtube.com/watch?v={video['snippet']['resourceId'].get('videoId', '')}",
#                         'channel': video['snippet']['videoOwnerChannelTitle'] if 'videoOwnerChannelTitle' in video['snippet'] else '',
#                         'timestamp': video['snippet']['publishedAt'],
#                         'signal': 'playlist'
#                     })
#         except Exception as e:
#             logger.error(f"Error fetching YouTube playlists: {e}")

#         return playlist_data

#----------------------------------------------------------------------------------------------------------------



# from googleapiclient.discovery import build
# import logging

# logger = logging.getLogger(__name__)

# class YouTubeService:
#     def __init__(self, api_key):
#         """Initialize the YouTube API client with a simple API key."""
#         try:
#             # CHANGED: Use developerKey instead of OAuth credentials
#             self.youtube = build('youtube', 'v3', developerKey=api_key)
#         except Exception as e:
#             logger.error(f"Failed to initialize YouTubeService: {e}")
#             raise

#     def fetch_liked_videos(self, max_results=10):
#         """
#         TEMPORARY TESTING OVERRIDE: 
#         Since an API key cannot fetch private 'liked' videos, we will fetch 
#         public popular videos just to feed text into the ML pipeline.
#         """
#         liked_videos = []
#         try:
#             # CHANGED: Fetch public mostPopular videos instead of myRating="like"
#             request = self.youtube.videos().list(
#                 part="snippet",
#                 chart="mostPopular",
#                 regionCode="US",
#                 maxResults=max_results
#             )
#             response = request.execute()
            
#             for item in response.get('items', []):
#                 liked_videos.append({
#                     'source': 'YouTube',
#                     'type': 'Video',
#                     'title': item['snippet']['title'],
#                     'content': item['snippet'].get('description', '')[:200], # truncate long descriptions
#                     'url': f"https://www.youtube.com/watch?v={item['id']}",
#                     'channel': item['snippet']['channelTitle'],
#                     'timestamp': item['snippet']['publishedAt'],
#                     'signal': 'liked' # keeping the same schema for your pipeline
#                 })
#         except Exception as e:
#             logger.error(f"Error fetching YouTube liked videos: {e}")
            
#         return liked_videos

#     def fetch_playlists(self, max_playlists=5, max_items_per_playlist=5):
#         """TEMPORARY TESTING OVERRIDE: Return empty list to avoid OAuth crashes."""
#         return []