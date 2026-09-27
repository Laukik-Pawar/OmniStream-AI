from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

# Map YouTube's native numeric category IDs to your dashboard's candidate labels
YOUTUBE_CATEGORY_MAP = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", 
    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", 
    "20": "Gaming", "22": "Lifestyle", "23": "Comedy", 
    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", 
    "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism"
}

class YouTubeService:
    @staticmethod
    def fetch_liked_videos(session_credentials, limit=25):
        """Fetches the user's liked videos and their official YouTube categories."""
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
            playlist_request = youtube.playlistItems().list(
                part="snippet",
                playlistId=likes_playlist_id,
                maxResults=limit
            )
            playlist_response = playlist_request.execute()
            
            playlist_items = playlist_response.get('items', [])
            if not playlist_items:
                return []

            # 3. We must fetch the actual Video objects to get the categoryId
            video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_items]
            
            videos_request = youtube.videos().list(
                part="snippet",
                id=",".join(video_ids)
            )
            videos_response = videos_request.execute()
            
            # Create a dictionary to quickly look up category by video ID
            video_categories = {}
            for video in videos_response.get('items', []):
                vid_id = video['id']
                cat_id = video['snippet'].get('categoryId')
                video_categories[vid_id] = YOUTUBE_CATEGORY_MAP.get(cat_id, "General")

            # 4. Build the final content items with the native genre included
            for item in playlist_items:
                snippet = item['snippet']
                vid_id = snippet['resourceId']['videoId']
                
                content_items.append({
                    'source': 'youtube',
                    'type': 'Video',
                    'title': snippet.get('title'),
                    'content': snippet.get('description', 'No Description'),
                    'url': f"https://youtube.com/watch?v={vid_id}",
                    'channel': snippet.get('videoOwnerChannelTitle', 'Unknown Channel'),
                    'timestamp': snippet.get('publishedAt'),
                    'signal': 'liked',
                    'native_genre': video_categories.get(vid_id, "General")
                })

        except Exception as e:
            logger.error(f"Error fetching YouTube history: {e}")
            raise
            
        return content_items

# The commented-out code blocks at the bottom of your file can remain commented out.