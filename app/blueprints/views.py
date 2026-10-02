# from flask import Blueprint, render_template, session, flash, redirect, url_for, request, jsonify
# from app.services.youtube_service import YouTubeService
# from app.services.reddit_service import RedditService
# from app.services.ml_service import MLService
# from datetime import datetime
# import pytz
# from app.services.recommendation_service import RecommendationService

# views_bp = Blueprint('views', __name__)

# @views_bp.route('/dashboard')
# def dashboard():
#     """Aggregates authenticated user data and passes it through the ML pipeline."""
#     content_items = []

#     # 1. Fetch Reddit Data
#     if 'reddit_credentials' in session:
#         try:
#             print("DEBUG REDDIT CREDENTIALS:", session['reddit_credentials'], flush=True)
#             reddit = RedditService(session['reddit_credentials'])
#             content_items.extend(reddit.fetch_upvoted_posts())
#         except Exception as e:
#             flash(f"Failed to load Reddit data: {str(e)}", "warning")
#             session.pop('reddit_credentials', None)

#     # 2. Fetch YouTube Data
#     if 'youtube_credentials' in session:
#         try:
#             yt_videos = YouTubeService.fetch_liked_videos(session['youtube_credentials'])
#             content_items.extend(yt_videos)
#         except Exception:
#             flash("YouTube session expired. Please log in again.", "danger")
#             session.pop('youtube_credentials', None)
            
#     # 3. Process the merged datasets through the ML pipelines
#     time_keywords = {}
#     if content_items:
#         categorized_data, time_keywords = MLService.categorize_content(content_items)
#     else:
#         categorized_data = {}
#         flash("No content available. Please connect your accounts.", "info")

#     # 4. Determine Current Local Time Block (using America/New_York timezone)
#     local_tz = pytz.timezone('America/New_York')
#     current_hour = datetime.now(local_tz).hour

#     if 0 <= current_hour < 6:
#         current_time_block = 'Night'
#     elif 6 <= current_hour < 12:
#         current_time_block = 'Morning'
#     elif 12 <= current_hour < 18:
#         current_time_block = 'Afternoon'
#     else:
#         current_time_block = 'Evening'

#     # 5. Calculate the "Most Viewed Genre" Ranking for this specific time block
#     genre_counts = {}
#     ranked_genres = []
    
#     if categorized_data and current_time_block in categorized_data:
#         # Tally up how many items belong to each genre during this time of day
#         for item in categorized_data[current_time_block]:
#             g = item.get('genre', 'General')
#             genre_counts[g] = genre_counts.get(g, 0) + 1
        
#         # Sort genres mathematically from highest view count to lowest
#         ranked_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)

#     # 6. Fetch live recommendations for the current time block
#     active_queries = time_keywords.get(current_time_block, []) if time_keywords else []
#     live_recommendations = RecommendationService.fetch_google_cse_results(active_queries)

#     # 7. Render the frontend template with the new variables
#     return render_template(
#         'dashboard.html', 
#         data=categorized_data, 
#         time_keywords=time_keywords,
#         recommendations=live_recommendations,
#         current_block=current_time_block,
#         ranked_genres=ranked_genres
#     )

# @views_bp.route('/api/recommendations')
# def api_recommendations():
#     """Async endpoint to fetch live CSE data for infinite scrolling."""
#     query = request.args.get('q', '')
#     offset = request.args.get('offset', 1, type=int)
    
#     # Enforce maximum constraints (50 results total, 5 per page)
#     if offset > 46:
#         return jsonify([])
        
#     results = MLService.fetch_paginated_cse(query, start_index=offset, num=5)
#     return jsonify(results)

# import os
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from flask import Blueprint, render_template, current_app, request, jsonify
# from app.services.ml_service import MLService
# import pandas as pd

# views_bp = Blueprint('views', __name__)

# def get_db_connection():
#     return psycopg2.connect(os.environ.get('DATABASE_URL'))

# @views_bp.route('/dashboard')
# def dashboard():
#     """Reads pre-calculated ML clusters, interactions, and recommendations directly from PostgreSQL."""
#     # 1. Determine the active time block
# # 1. Determine the active time block correctly covering all 24 hours
#     current_hour = pd.Timestamp.now(tz='America/New_York').hour
#     if 0 <= current_hour < 6:
#         current_block = 'Night'
#     elif 6 <= current_hour < 12:
#         current_block = 'Morning'
#     elif 12 <= current_hour < 18:
#         current_block = 'Afternoon'
#     else:
#         current_block = 'Evening' # Covers 18:00 (6 PM) through 23:59 (11:59 PM)

#     try:
#         with get_db_connection() as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
#                 # 2. Fetch the pre-calculated cluster keywords (Latest only)
#                 cursor.execute("""
#                     SELECT DISTINCT ON (time_block, genre) 
#                         time_block, genre, search_query as query 
#                     FROM temporal_clusters
#                     ORDER BY time_block, genre, last_updated DESC
#                 """)
#                 cluster_rows = cursor.fetchall()
                
#                 time_keywords = { 'Night': [], 'Morning': [], 'Afternoon': [], 'Evening': [] }
#                 for row in cluster_rows:
#                     time_keywords[row['time_block']].append({
#                         'genre': row['genre'],
#                         'query': row['query']
#                     })

#                 # 3. Fetch the cached items (Strict 14-day lookback for performance)
#                 cursor.execute("""
#                     SELECT url, source, title, content, timestamp, time_block, ml_genre as genre 
#                     FROM item_mappings 
#                     WHERE timestamp >= NOW() - INTERVAL '14 days'
#                     ORDER BY timestamp DESC
#                 """)
#                 item_rows = cursor.fetchall()
                
#                 categorized_data = { 'Night': [], 'Morning': [], 'Afternoon': [], 'Evening': [] }
#                 for item in item_rows:
#                     if item['timestamp']:
#                         item['timestamp'] = item['timestamp'].strftime('%Y-%m-%d %H:%M')
#                     if item['time_block'] in categorized_data:
#                         categorized_data[item['time_block']].append(item)

#                 # 4. Fetch pre-calculated recommendations for the current time block
#                 cursor.execute("""
#                     SELECT title, url, snippet, source_domain, genre, matched_query, image_url, time_block
#                     FROM recommendations
#                     WHERE genre = %s
#                 """, (current_block,))
#                 live_recommendations = cursor.fetchall()

#         # 5. Determine the sorted genres for the current active block
#         ranked_genres = [item['genre'] for item in time_keywords.get(current_block, [])]

#         return render_template(
#             'dashboard.html',
#             data=categorized_data,
#             time_keywords=time_keywords,
#             current_block=current_block,
#             ranked_genres=ranked_genres,
#             recommendations=live_recommendations # <--- Pass the fetched recommendations here!
#         )

#     except Exception as e:
#         current_app.logger.error(f"Database error loading dashboard: {e}")
#         return render_template('dashboard.html', data=None)


# @views_bp.route('/api/recommendations', methods=['GET'])
# def api_recommendations():
#     """Hybrid endpoint: Serves stored DB items for initial loads using offset, 
#        then falls back to live CSE fetching if the database batch runs out."""
#     query_genre = request.args.get('q', '')
#     offset = request.args.get('offset', 0, type=int)
#     limit = 4  # Batch size matching your UI layout

#     results = []

#     try:
#         # 1. Query your Neon PostgreSQL database for stored records matching the genre/offset
#         with get_db_connection() as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#                 cursor.execute("""
#                     SELECT title, url, snippet, source_domain, genre, matched_query, image_url, time_block
#                     FROM recommendations
#                     WHERE genre = %s
#                     OFFSET %s LIMIT %s
#                 """, (query_genre, offset, limit))
#                 db_items = cursor.fetchall()
#                 results = [dict(item) for item in db_items]

#         # 2. Real-time fallback: If database batch runs out of items for this offset, fetch live content
#         if len(results) < limit:
#             needed = limit - len(results)
#             # Use your MLService paginated fetch for live fallback
#             live_items = MLService.fetch_paginated_cse(query_genre, start_index=offset + len(results), num=needed)
            
#             for article in live_items:
#                 results.append({
#                     "title": article.get("title"),
#                     "url": article.get("url") or article.get("link"),
#                     "snippet": article.get("snippet"),
#                     "image_url": article.get("image_url"),
#                     "source_domain": article.get("source_domain", "web"),
#                     "genre": query_genre
#                 })

#         return jsonify(results)

#     except Exception as e:
#         current_app.logger.error(f"Error fetching infinite scroll recommendations: {e}")
#         return jsonify([]), 500
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Blueprint, render_template, current_app, request, jsonify
from app.services.ml_service import MLService
import pandas as pd

views_bp = Blueprint('views', __name__)

def get_db_connection():
    return psycopg2.connect(os.environ.get('DATABASE_URL'))

@views_bp.route('/dashboard')
def dashboard():
    """Reads pre-calculated ML clusters, interactions, and recommendations directly from PostgreSQL."""
    # Determine the active time block correctly covering all 24 hours
    current_hour = pd.Timestamp.now(tz='America/New_York').hour
    if 0 <= current_hour < 6:
        current_block = 'Night'
    elif 6 <= current_hour < 12:
        current_block = 'Morning'
    elif 12 <= current_hour < 18:
        current_block = 'Afternoon'
    else:
        current_block = 'Evening' # Covers 18:00 (6 PM) through 23:59 (11:59 PM)

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                # 1. Fetch the pre-calculated cluster keywords (Latest only)
                cursor.execute("""
                    SELECT DISTINCT ON (time_block, genre) 
                        time_block, genre, search_query as query 
                    FROM temporal_clusters
                    ORDER BY time_block, genre, last_updated DESC
                """)
                cluster_rows = cursor.fetchall()
                
                time_keywords = { 'Night': [], 'Morning': [], 'Afternoon': [], 'Evening': [] }
                for row in cluster_rows:
                    time_keywords[row['time_block']].append({
                        'genre': row['genre'],
                        'query': row['query']
                    })

                # 2. Fetch the cached items (Strict 14-day lookback for performance)
                cursor.execute("""
                    SELECT url, source, title, content, timestamp, time_block, ml_genre as genre 
                    FROM item_mappings 
                    WHERE timestamp >= NOW() - INTERVAL '14 days'
                    ORDER BY timestamp DESC
                """)
                item_rows = cursor.fetchall()
                
                categorized_data = { 'Night': [], 'Morning': [], 'Afternoon': [], 'Evening': [] }
                for item in item_rows:
                    if item['timestamp']:
                        item['timestamp'] = item['timestamp'].strftime('%Y-%m-%d %H:%M')
                    if item['time_block'] in categorized_data:
                        categorized_data[item['time_block']].append(item)

                # 3. Fetch pre-calculated recommendations case-insensitively for the current time block
                cursor.execute("""
                    SELECT title, url, snippet, source_domain, genre, matched_query, image_url, time_block
                    FROM recommendations
                    WHERE LOWER(time_block) = LOWER(%s)
                """, (current_block,))
                live_recommendations = cursor.fetchall()
                
                print(f"DEBUG: Active Block = {current_block} | Fetched Recommendations Count = {len(live_recommendations)}", flush=True)

        # 4. Determine the sorted genres for the current active block
        ranked_genres = [item['genre'] for item in time_keywords.get(current_block, [])]

        return render_template(
            'dashboard.html',
            data=categorized_data,
            time_keywords=time_keywords,
            current_block=current_block,
            ranked_genres=ranked_genres,
            recommendations=live_recommendations 
        )

    except Exception as e:
        current_app.logger.error(f"Database error loading dashboard: {e}")
        print(f"DEBUG ERROR: {e}", flush=True)
        return render_template('dashboard.html', data=None)


@views_bp.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    """Hybrid endpoint: Serves stored DB items for initial loads using offset, 
       then falls back to live CSE fetching if the database batch runs out."""
    query_genre = request.args.get('q', '')
    offset = request.args.get('offset', 0, type=int)
    limit = 4  # Batch size matching your UI layout

    results = []

    try:
        # 1. Query your Neon PostgreSQL database for stored records matching the genre case-insensitively
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT title, url, snippet, source_domain, genre, matched_query, image_url, time_block
                    FROM recommendations
                    WHERE genre ILIKE %s
                    OFFSET %s LIMIT %s
                """, (query_genre, offset, limit))
                db_items = cursor.fetchall()
                results = [dict(item) for item in db_items]

        # 2. Real-time fallback: If database batch runs out of items for this offset, fetch live content
        if len(results) < limit:
            needed = limit - len(results)
            live_items = MLService.fetch_paginated_cse(query_genre, start_index=offset + len(results), num=needed)
            
            for article in live_items:
                results.append({
                    "title": article.get("title"),
                    "url": article.get("url") or article.get("link"),
                    "snippet": article.get("snippet"),
                    "image_url": article.get("image_url"),
                    "source_domain": article.get("source_domain", "web"),
                    "genre": query_genre
                })

        return jsonify(results)

    except Exception as e:
        current_app.logger.error(f"Error fetching infinite scroll recommendations: {e}")
        return jsonify([]), 500