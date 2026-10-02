from flask import Blueprint, jsonify, session, request
from app.services.reddit_service import RedditService
from app.services.youtube_service import YouTubeService
from app.services.ml_service import MLService
from app.services.content_service import ContentService
import os
import psycopg2
from psycopg2.extras import RealDictCursor

dashboard_bp = Blueprint('dashboard', __name__)


def get_db_connection():
    """Helper to get a Postgres connection."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL is not set")
    return psycopg2.connect(db_url)


@dashboard_bp.route('/generate-recommendations', methods=['GET'])
def generate_recommendations():
    """
    On-demand endpoint: Fetch live history from Reddit/YouTube,
    run the ML pipeline, and return time-aware recommendations.
    """
    if 'reddit_credentials' not in session and 'youtube_credentials' not in session:
        return jsonify({'error': 'Not authenticated with Reddit or YouTube'}), 401

    try:
        history_data = []

        # 1. Reddit
        if 'reddit_credentials' in session:
            try:
                reddit_service = RedditService(session['reddit_credentials'])
                reddit_data = reddit_service.fetch_history(limit=15)
                if reddit_data:
                    history_data.extend(reddit_data)
            except Exception as e:
                print(f"Error fetching Reddit data: {e}")

        # 2. YouTube
        if 'youtube_credentials' in session:
            try:
                youtube_data = YouTubeService.fetch_liked_videos(
                    session['youtube_credentials'], limit=10
                )
                if youtube_data:
                    history_data.extend(youtube_data)
            except Exception as e:
                print(f"Error fetching YouTube data: {e}")

        if not history_data:
            return jsonify({
                'message': 'No history found across Reddit or YouTube to analyze.'
            }), 404

        # 3. Run ML pipeline
        categorized_data, time_keywords = MLService.categorize_content(history_data)

        if not time_keywords:
            return jsonify({
                'success': True,
                'data': {},
                'message': 'No clusters generated from history'
            })

        # 4. Build live recommendations from time-aware queries
        content_service = ContentService()
        recommendations = {}

        for time_block, query_list in time_keywords.items():
            block_results = []
            for q_obj in query_list:
                raw_results = content_service.scrape_content(
                    q_obj['query'], num_results=3
                )
                # Normalize keys (ContentService returns 'link')
                articles = []
                for item in raw_results:
                    articles.append({
                        'title': item.get('title'),
                        'url': item.get('link') or item.get('url'),
                        'snippet': item.get('snippet', '')
                    })
                block_results.append({
                    'genre': q_obj.get('genre', 'General'),
                    'query': q_obj['query'],
                    'articles': articles
                })
            recommendations[time_block] = block_results

        return jsonify({
            'success': True,
            'data': recommendations,
            'categorized_history': categorized_data
        })

    except Exception as e:
        print(f"Error in generate-recommendations route: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@dashboard_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    Hybrid infinite-scroll endpoint.
    First serves pre-computed recommendations from the database,
    then falls back to live Google CSE if the DB batch is exhausted.
    """
    query_genre = request.args.get('q', '').strip()
    offset = request.args.get('offset', 0, type=int)
    limit = 4  # Matches typical UI card layout

    if not query_genre:
        return jsonify([])

    results = []

    try:
        # 1. Try to serve from pre-computed recommendations table
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT title, url, snippet, source_domain, genre,
                           matched_query, image_url, time_block
                    FROM recommendations
                    WHERE genre ILIKE %s
                    ORDER BY id DESC
                    OFFSET %s LIMIT %s
                """, (query_genre, offset, limit))
                db_items = cursor.fetchall()
                results = [dict(item) for item in db_items]

        # 2. Fallback to live search if we didn't get a full page
        if len(results) < limit:
            needed = limit - len(results)
            content_service = ContentService()
            live_items = content_service.scrape_content(
                f"{query_genre} recommendations",
                num_results=needed
            )

            for item in live_items:
                results.append({
                    'title': item.get('title'),
                    'url': item.get('link') or item.get('url'),
                    'snippet': item.get('snippet', ''),
                    'source_domain': item.get('source_domain', 'web'),
                    'genre': query_genre,
                    'image_url': item.get('image_url', ''),
                    'matched_query': f"{query_genre} recommendations"
                })

        return jsonify(results)

    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        return jsonify([]), 500
    
@dashboard_bp.route('/ingest-reddit', methods=['POST'])
def ingest_reddit():
    if 'reddit_credentials' not in session:
        return jsonify({'error': 'Not authenticated with Reddit'}), 401

    try:
        service = RedditService(session['reddit_credentials'])
        count = service.ingest_upvoted_posts(limit=50)
        return jsonify({
            'success': True,
            'message': f'Ingested {count} new posts',
            'count': count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500