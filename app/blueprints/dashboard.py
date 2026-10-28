
from flask import Blueprint, jsonify, session, request
from app.services.reddit_service import RedditService
from app.services.youtube_service import YouTubeService
from app.services.ml_service import MLService
from app.services.content_service import ContentService
from app.services.recommendation_engine import RecommendationEngine
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from zoneinfo import ZoneInfo

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
    Fetch Reddit/YouTube history, discover relevant articles and videos,
    rank candidates by user interests, and return recommendations.
    """
    if (
        'reddit_credentials' not in session
        and 'youtube_credentials' not in session
    ):
        return jsonify({
            'error': 'Not authenticated with Reddit or YouTube'
        }), 401

    try:
        history_data = []

        # 1. Fetch Reddit history.
        if 'reddit_credentials' in session:
            try:
                reddit_service = RedditService(
                    session['reddit_credentials']
                )
                reddit_data = reddit_service.fetch_history(limit=15)

                if reddit_data:
                    history_data.extend(reddit_data)

            except Exception as e:
                print(f"Error fetching Reddit history: {e}")

        # 2. Fetch liked YouTube videos as interest signals.
        if 'youtube_credentials' in session:
            try:
                youtube_history = YouTubeService.fetch_liked_videos(
                    session['youtube_credentials'],
                    limit=10
                )

                if youtube_history:
                    history_data.extend(youtube_history)

            except Exception as e:
                print(f"Error fetching YouTube history: {e}")

        if not history_data:
            return jsonify({
                'message': (
                    'No history found across Reddit or YouTube '
                    'to analyze.'
                )
            }), 404

        # 3. Categorize history and generate time-aware search queries.
        categorized_data, time_keywords = (
            MLService.categorize_content(history_data)
        )

        if not time_keywords:
            return jsonify({
                'success': True,
                'data': {},
                'categorized_history': categorized_data,
                'message': 'No clusters generated from history'
            })

        content_service = ContentService()
        recommendations = {}

        # 4. Discover and rank articles and videos for each query.
        
        current_hour = datetime.now(
            ZoneInfo("America/New_York")
        ).hour

        if current_hour < 6:
            current_block = "Night"
        elif current_hour < 12:
            current_block = "Morning"
        elif current_hour < 18:
            current_block = "Afternoon"
        else:
            current_block = "Evening"

        selected_queries = time_keywords.get(
            current_block, []
        )[:2]

        for time_block, query_list in [
            (current_block, selected_queries)
        ]:

            block_results = []

            for q_obj in query_list:
                query = q_obj.get('query', '').strip()
                genre = q_obj.get('genre', 'General')

                if not query:
                    continue

                # A. Discover web articles.
                articles = []

                try:
                    raw_articles = content_service.scrape_content(
                        query,
                        num_results=5
                    )

                    for item in raw_articles:
                        url = item.get('url') or item.get('link')

                        if not url:
                            continue

                        articles.append({
                            'title': item.get('title') or 'Untitled',
                            'url': url,
                            'snippet': item.get('snippet', ''),
                            'source_domain': item.get(
                                'source_domain', 'Web'
                            ),
                            'image_url': item.get('image_url', ''),
                            'content_type': 'article',
                            'query_genre': genre
                        })

                except Exception as e:
                    print(
                        f"Article discovery failed for '{query}': {e}"
                    )

                # B. Discover new YouTube videos, if connected.
                videos = []

                if 'youtube_credentials' in session:
                    try:
                        videos = YouTubeService.search_videos(
                            query,
                            session['youtube_credentials'],
                            limit=5
                        )

                        # This is a search hint, not a verified genre.
                        for video in videos:
                            video['query_genre'] = genre

                    except Exception as e:
                        print(
                            f"Video discovery failed for '{query}': {e}"
                        )

                # C. Personalize both content types using the same history.
                ranked_articles = RecommendationEngine.rank_candidates(
                    history_data,
                    articles,
                    limit=3
                )

                ranked_videos = RecommendationEngine.rank_candidates(
                    history_data,
                    videos,
                    limit=3
                )

                
                block_results.append({
                    'genre': genre,
                    'query': query,
                    'keywords': q_obj.get('keywords', []),
                    'cluster_id': q_obj.get('cluster_id'),
                    'query_strategy': q_obj.get(
                        'query_strategy', 'unknown'
                    ),
                    'articles': ranked_articles,
                    'videos': ranked_videos
                })


            recommendations[time_block] = block_results

        return jsonify({
            'success': True,
            'data': recommendations,
            'categorized_history': categorized_data
        })

    except Exception as e:
        print(f"Error in generate-recommendations route: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@dashboard_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    Hybrid infinite-scroll endpoint.

    Serves pre-computed recommendations from the database first,
    then uses live Google Custom Search when more results are needed.
    """
    query_genre = request.args.get('q', '').strip()
    offset = request.args.get('offset', 0, type=int)
    limit = 4

    if not query_genre:
        return jsonify([])

    results = []

    try:
        # 1. Retrieve pre-computed recommendations.
        with get_db_connection() as conn:
            with conn.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:
                cursor.execute(
                    """
                    SELECT
                        title,
                        url,
                        snippet,
                        source_domain,
                        genre,
                        matched_query,
                        image_url,
                        time_block
                    FROM recommendations
                    WHERE genre ILIKE %s
                    ORDER BY id DESC
                    OFFSET %s LIMIT %s
                    """,
                    (query_genre, offset, limit)
                )

                db_items = cursor.fetchall()
                results = [dict(item) for item in db_items]

        # 2. Use live search if the database has insufficient results.
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
                    'source_domain': item.get(
                        'source_domain', 'web'
                    ),
                    'genre': query_genre,
                    'image_url': item.get('image_url', ''),
                    'matched_query': (
                        f"{query_genre} recommendations"
                    )
                })

        return jsonify(results)

    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        return jsonify([]), 500


@dashboard_bp.route('/ingest-reddit', methods=['POST'])
def ingest_reddit():
    """Ingest the user's upvoted Reddit posts."""
    if 'reddit_credentials' not in session:
        return jsonify({
            'error': 'Not authenticated with Reddit'
        }), 401

    try:
        service = RedditService(session['reddit_credentials'])
        count = service.ingest_upvoted_posts(limit=50)

        return jsonify({
            'success': True,
            'message': f'Ingested {count} new posts',
            'count': count
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
