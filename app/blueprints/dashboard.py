
from flask import Blueprint, jsonify, session, request
from app.services.reddit_service import RedditService
# from app.services.youtube_service import YouTubeService
# from app.services.ml_service import MLService
from app.services.content_service import ContentService
# from app.services.recommendation_engine import RecommendationEngine
import os
import psycopg2
from psycopg2.extras import RealDictCursor
# from datetime import datetime
# from zoneinfo import ZoneInfo

dashboard_bp = Blueprint('dashboard', __name__)


def get_db_connection():
    """Helper to get a Postgres connection."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL is not set")
    return psycopg2.connect(db_url)






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


@dashboard_bp.route('/experimental/queries', methods=['GET'])
def experimental_queries():
    """Preview search queries without running the recommendation pipeline."""
    if 'reddit_credentials' not in session:
        return jsonify({
            'success': False,
            'error': 'Connect Reddit to test search queries.'
        }), 401

    try:
        reddit_service = RedditService(session['reddit_credentials'])
        history = reddit_service.fetch_history(limit=15) or []

        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
            ENGLISH_STOP_WORDS
        )

        documents = [
            f"{item.get('title', '')} {item.get('content', '')}".strip()
            for item in history
        ]

        documents = [doc for doc in documents if doc]

        if not documents:
            return jsonify({
                'success': True,
                'queries': [],
                'message': 'No usable Reddit history found.'
            })

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        ranked = sorted(
            zip(terms, scores),
            key=lambda pair: pair[1],
            reverse=True
        )

        keywords = [
            term for term, score in ranked[:10]
        ]

        queries = [
            {
                'keywords': keywords[i:i + 3],
                'query': ' '.join(keywords[i:i + 3])
            }
            for i in range(0, min(len(keywords), 6), 3)
        ]

        return jsonify({
            'success': True,
            'source_count': len(history),
            'queries': queries
        })

    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 400

    except Exception:
        import logging
        logging.exception('Experimental query preview failed')
        return jsonify({
            'success': False,
            'error': 'Unable to generate query preview.'
        }), 500
