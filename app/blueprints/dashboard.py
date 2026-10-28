
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
    """Generate lightweight, interaction-based search queries."""
    if 'reddit_credentials' not in session:
        return jsonify({
            'success': False,
            'error': 'Connect Reddit to test search queries.'
        }), 401

    try:
        import re
        import logging
        from sklearn.feature_extraction.text import TfidfVectorizer

        reddit_service = RedditService(session['reddit_credentials'])
        history = reddit_service.fetch_history(limit=15) or []

        # Use each interaction as its own document.
        documents = []

        for item in history:
            title = str(item.get('title') or '').strip()
            content = str(item.get('content') or '').strip()

            # Titles are often more focused than full post bodies.
            text = f"{title} {title} {content[:500]}".strip()

            if text:
                documents.append(text)

        if not documents:
            return jsonify({
                'success': True,
                'source_count': len(history),
                'queries': [],
                'message': 'No usable Reddit history found.'
            })

        # Ignore common Reddit / conversational filler.
        custom_stopwords = {
            'reddit', 'post', 'comment', 'comments',
            'subreddit', 'deleted', 'removed',
            'https', 'http', 'www',
            'hope', 'think', 'really', 'just',
            'like', 'know', 'want', 'got',
            'did', 'does', 'doing', 'thing',
            'things', 'people', 'today'
        }

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9]+\b',
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(documents)
        terms = vectorizer.get_feature_names_out()

        candidates = []

        for doc_index, document in enumerate(documents):
            row = matrix.getrow(doc_index)

            ranked_terms = sorted(
                zip(row.indices, row.data),
                key=lambda pair: pair[1],
                reverse=True
            )

            selected = []
            used_words = set()

            for term_index, score in ranked_terms:
                term = terms[term_index]
                words = term.lower().split()

                # Reject generic or noisy phrases.
                if any(word in custom_stopwords for word in words):
                    continue

                # Avoid repeated words across selected phrases.
                if any(word in used_words for word in words):
                    continue

                selected.append(term)
                used_words.update(words)

                if len(selected) == 3:
                    break

            if len(selected) < 2:
                continue

            query = ' '.join(selected)

            candidates.append({
                'keywords': selected,
                'query': query,
                'score': float(sum(
                    matrix[doc_index, vectorizer.vocabulary_[term]]
                    for term in selected
                ))
            })

        # Prefer stronger candidates while avoiding duplicates.
        candidates.sort(
            key=lambda item: item['score'],
            reverse=True
        )

        queries = []
        seen = set()

        for candidate in candidates:
            normalized = re.sub(
                r'\s+', ' ', candidate['query'].lower()
            ).strip()

            if normalized in seen:
                continue

            seen.add(normalized)

            queries.append({
                'keywords': candidate['keywords'],
                'query': candidate['query']
            })

            if len(queries) == 5:
                break

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
        logging.exception('Experimental query preview failed')
        return jsonify({
            'success': False,
            'error': 'Unable to generate query preview.'
        }), 500
