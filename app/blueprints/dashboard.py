
from flask import Blueprint, jsonify, session, request
from app.services.reddit_service import RedditService
from app.services.youtube_service import YouTubeService
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
        reddit_history = reddit_service.fetch_history(limit=15) or []

        youtube_history = []

        if session.get('youtube_credentials'):
            try:
                youtube_history = YouTubeService.fetch_liked_videos(
                    session['youtube_credentials'],
                    limit=15
                ) or []
            except Exception:
                logging.exception('YouTube history fetch failed')

        history = reddit_history + youtube_history


        # Use each interaction as its own document.
        
        # STEP 11: Context-preserving query generation.
        # Use TF-IDF to identify useful interactions, but
        # preserve title phrases when constructing queries.

        import re
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
            ENGLISH_STOP_WORDS
        )

        def clean_title(title):
            title = str(title or "")

            # Convert hashtags into searchable words.
            title = re.sub(r"#(?=\w)", " ", title)

            # Remove URLs, emojis and unnecessary punctuation.
            title = re.sub(r"https?://\S+", " ", title)
            title = re.sub(r"[^\w\s'-]", " ", title)

            # Remove common conversational filler.
            filler = {
                "guys", "please", "hey", "omg", "lol",
                "official", "viral", "new", "latest"
            }

            words = [
                word for word in title.split()
                if word.lower() not in filler
            ]

            return " ".join(words).strip()

        # Build one candidate per interaction.
        candidates = []

        for item in history:
            title = clean_title(item.get("title"))
            source = item.get("source", "unknown")

            if not title:
                continue

            words = title.split()

            # Reject extremely short or generic titles.
            if len(words) < 2:
                continue

            # Keep the title's word order and context.
            query = " ".join(words[:10])

            candidates.append({
                "query": query,
                "keywords": words[:10],
                "source": source,
                "title": item.get("title", "")
            })

        # Remove exact duplicate queries.
        unique_candidates = []
        seen = set()

        for item in candidates:
            normalized = item["query"].lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_candidates.append(item)

        # Use TF-IDF to score candidate titles.
        if unique_candidates:
            documents = [
                item["query"]
                for item in unique_candidates
            ]

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            try:
                matrix = vectorizer.fit_transform(documents)

                for index, item in enumerate(unique_candidates):
                    row = matrix.getrow(index)
                    item["score"] = float(row.sum())

            except ValueError:
                for item in unique_candidates:
                    item["score"] = 0.0

        # Prefer informative titles over generic sentences.
        generic_words = {
            "what", "when", "where", "why", "how",
            "this", "that", "these", "those",
            "anyone", "everyone", "something"
        }

        for item in unique_candidates:
            words = item["query"].lower().split()

            meaningful = [
                word for word in words
                if word not in ENGLISH_STOP_WORDS
                and word not in generic_words
                and len(word) > 2
            ]

            item["score"] += min(len(meaningful), 6) * 0.1

        unique_candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Select candidates from both platforms.
        reddit_candidates = [
            item for item in unique_candidates
            if item["source"].lower() == "reddit"
        ]

        youtube_candidates = [
            item for item in unique_candidates
            if item["source"].lower() == "youtube"
        ]

        other_candidates = [
            item for item in unique_candidates
            if item["source"].lower() not in ("reddit", "youtube")
        ]

        selected = []
        selected_queries = set()

        # Alternate sources when both are available.
        while len(selected) < 5:
            added = False

            for pool in (
                reddit_candidates,
                youtube_candidates,
                other_candidates
            ):
                if not pool or len(selected) >= 5:
                    continue

                item = pool.pop(0)
                key = item["query"].lower()

                if key not in selected_queries:
                    selected.append(item)
                    selected_queries.add(key)
                    added = True

            if not added:
                break

        queries = [
            {
                "query": item["query"],
                "keywords": item["keywords"],
                "source": item["source"],
                "original_title": item["title"]
            }
            for item in selected
        ]


        
        return jsonify({
            'success': True,
            'source_count': len(history),
            'reddit_count': len(reddit_history),
            'youtube_count': len(youtube_history),
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
