import os
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Blueprint, render_template, current_app, request, jsonify, session, redirect, url_for
from app.services.ml_service import MLService
import pandas as pd

views_bp = Blueprint('views', __name__)

def get_db_connection():
    return psycopg2.connect(os.environ.get('DATABASE_URL'))

@views_bp.route('/')
def index():
    """Root page – redirect to dashboard if logged in, otherwise show home."""
    if session.get('user_identifier') or session.get('youtube_credentials') or session.get('reddit_credentials'):
        return redirect(url_for('views.dashboard'))
    return render_template('index.html')

@views_bp.route('/dashboard')
def dashboard():
    """Reads pre-calculated data for the currently logged-in user only."""
    
    # 1. Require a logged-in user
    user_identifier = session.get('user_identifier')
    if not user_identifier:
        # You can change this to whatever your login page is
        return redirect(url_for('views.index') if 'views.index' in current_app.view_functions else '/')

    # 2. Determine current time block
    current_hour = pd.Timestamp.now(tz='America/New_York').hour
    if 0 <= current_hour < 6:
        current_block = 'Night'
    elif 6 <= current_hour < 12:
        current_block = 'Morning'
    elif 12 <= current_hour < 18:
        current_block = 'Afternoon'
    else:
        current_block = 'Evening'

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:

                # 3. Temporal clusters (filtered by user)
                cursor.execute("""
                    SELECT DISTINCT ON (time_block, genre) 
                        time_block, genre, search_query as query 
                    FROM temporal_clusters
                    WHERE user_identifier = %s
                    ORDER BY time_block, genre, last_updated DESC
                """, (user_identifier,))
                cluster_rows = cursor.fetchall()

                time_keywords = {
                    'Night': [], 'Morning': [], 'Afternoon': [], 'Evening': []
                }
                for row in cluster_rows:
                    time_keywords[row['time_block']].append({
                        'genre': row['genre'],
                        'query': row['query']
                    })

                # 4. Item mappings (filtered by user + last 14 days)
                cursor.execute("""
                    SELECT url, source, title, content, timestamp, time_block, ml_genre as genre 
                    FROM item_mappings 
                    WHERE user_identifier = %s
                      AND timestamp >= NOW() - INTERVAL '14 days'
                    ORDER BY timestamp DESC
                """, (user_identifier,))
                item_rows = cursor.fetchall()

                categorized_data = {
                    'Night': [], 'Morning': [], 'Afternoon': [], 'Evening': []
                }
                for item in item_rows:
                    if item['timestamp']:
                        item['timestamp'] = item['timestamp'].strftime('%Y-%m-%d %H:%M')
                    if item['time_block'] in categorized_data:
                        categorized_data[item['time_block']].append(item)

                # 5. Recommendations for current time block (filtered by user)
                cursor.execute("""
                    SELECT title, url, snippet, source_domain, genre, 
                           matched_query, image_url, time_block
                    FROM recommendations
                    WHERE user_identifier = %s
                      AND LOWER(time_block) = LOWER(%s)
                """, (user_identifier, current_block))
                live_recommendations = cursor.fetchall()

                print(
                    f"DEBUG: User={user_identifier} | Block={current_block} | "
                    f"Recs={len(live_recommendations)}",
                    flush=True
                )

        ranked_genres = [item['genre'] for item in time_keywords.get(current_block, [])]

        return render_template(
            'dashboard.html',
            data=categorized_data,
            time_keywords=time_keywords,
            current_block=current_block,
            ranked_genres=ranked_genres,
            recommendations=live_recommendations,
            user_identifier=user_identifier          # optional, useful in template
        )

    except Exception as e:
        current_app.logger.error(f"Database error loading dashboard: {e}")
        print(f"DEBUG ERROR: {e}", flush=True)
        return render_template('dashboard.html', data=None)


@views_bp.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    """Infinite scroll – only returns recommendations for the current user."""
    user_identifier = session.get('user_identifier')
    if not user_identifier:
        return jsonify({'error': 'Not authenticated'}), 401

    query_genre = request.args.get('q', '').strip()
    offset = request.args.get('offset', 0, type=int)
    limit = 4

    if not query_genre:
        return jsonify([])

    results = []

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT title, url, snippet, source_domain, genre, 
                           matched_query, image_url, time_block
                    FROM recommendations
                    WHERE user_identifier = %s
                      AND genre ILIKE %s
                    ORDER BY id DESC
                    OFFSET %s LIMIT %s
                """, (user_identifier, query_genre, offset, limit))
                db_items = cursor.fetchall()
                results = [dict(item) for item in db_items]

        # Live fallback if not enough results
        if len(results) < limit:
            needed = limit - len(results)
            live_items = MLService.fetch_paginated_cse(
                query_genre, 
                start_index=offset + len(results), 
                num=needed
            )
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
        current_app.logger.error(f"Error fetching recommendations: {e}")
        return jsonify([]), 500 