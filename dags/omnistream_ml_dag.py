import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Assuming your Flask app is in the PYTHONPATH for Airflow
from app.services.youtube_service import YouTubeService
from app.services.reddit_service import RedditService
from app.services.ml_service import MLService
from app.services.recommendation_service import RecommendationService

def execute_ml_pipeline():
    """Process every authenticated user independently."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL is missing")

    with psycopg2.connect(db_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT user_identifier, youtube_token, reddit_token
                FROM user_oauth_tokens
                WHERE reddit_token IS NOT NULL OR youtube_token IS NOT NULL
            """)
            active_users = cursor.fetchall()

    if not active_users:
        print("No active users found. Exiting.")
        return

    for user in active_users:
        user_id = user['user_identifier']
        print(f"\n=== Processing user: {user_id} ===")

        content_items = []

        # ---------- Reddit ----------
        if user.get('reddit_token'):
            try:
                reddit = RedditService(user['reddit_token'])
                ingested = reddit.ingest_upvoted_posts(limit=50)
                print(f"[{user_id}] Ingested {ingested} new Reddit posts")
                content_items.extend(reddit.fetch_upvoted_posts(limit=50))
            except Exception as e:
                print(f"[{user_id}] Reddit error: {e}")

        # ---------- YouTube ----------
        if user.get('youtube_token'):
            try:
                yt_videos = YouTubeService.fetch_liked_videos(
                    user['youtube_token'], limit=25
                )
                content_items.extend(yt_videos)
            except Exception as e:
                print(f"[{user_id}] YouTube error: {e}")

        if not content_items:
            print(f"[{user_id}] No content. Skipping.")
            continue

        # ---------- ML ----------
        categorized_data, time_keywords = MLService.categorize_content(content_items)

        # ---------- Store results (per user) ----------
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:

                # A. item_mappings
                for time_block, items in categorized_data.items():
                    for item in items:
                        cursor.execute("""
                            INSERT INTO item_mappings 
                                (user_identifier, url, source, title, content, timestamp, time_block, ml_genre)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (user_identifier, url) DO UPDATE SET
                                time_block = EXCLUDED.time_block,
                                ml_genre   = EXCLUDED.ml_genre;
                        """, (
                            user_id,
                            item.get('url'),
                            item.get('source'),
                            item.get('title'),
                            item.get('content'),
                            item.get('timestamp'),
                            time_block,
                            item.get('genre')
                        ))

                # B. temporal_clusters
                for time_block, queries in time_keywords.items():
                    for q_obj in queries:
                        cursor.execute("""
                            INSERT INTO temporal_clusters 
                                (user_identifier, time_block, genre, search_query)
                            VALUES (%s, %s, %s, %s);
                        """, (user_id, time_block, q_obj['genre'], q_obj['query']))

                # C. recommendations
                for time_block, queries in time_keywords.items():
                    recs = RecommendationService.fetch_google_cse_results(queries)
                    for rec in recs:
                        cursor.execute("""
                            INSERT INTO recommendations 
                                (user_identifier, title, url, snippet, source_domain, 
                                 genre, matched_query, image_url, time_block)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (user_identifier, url) DO UPDATE SET
                                title       = EXCLUDED.title,
                                snippet     = EXCLUDED.snippet,
                                genre       = EXCLUDED.genre,
                                time_block  = EXCLUDED.time_block;
                        """, (
                            user_id,
                            rec.get('title'),
                            rec.get('url'),
                            rec.get('snippet'),
                            rec.get('source_domain'),
                            rec.get('genre'),
                            rec.get('matched_query'),
                            rec.get('image_url'),
                            time_block
                        ))

            conn.commit()

        print(f"[{user_id}] Pipeline completed successfully.")
# ==========================================
# DAG CONFIGURATION
# ==========================================
default_args = {
    'owner': 'omnistream_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'omnistream_ml_pipeline',
    default_args=default_args,
    description='Nightly ML clustering for OmniStream AI',
    schedule_interval='0 2 * * *', # Runs daily at 2:00 AM
    start_date=datetime(2026, 9, 29),
    catchup=False,
    tags=['mlops', 'recommendation'],
) as dag:

    run_pipeline = PythonOperator(
        task_id='execute_ml_pipeline',
        python_callable=execute_ml_pipeline,
    )

    run_pipeline