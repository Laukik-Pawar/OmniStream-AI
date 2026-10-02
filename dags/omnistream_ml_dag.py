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
    """Ingest live data → run ML clustering → store recommendations."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL is missing")

    content_items = []

    # ==========================================
    # 1. Get tokens
    # ==========================================
    with psycopg2.connect(db_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT youtube_token, reddit_token 
                FROM user_oauth_tokens 
                WHERE user_identifier = 'admin'
            """)
            tokens = cursor.fetchone()

    if not tokens:
        print("No tokens found. Exiting.")
        return

    # ==========================================
    # 2. Ingest Reddit (NEW)
    # ==========================================
    if tokens.get('reddit_token'):
        try:
            reddit_creds = tokens['reddit_token']  # this is already a dict (JSONB)
            reddit = RedditService(reddit_creds)
            
            # First pull fresh data into the DB
            ingested = reddit.ingest_upvoted_posts(limit=50)
            print(f"Ingested {ingested} new Reddit posts")

            # Then read them back for the ML pipeline
            content_items.extend(reddit.fetch_upvoted_posts(limit=50))
        except Exception as e:
            print(f"Reddit ingestion/fetch error: {e}")

    # ==========================================
    # 3. Fetch YouTube
    # ==========================================
    if tokens.get('youtube_token'):
        try:
            yt_videos = YouTubeService.fetch_liked_videos(tokens['youtube_token'], limit=25)
            content_items.extend(yt_videos)
        except Exception as e:
            print(f"YouTube fetch error: {e}")

    if not content_items:
        print("No content fetched. Exiting.")
        return

    # ==========================================
    # 4. Run ML
    # ==========================================
    categorized_data, time_keywords = MLService.categorize_content(content_items)

    # ==========================================
    # 5. Store results
    # ==========================================
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cursor:
            # A. Item mappings
            for time_block, items in categorized_data.items():
                for item in items:
                    cursor.execute("""
                        INSERT INTO item_mappings (url, source, title, content, timestamp, time_block, ml_genre)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET 
                            time_block = EXCLUDED.time_block,
                            ml_genre = EXCLUDED.ml_genre;
                    """, (
                        item.get('url'),
                        item.get('source'),
                        item.get('title'),
                        item.get('content'),
                        item.get('timestamp'),
                        time_block,
                        item.get('genre')
                    ))

            # B. Temporal clusters
            for time_block, queries in time_keywords.items():
                for q_obj in queries:
                    cursor.execute("""
                        INSERT INTO temporal_clusters (time_block, genre, search_query)
                        VALUES (%s, %s, %s);
                    """, (time_block, q_obj['genre'], q_obj['query']))

            # C. Live recommendations
            for time_block, queries in time_keywords.items():
                recs = RecommendationService.fetch_google_cse_results(queries)
                for rec in recs:
                    cursor.execute("""
                        INSERT INTO recommendations 
                            (title, url, snippet, source_domain, genre, matched_query, image_url, time_block)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET 
                            title = EXCLUDED.title,
                            snippet = EXCLUDED.snippet,
                            genre = EXCLUDED.genre,
                            time_block = EXCLUDED.time_block;
                    """, (
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

    print("OmniStream ML Pipeline completed successfully.")
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