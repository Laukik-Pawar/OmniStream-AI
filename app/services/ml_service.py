import os

# Force Hugging Face to cache the model in the temporary directory to avoid Docker PermissionErrors
os.environ['HF_HOME'] = '/tmp/huggingface_cache'

import re
import psycopg2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from scipy.sparse import diags
from transformers import pipeline
import logging
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# Initialize the zero-shot classifier once globally
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    logger.error(f"Failed to load BART model: {e}")
    classifier = None

class MLService:

    @staticmethod
    def sanitize_text(text):
        """Preserves natural language while destroying SEO spam, hashtags, and links."""
        if not text:
            return ""
        # Strip out all hashtags
        text = re.sub(r'#\w+', '', text)
        # Strip out URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Strip out common YouTube spam blocks and social promos
        text = re.sub(r'(?i)(follow|subscribe|instagram|twitter|facebook|patreon|tags:).*', '', text)
        return text

    @staticmethod
    def _learn_and_cache_posts(content_items):
        """Classifies individual posts using both subreddit and title context, caching by URL."""
        candidate_labels = ["Technology", "Entertainment", "Education", "Lifestyle", "Finance", "Sports", "Gaming", "Comedy", "Science"]
        
        # 1. Identify which specific posts have no native_genre yet
        unknown_posts = [item for item in content_items if item.get('source') == 'reddit' and not item.get('native_genre')]
                    
        if not unknown_posts:
            return content_items
            
        # 2. Use zero-shot AI to classify each specific post
        new_mappings = {}
        for item in unknown_posts:
            sub = item.get('subreddit', '').replace('r/', '')
            title = item.get('title', '')
            post_url = item.get('url')
            
            if not post_url:
                continue
                
            prompt = f"Subreddit: {sub}. Post Title: {title}"
            
            if classifier:
                result = classifier(prompt, candidate_labels)
                new_mappings[post_url] = result['labels'][0]
            else:
                new_mappings[post_url] = "General"
                
        # 3. Permanently write these new discoveries to the database
        db_url = os.environ.get('DATABASE_URL')
        if db_url and new_mappings:
            try:
                with psycopg2.connect(db_url) as conn:
                    with conn.cursor() as cursor:
                        for url, genre in new_mappings.items():
                            cursor.execute("""
                                INSERT INTO post_genres (post_url, genre) 
                                VALUES (%s, %s) ON CONFLICT (post_url) DO NOTHING;
                            """, (url, genre))
                    conn.commit() 
            except Exception as e:
                logger.error(f"Failed to cache new posts: {e}")
                
        # 4. Apply the newly learned genres back to the current request
        for item in unknown_posts:
            post_url = item.get('url')
            if post_url in new_mappings:
                item['native_genre'] = new_mappings[post_url]
                
        return content_items

    @staticmethod
    def detect_genres(cluster_terms, content_items):
        """Maps clusters to genres based on terms and cached item genres."""
        cluster_genres = {}
        candidate_labels = ["Technology", "Entertainment", "Education", "Lifestyle", "Finance", "Sports", "Gaming", "Comedy", "Science"]
        
        for cluster_id, terms in cluster_terms.items():
            prompt = " ".join(terms)
            if classifier:
                result = classifier(prompt, candidate_labels)
                cluster_genres[cluster_id] = result['labels'][0]
            else:
                cluster_genres[cluster_id] = "General"
                
        return cluster_genres

    @staticmethod
    def categorize_content(content_items, num_categories=5):
        """Vectorizes history and extracts clean, strict keywords for API recommendations."""
        if not content_items:
            return {}, {}

        # 1. Teach the system any unknown specific posts BEFORE starting the math
        content_items = MLService._learn_and_cache_posts(content_items)

        # 2. Extract text (combining Title with the Sanitized Description)
        texts = [f"{item.get('title', '')} {MLService.sanitize_text(item.get('content', ''))}" for item in content_items]
        
        # Remove any remaining special characters
        clean_texts = [re.sub(r'[^\w\s]', ' ', text) for text in texts]

        utc_timestamps = pd.to_datetime([item.get('timestamp') for item in content_items], utc=True, format='ISO8601')
        local_timestamps = utc_timestamps.tz_convert('America/New_York')
        
        for idx, item in enumerate(content_items):
            item['timestamp'] = local_timestamps[idx].strftime('%Y-%m-%d %H:%M')
            
        # 3. Expand Stop Words to aggressively block conversational filler and internet metadata
        custom_junk = [
            'com', 'www', 'http', 'https', 'video', 'watch', 'post', 
            'description', 'link', 'youtube', 'reddit', 'album', 'channel',
            'let', 'make', 'get', 'just', 'like', 'good', 'best', 'new', 'time'
        ]
        combined_stops = list(ENGLISH_STOP_WORDS) + custom_junk

        # Vectorization
        vectorizer = TfidfVectorizer(stop_words=combined_stops, max_features=1000)
        X = vectorizer.fit_transform(clean_texts)
        
        most_recent = max(local_timestamps)
        days_old = np.array([(most_recent - ts).days for ts in local_timestamps])
        
        half_life = 30
        decay_weights = np.exp(-np.log(2) * days_old / half_life)
        
        weight_matrix = diags(decay_weights)
        X_weighted = weight_matrix.dot(X)
        
        num_clusters = min(len(texts), num_categories)
        if num_clusters < 1:
            return {}, {}

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5)
        kmeans.fit(X_weighted)
        
        # 4. Keyword Extraction (STRICT: Top 3 terms only, length > 3)
        terms = vectorizer.get_feature_names_out()
        cluster_terms = {}
        for i in range(num_clusters):
            center_terms = kmeans.cluster_centers_[i].argsort()[::-1]
            valid_terms = [terms[idx] for idx in center_terms if not terms[idx].isnumeric() and len(terms[idx]) > 3]
            cluster_terms[i] = valid_terms[:3]

        cluster_genres = MLService.detect_genres(cluster_terms, content_items)

        # 5. Temporal Routing
        df = pd.DataFrame({
            'interaction_hour': [ts.hour for ts in local_timestamps],
            'cluster': kmeans.labels_,
            'timestamp_obj': local_timestamps
        })
        
        bins = [0, 6, 12, 18, 24]
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        df['time_of_day'] = pd.cut(df['interaction_hour'], bins=bins, labels=labels, right=False)
        
        categorized_data = {label: [] for label in labels}
        all_time_counts = {label: {} for label in labels}
        recent_counts = {label: {} for label in labels}
        
        cutoff_date = pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(days=14)

        for idx, item in enumerate(content_items):
            time_block = df['time_of_day'].iloc[idx]
            cluster_id = int(df['cluster'].iloc[idx])
            ts = df['timestamp_obj'].iloc[idx]
            
            item['cluster_id'] = cluster_id
            item['genre'] = item.get('native_genre') or cluster_genres.get(cluster_id, "General")
            
            categorized_data[time_block].append(item)
            all_time_counts[time_block][cluster_id] = all_time_counts[time_block].get(cluster_id, 0) + 1
            
            if pd.notna(ts) and ts >= cutoff_date:
                recent_counts[time_block][cluster_id] = recent_counts[time_block].get(cluster_id, 0) + 1
            
        # 6. Recommendation Keyword Export
        time_keywords = {}
        for block in labels:
            counts_to_use = recent_counts[block] if recent_counts[block] else all_time_counts[block]
            
            if counts_to_use:
                top_clusters = sorted(counts_to_use, key=counts_to_use.get, reverse=True)[:3]
                
                block_queries = []
                for cluster_id in top_clusters:
                    keywords = " ".join(cluster_terms[cluster_id])
                    genre = cluster_genres.get(cluster_id, "General")
                    block_queries.append({"query": keywords, "genre": genre})
                
                time_keywords[block] = block_queries
            else:
                time_keywords[block] = [{"query": "technology news", "genre": "General"}]

        active_data = {k: v for k, v in categorized_data.items() if v}
        return active_data, time_keywords

    @staticmethod
    def fetch_paginated_cse(query, start_index=1, num=5):
        """Standalone method to support the infinite scroll API endpoint."""
        api_key = os.environ.get('GOOGLE_API_KEY')
        cse_id = os.environ.get('GOOGLE_CSE_ID')
        
        if not api_key or not cse_id or not query:
            return []

        try:
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(
                q=query, 
                cx=cse_id, 
                num=num,
                start=start_index,
                safe="off"
            ).execute()
            
            results = []
            for item in res.get('items', []):
                pagemap = item.get('pagemap', {})
                thumbnail_url = ''
                if 'cse_thumbnail' in pagemap and len(pagemap['cse_thumbnail']) > 0:
                    thumbnail_url = pagemap['cse_thumbnail'][0].get('src', '')
                elif 'cse_image' in pagemap and len(pagemap['cse_image']) > 0:
                    thumbnail_url = pagemap['cse_image'][0].get('src', '')

                results.append({
                    'title': item.get('title', 'No Title'),
                    'url': item.get('link', '#'),
                    'snippet': item.get('snippet', ''),
                    'source_domain': item.get('displayLink', 'Web'),
                    'image_url': thumbnail_url
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching paginated CSE results: {e}")
            return []