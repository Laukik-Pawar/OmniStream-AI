import os
# Force Hugging Face to use a writable directory inside the Docker container
os.environ["HF_HOME"] = "/tmp/huggingface_cache"

# Authenticate the local model download to avoid rate limits and warnings
if os.environ.get("HUGGINGFACE_API_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_API_TOKEN")
import requests
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import diags
from transformers import pipeline
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

logger = logging.getLogger(__name__)

# Initialize the BART zero-shot classifier globally so it only loads into memory once
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    logger.error(f"Failed to load BART model: {e}")
    classifier = None

class MLService:


    @staticmethod
    def categorize_content(content_items, num_categories=5):
        """Vectorize full history, but isolate the last 14 days for live web recommendations."""
        if not content_items:
            return {}, {}

        # 1. Parse timestamps and extract text
        texts = [f"{item.get('title', '')} {item.get('content', '')}" for item in content_items]
        
        # NEW: Aggressive Text Cleaning (Remove URLs and special characters)
        clean_texts = [re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) for text in texts]
        clean_texts = [re.sub(r'[^\w\s]', ' ', text) for text in clean_texts]

        utc_timestamps = pd.to_datetime([item.get('timestamp') for item in content_items], utc=True, format='ISO8601')
        local_timestamps = utc_timestamps.tz_convert('America/New_York')
        
        for idx, item in enumerate(content_items):
            item['timestamp'] = local_timestamps[idx].strftime('%Y-%m-%d %H:%M')
            
        # NEW: Expand Stop Words to ignore internet metadata
        custom_junk = [
            'com', 'www', 'http', 'https', 'video', 'watch', 'post', 
            'description', 'link', 'youtube', 'reddit', 'album', 'channel'
        ]
        combined_stops = list(ENGLISH_STOP_WORDS) + custom_junk

        # 2. Process ALL data for the Temporal and Genre Dashboard Tabs
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
        
        terms = vectorizer.get_feature_names_out()
        cluster_terms = {}
        for i in range(num_clusters):
            # NEW: Filter out numbers and extremely short words from final keywords
            center_terms = kmeans.cluster_centers_[i].argsort()[::-1]
            valid_terms = [terms[idx] for idx in center_terms if not terms[idx].isnumeric() and len(terms[idx]) > 2]
            cluster_terms[i] = valid_terms[:5]

        cluster_genres = MLService.detect_genres(cluster_terms)

        # 3. Setup Temporal Routing
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
            item['genre'] = cluster_genres.get(cluster_id, "General")
            
            categorized_data[time_block].append(item)
            all_time_counts[time_block][cluster_id] = all_time_counts[time_block].get(cluster_id, 0) + 1
            
            if pd.notna(ts) and ts >= cutoff_date:
                recent_counts[time_block][cluster_id] = recent_counts[time_block].get(cluster_id, 0) + 1
            
        # 4. Extract Keywords specifically for the Recommendations Tab
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
                time_keywords[block] = [{"query": "news", "genre": "General"}]

        active_data = {k: v for k, v in categorized_data.items() if v}
        return active_data, time_keywords
    
    @staticmethod
    def detect_genres(cluster_terms):
        """Remote zero-shot classification using Hugging Face Serverless API."""
        if not cluster_terms:
            return {}

        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            logger.warning("HUGGINGFACE_API_TOKEN not set. Falling back to generic labels.")
            return {k: "General" for k in cluster_terms.keys()}

        # Pointing to the active Hugging Face inference router
        api_url = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        possible_labels = [
            "Technology", "Entertainment", "Sports", "Education", "News", 
            "Health", "Business", "Lifestyle", "Science", "Travel", "Finance", "Gaming"
        ]

        cluster_genres = {}
        for cluster_id, terms in cluster_terms.items():
            if not terms:
                continue
            
            text_to_classify = " ".join(str(t) for t in terms)
            payload = {
                "inputs": text_to_classify,
                "parameters": {"candidate_labels": possible_labels}
            }

            try:
                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    
                    logger.warning(f"HF Raw Response: {result}")
                    
                    if isinstance(result, list):
                        result = result[0]
                        
                    if 'labels' in result:
                        cluster_genres[int(cluster_id)] = result['labels'][0]
                    elif 'label' in result:
                        cluster_genres[int(cluster_id)] = result['label']
                    elif 'error' in result:
                        logger.error(f"HF Model Loading: {result['error']}")
                        cluster_genres[int(cluster_id)] = "General"
                    else:
                        cluster_genres[int(cluster_id)] = "General"
                else:
                    logger.error(f"HF API Error: {response.text}")
                    cluster_genres[int(cluster_id)] = "General"
            except Exception as e:
                logger.error(f"Error connecting to HF API: {e}")
                cluster_genres[int(cluster_id)] = "General"

        return cluster_genres

    @staticmethod
    def fetch_cse_recommendation(query):
        """Fetches the top live web result for the cluster keywords using Google CSE."""
        api_key = os.environ.get("GOOGLE_CSE_API_KEY")
        cse_id = os.environ.get("GOOGLE_CSE_ID")
        
        if not api_key or not cse_id:
            return None
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": 1  # We only need the absolute top result
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                results = response.json().get("items", [])
                if results:
                    return {
                        "title": results[0].get("title"),
                        "link": results[0].get("link"),
                        "snippet": results[0].get("snippet")
                    }
        except Exception as e:
            logger.error(f"CSE Fetch Error: {e}")
        return None

    @staticmethod
    def generate_recommendations(content_items, num_clusters=3):
        """
        Groups user interactions into K-Means clusters, maps them to dominant viewing times, 
        and fetches brand new content via Google CSE based on the extracted keywords.
        """
        if not content_items or len(content_items) < num_clusters:
            return []

        try:
            # 1. Feature Extraction & Temporal Mapping
            df = pd.DataFrame(content_items)
            df['text_features'] = df['title'].fillna('') + " " + df['content'].fillna('')
            
            # Map timestamps to find the user's viewing patterns
            timestamps = pd.to_datetime(df['timestamp'])
            df['hour'] = timestamps.dt.hour
            df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            X = vectorizer.fit_transform(df['text_features'])

            # 2. K-Means Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            df['cluster'] = kmeans.fit_predict(X)

            # Find the most frequent time of day for each cluster
            dominant_times = df.groupby('cluster')['time_of_day'].agg(lambda x: x.value_counts().index[0]).to_dict()

            # 3. Extract Centroid Keywords
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()
            
            cluster_profiles = []
            for i in range(num_clusters):
                top_terms = [terms[ind] for ind in order_centroids[i, :5]]
                cluster_profiles.append(" ".join(top_terms))

            # 4. Zero-Shot Classification & Google CSE Fetch
            candidate_labels = [
                "Technology", "Entertainment", "Education", "Lifestyle", 
                "Finance", "Sports", "Gaming", "Science"
            ]

            recommendations = []
            
            for idx, profile in enumerate(cluster_profiles):
                # Classify Genre
                if classifier:
                    result = classifier(profile, candidate_labels)
                    top_category = result['labels'][0]
                else:
                    top_category = "General Interest"

                # Fetch brand new content from Google CSE using the cluster keywords
                new_content = MLService.fetch_cse_recommendation(profile)
                
                keywords_list = [t.capitalize() for t in profile.split(' ')]
                view_time = dominant_times.get(idx, 'Anytime')

                recommendations.append({
                    'category': top_category,
                    'keywords': keywords_list,
                    'description': f"Recommended for your {view_time} viewing habits.",
                    'new_title': new_content['title'] if new_content else "Live web discovery inactive.",
                    'new_url': new_content['link'] if new_content else "#",
                    'new_snippet': new_content['snippet'] if new_content else "Add GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID to your .env file to enable live search."
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error in recommendation pipeline: {e}")
            return []

    @staticmethod
    def fetch_paginated_cse(query, start_index=1, num=5):
        """Fetches paginated search results dynamically from Google CSE."""
        api_key = os.environ.get("GOOGLE_CSE_API_KEY")
        cse_id = os.environ.get("GOOGLE_CSE_ID")
        
        if not api_key or not cse_id or not query:
            return []
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "start": start_index,
            "num": num
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                results = response.json().get("items", [])
                return [{
                    "title": r.get("title", "Unknown Title"),
                    "link": r.get("link", "#"),
                    "snippet": r.get("snippet", "No description available.")
                } for r in results]
            else:
                logger.error(f"CSE API Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"CSE Paginated Fetch Error: {e}")
            
        return []