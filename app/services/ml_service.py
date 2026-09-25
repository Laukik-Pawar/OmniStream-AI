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
        """Vectorize combined history via TF-IDF, apply time-decay, and cluster."""
        if not content_items:
            return {} # Return empty dict to prevent Jinja errors

        texts = [f"{item.get('title', '')} {item.get('content', '')}" for item in content_items]
        
        # 1. Parse timestamps as UTC, then immediately convert to localized Eastern Time
        utc_timestamps = pd.to_datetime([item.get('timestamp') for item in content_items], utc=True, format='ISO8601')
        local_timestamps = utc_timestamps.tz_convert('America/New_York')
        
        # Update the original dictionaries so the UI displays the correct localized time
        for idx, item in enumerate(content_items):
            item['timestamp'] = local_timestamps[idx].strftime('%Y-%m-%d %H:%M')
            
        # 2. Convert text to numeric features
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # 3. TEMPORAL ANALYTICS: Calculate time decay weights using localized times
        most_recent = max(local_timestamps)
        days_old = np.array([(most_recent - ts).days for ts in local_timestamps])
        
        # Apply an exponential decay factor (e.g., half-life of 30 days)
        half_life = 30
        decay_weights = np.exp(-np.log(2) * days_old / half_life)
        
        # Scale the TF-IDF matrix by the temporal weights
        weight_matrix = diags(decay_weights)
        X_weighted = weight_matrix.dot(X)
        
        num_clusters = min(len(texts), num_categories)
        if num_clusters < 1:
            return {}

        # 4. Apply K-Means clustering on the temporally-adjusted data
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5)
        kmeans.fit(X_weighted)
        
        terms = vectorizer.get_feature_names_out()
        cluster_terms = {}
        for i in range(num_clusters):
            center_terms = kmeans.cluster_centers_[i].argsort()[-5:][::-1]
            cluster_terms[i] = [terms[idx] for idx in center_terms]

        # 5. REMOTE CLASSIFICATION: Fetch genres using Hugging Face
        cluster_genres = MLService.detect_genres(cluster_terms)

        # 6. INTRADAY ROUTING: Map the localized hour to a specific time block
        df = pd.DataFrame({
            'interaction_hour': local_timestamps.hour, # Correctly extracts localized hour
            'cluster': kmeans.labels_
        })
        
        bins = [0, 6, 12, 18, 24]
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        df['time_of_day'] = pd.cut(df['interaction_hour'], bins=bins, labels=labels, right=False)
        
        # 7. ASSEMBLE TEMPLATE DATA: Group enriched items by time block
        categorized_data = {label: [] for label in labels}
        
        for idx, item in enumerate(content_items):
            time_block = df['time_of_day'].iloc[idx]
            cluster_id = df['cluster'].iloc[idx]
            
            # Enrich the original item dictionary with ML outputs
            item['cluster_id'] = int(cluster_id)
            item['genre'] = cluster_genres.get(int(cluster_id), "General")
            
            # Append to the correct time block list
            categorized_data[time_block].append(item)
            
        # Filter out empty time blocks so the UI only shows active periods
        return {k: v for k, v in categorized_data.items() if v}

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