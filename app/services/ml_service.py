import os
import requests
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import diags
import logging

logger = logging.getLogger(__name__)

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