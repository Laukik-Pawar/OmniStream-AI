import os
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class MLService:
    @staticmethod
    def categorize_content(content_items, num_categories=5):
        """Vectorize combined history via TF-IDF and cluster with KMeans."""
        if not content_items:
            return pd.DataFrame(columns=['search_term', 'cluster', 'timestamp']), {}

        texts = [f"{item.get('title', '')} {item.get('content', '')}" for item in content_items]
        
        # Convert text to numeric features
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        num_clusters = min(len(texts), num_categories)
        if num_clusters < 1:
            return pd.DataFrame(columns=['search_term', 'cluster', 'timestamp']), {}

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5)
        kmeans.fit(X)
        
        terms = vectorizer.get_feature_names_out()
        cluster_terms = {}
        for i in range(num_clusters):
            # Get the top 5 terms for each cluster
            center_terms = kmeans.cluster_centers_[i].argsort()[-5:][::-1]
            cluster_terms[i] = [terms[idx] for idx in center_terms]

        df = pd.DataFrame({
            'search_term': texts,
            'cluster': kmeans.labels_,
            'timestamp': [item.get('timestamp', '') for item in content_items]
        })
        return df, cluster_terms

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
                    
                    # Force the raw API response to print in the terminal
                    logger.warning(f"HF Raw Response: {result}")
                    
                    # The new API wraps the response in a list. 
                    if isinstance(result, list):
                        result = result[0]
                        
                    # Extract the genre by checking for both the old and new router JSON formats
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