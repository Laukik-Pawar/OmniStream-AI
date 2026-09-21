from flask import Blueprint, jsonify, session
from app.services.reddit_service import RedditService
from app.services.ml_service import MLService
from app.services.content_service import ContentService

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/generate-recommendations', methods=['GET'])
def generate_recommendations():
    """Fetch history, run clustering, classify genres, and fetch search recommendations."""
    if 'reddit_credentials' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    try:
        # 1. Fetch History
        reddit_service = RedditService(session['reddit_credentials'])
        history_data = reddit_service.fetch_history(limit=15)
        
        if not history_data:
            return jsonify({'message': 'No history found to analyze.'}), 404

        # 2. Cluster Content
        df, cluster_terms = MLService.categorize_content(history_data, num_categories=3)
        
        # 3. Detect Genres via Hugging Face API
        genres = MLService.detect_genres(cluster_terms)
        
        # 4. Fetch web recommendations based on genres and keywords
        content_service = ContentService()
        recommendations = {}
        
        for cluster_id, terms in cluster_terms.items():
            genre = genres.get(cluster_id, "General")
            search_query = f"{genre} {' '.join(terms[:3])} recommendations"
            
            search_results = content_service.scrape_content(search_query, num_results=3)
            recommendations[genre] = {
                'keywords': terms,
                'articles': search_results
            }
            
        return jsonify({
            'success': True,
            'data': recommendations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500