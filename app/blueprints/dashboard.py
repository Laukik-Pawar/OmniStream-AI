from flask import Blueprint, jsonify, session, request
from app.services.reddit_service import RedditService
from app.services.youtube_service import YouTubeService
from app.services.ml_service import MLService
from app.services.content_service import ContentService
# Import your actual database model class here (e.g., from app.models import RecommendationModel)

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/generate-recommendations', methods=['GET'])
def generate_recommendations():
    """Fetch history, run clustering, classify genres, and fetch search recommendations."""
    if 'reddit_credentials' not in session and 'youtube_api_key' not in session:
        return jsonify({'error': 'Not authenticated with Reddit or YouTube'}), 401
        
    try:
        history_data = []

        # 1. Fetch Reddit History
        if 'reddit_credentials' in session:
            try:
                reddit_service = RedditService(session['reddit_credentials'])
                reddit_data = reddit_service.fetch_history(limit=15)
                if reddit_data:
                    history_data.extend(reddit_data)
            except Exception as e:
                print(f"Error fetching Reddit data: {e}")

        # 2. Fetch YouTube Data
        if 'youtube_api_key' in session:
            try:
                yt_service = YouTubeService(session['youtube_api_key'])
                youtube_data = yt_service.fetch_liked_videos(max_results=10)
                if youtube_data:
                    history_data.extend(youtube_data)
            except Exception as e:
                print(f"Error fetching YouTube data: {e}")

        if not history_data:
            return jsonify({'message': 'No history found across Reddit or YouTube to analyze.'}), 404

        # 3. Cluster Combined Content
        df, cluster_terms = MLService.categorize_content(history_data, num_categories=3)
        
        # 4. Detect Genres via Hugging Face API
        genres = MLService.detect_genres(cluster_terms)
        
        # 5. Fetch web recommendations based on genres and keywords
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


@dashboard_bp.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Hybrid endpoint: Serves stored DB items for initial loads, then fetches live content on scroll."""
    query_genre = request.args.get('q', '')
    offset = int(request.args.get('offset', 0))
    limit = 4  # Batch size matching your UI layout
    
    try:
        # Query your database model for stored records matching the genre/offset
        # (Make sure RecommendationModel matches your actual database model class name)
        db_items = RecommendationModel.query.filter_by(genre=query_genre)\
                                            .offset(offset)\
                                            .limit(limit)\
                                            .all()
        
        results = [item.to_dict() for item in db_items]
        
        # Real-time fallback: If database batch runs out of items for this offset, fetch live content
        if len(results) < limit:
            content_service = ContentService()
            needed = limit - len(results)
            search_query = f"{query_genre} recommendations"
            extra_articles = content_service.scrape_content(search_query, num_results=needed)
            
            for article in extra_articles:
                results.append({
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "snippet": article.get("snippet"),
                    "image_url": article.get("image_url"),
                    "source_domain": article.get("source_domain", "web"),
                    "genre": query_genre
                })
                
        return jsonify(results)
        
    except Exception as e:
        print(f"Error fetching infinite scroll recommendations: {e}")
        return jsonify([]), 500