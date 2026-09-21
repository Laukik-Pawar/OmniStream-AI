from flask import Blueprint, request, jsonify, session
from app.services.reddit_service import RedditService

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate and store API credentials in the session."""
    try:
        reddit_creds = {
            'client_id': request.json.get('reddit_client_id'),
            'client_secret': request.json.get('reddit_client_secret'),
            'user_agent': 'omnistream-ai:v1.0',
            'username': request.json.get('reddit_username'),
            'password': request.json.get('reddit_password')
        }
        
        # Verify Reddit connection validity by attempting to initialize the service
        RedditService(reddit_creds)
        
        # If successful, store in session
        session['reddit_credentials'] = reddit_creds
        
        return jsonify({'success': True, 'message': 'Authenticated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@auth_bp.route('/status', methods=['GET'])
def status():
    """Check if the user has active credentials in their session."""
    is_reddit_auth = 'reddit_credentials' in session
    # We will add YouTube token checking here later
    return jsonify({
        'reddit_authenticated': is_reddit_auth
    })