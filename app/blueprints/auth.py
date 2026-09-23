import os
import uuid
import flask
from flask import Blueprint, session, request, redirect, url_for, jsonify
from google_auth_oauthlib.flow import Flow
import praw

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
auth_bp = Blueprint('auth', __name__)

# --- YOUTUBE OAUTH 2.0 CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "client_secret.json")
YOUTUBE_SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

@auth_bp.route('/youtube/login')
def youtube_login():
    """Initiates the YouTube OAuth 2.0 flow."""
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=YOUTUBE_SCOPES,
        redirect_uri=url_for('auth.oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)

@auth_bp.route('/oauth2callback')
def oauth2callback():
    """Handles the redirect from Google and stores credentials in the session."""
    state = session.get('state')
    if not state or state != request.args.get('state'):
        return "State mismatch. CSRF attempt detected.", 400

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=YOUTUBE_SCOPES,
        state=state,
        redirect_uri=url_for('auth.oauth2callback', _external=True)
    )
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    
    session['youtube_credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    return redirect(url_for('views.dashboard'))


# --- REDDIT OAUTH 2.0 CONFIGURATION ---
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = 'web:omnistream-ai:v1.0'

@auth_bp.route('/reddit/login')
def reddit_login():
    """Initiates the Reddit OAuth 2.0 flow."""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        redirect_uri=url_for('auth.reddit_callback', _external=True),
        user_agent=REDDIT_USER_AGENT
    )
    state = str(uuid.uuid4())
    session['reddit_state'] = state
    
    # Request 'history' to read upvotes, and 'identity' to verify the user
    auth_url = reddit.auth.url(scopes=['history', 'identity', 'read'], state=state, duration='permanent')
    return redirect(auth_url)

@auth_bp.route('/reddit/callback')
def reddit_callback():
    """Handles the redirect from Reddit and exchanges the code for a refresh token."""
    state = request.args.get('state')
    if state != session.get('reddit_state'):
        return jsonify({'error': 'State mismatch. CSRF attempt detected.'}), 400
        
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'Authorization denied.'}), 400

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        redirect_uri=url_for('auth.reddit_callback', _external=True),
        user_agent=REDDIT_USER_AGENT
    )
    
    # Exchange the authorization code for a permanent refresh token
    refresh_token = reddit.auth.authorize(code)
    
    session['reddit_credentials'] = {
        'client_id': REDDIT_CLIENT_ID,
        'client_secret': REDDIT_CLIENT_SECRET,
        'refresh_token': refresh_token,
        'user_agent': REDDIT_USER_AGENT
    }
    return redirect(url_for('views.dashboard'))

@auth_bp.route('/status')
def status():
    """Simple health check to verify connection state."""
    return jsonify({
        'youtube_connected': 'youtube_credentials' in session,
        'reddit_connected': 'reddit_credentials' in session
    })