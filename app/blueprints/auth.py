import os
import uuid
import psycopg2
from psycopg2.extras import Json
from flask import Blueprint, session, request, redirect, url_for, jsonify
try:
    from google_auth_oauthlib.flow import Flow
except ImportError:
    Flow = None
import praw

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
auth_bp = Blueprint('auth', __name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "client_secret.json")
YOUTUBE_SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = 'web:omnistream-ai:v1.0'

def save_credentials_to_db(platform, credentials_dict):
    """Upserts OAuth tokens into Neon DB for Airflow background access."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("Warning: DATABASE_URL not set, skipping token storage.")
        return

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                # Ensure the admin row exists before attempting to update it
                cursor.execute("""
                    INSERT INTO user_oauth_tokens (user_identifier) 
                    VALUES ('admin') 
                    ON CONFLICT (user_identifier) DO NOTHING;
                """)
                
                if platform == 'youtube':
                    cursor.execute("""
                        UPDATE user_oauth_tokens 
                        SET youtube_token = %s, last_updated = CURRENT_TIMESTAMP 
                        WHERE user_identifier = 'admin';
                    """, (Json(credentials_dict),))
                elif platform == 'reddit':
                    cursor.execute("""
                        UPDATE user_oauth_tokens 
                        SET reddit_token = %s, last_updated = CURRENT_TIMESTAMP 
                        WHERE user_identifier = 'admin';
                    """, (Json(credentials_dict),))
            conn.commit()
    except Exception as e:
        print(f"Database error saving {platform} token: {e}")

# --- YOUTUBE ROUTES ---
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
    """Handles the redirect from Google and stores credentials."""
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
    
    credentials_dict = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    
    # Store in session for immediate frontend use and in DB for Airflow
    session['youtube_credentials'] = credentials_dict
    save_credentials_to_db('youtube', credentials_dict)
    
    return redirect(url_for('views.dashboard'))

# --- REDDIT ROUTES ---
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
    
    auth_url = reddit.auth.url(scopes=['history', 'identity', 'read'], state=state, duration='permanent')
    return redirect(auth_url)

@auth_bp.route('/reddit/callback')
def reddit_callback():
    """Handles the redirect from Reddit, exchanges the code, and saves to PostgreSQL."""
    state = request.args.get('state')
    if state != session.get('reddit_state'):
        return jsonify({'error': 'State mismatch. CSRF attempt detected.'}), 400
        
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'Authorization denied.'}), 400

    reddit_init = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        redirect_uri=url_for('auth.reddit_callback', _external=True),
        user_agent=REDDIT_USER_AGENT
    )
    
    refresh_token = reddit_init.auth.authorize(code)
    
    reddit_auth = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        refresh_token=refresh_token,
        user_agent=REDDIT_USER_AGENT
    )
    username = reddit_auth.user.me().name
    
    credentials_dict = {
        'username': username,
        'refresh_token': refresh_token
    }
    
    # Store in session for immediate frontend use and in DB for Airflow
    session['reddit_credentials'] = credentials_dict
    save_credentials_to_db('reddit', credentials_dict)
    
    return redirect(url_for('views.dashboard'))

@auth_bp.route('/status')
def status():
    """Simple health check to verify connection state."""
    return jsonify({
        'youtube_connected': 'youtube_credentials' in session,
        'reddit_connected': 'reddit_credentials' in session
    })