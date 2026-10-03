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


def get_db_connection():
    return psycopg2.connect(os.environ.get('DATABASE_URL'))


def save_credentials_to_db(user_identifier: str, platform: str, credentials_dict: dict):
    """Upserts OAuth tokens for any user."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO user_oauth_tokens (user_identifier)
                    VALUES (%s)
                    ON CONFLICT (user_identifier) DO NOTHING;
                """, (user_identifier,))

                if platform == 'youtube':
                    cursor.execute("""
                        UPDATE user_oauth_tokens
                        SET youtube_token = %s, last_updated = CURRENT_TIMESTAMP
                        WHERE user_identifier = %s;
                    """, (Json(credentials_dict), user_identifier))
                elif platform == 'reddit':
                    cursor.execute("""
                        UPDATE user_oauth_tokens
                        SET reddit_token = %s, last_updated = CURRENT_TIMESTAMP
                        WHERE user_identifier = %s;
                    """, (Json(credentials_dict), user_identifier))
            conn.commit()
    except Exception as e:
        print(f"Database error saving {platform} token for {user_identifier}: {e}")


# --- YOUTUBE ROUTES ---
@auth_bp.route('/youtube/login')
def youtube_login():
    if Flow is None:
        return "google-auth-oauthlib is not installed", 500

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=YOUTUBE_SCOPES,
        redirect_uri=url_for('auth.oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'               # forces refresh_token
    )
    session['state'] = state
    return redirect(authorization_url)


@auth_bp.route('/oauth2callback')
def oauth2callback():
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

    # Prefer existing user_identifier (from Reddit) if present
    user_identifier = session.get('user_identifier') or 'youtube_user'

    session['user_identifier'] = user_identifier
    session['youtube_credentials'] = credentials_dict

    save_credentials_to_db(user_identifier, 'youtube', credentials_dict)

    return redirect(url_for('views.dashboard'))


# --- REDDIT ROUTES ---
@auth_bp.route('/reddit/login')
def reddit_login():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        redirect_uri=url_for('auth.reddit_callback', _external=True),
        user_agent=REDDIT_USER_AGENT
    )
    state = str(uuid.uuid4())
    session['reddit_state'] = state
    auth_url = reddit.auth.url(
        scopes=['history', 'identity', 'read'],
        state=state,
        duration='permanent'
    )
    return redirect(auth_url)


@auth_bp.route('/reddit/callback')
def reddit_callback():
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

    # Always use Reddit username as the main identity
    session['reddit_credentials'] = credentials_dict
    session['user_identifier'] = username

    # Auto-create / update user row
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (reddit_username, reddit_refresh_token)
                    VALUES (%s, %s)
                    ON CONFLICT (reddit_username)
                    DO UPDATE SET reddit_refresh_token = EXCLUDED.reddit_refresh_token
                    RETURNING id;
                """, (username, refresh_token))
                cursor.fetchone()

                # Optional: move YouTube token from 'youtube_user' to real username
                cursor.execute("""
                    UPDATE user_oauth_tokens
                    SET youtube_token = (
                        SELECT youtube_token FROM user_oauth_tokens
                        WHERE user_identifier = 'youtube_user'
                    )
                    WHERE user_identifier = %s
                      AND youtube_token IS NULL
                      AND EXISTS (
                          SELECT 1 FROM user_oauth_tokens WHERE user_identifier = 'youtube_user'
                      );
                """, (username,))
            conn.commit()
    except Exception as e:
        print(f"Error creating/linking user: {e}")

    save_credentials_to_db(username, 'reddit', credentials_dict)

    return redirect(url_for('views.dashboard'))


@auth_bp.route('/status')
def status():
    return jsonify({
        'youtube_connected': 'youtube_credentials' in session,
        'reddit_connected': 'reddit_credentials' in session,
        'user_identifier': session.get('user_identifier')
    })