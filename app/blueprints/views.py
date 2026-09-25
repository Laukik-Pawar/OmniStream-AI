from flask import Blueprint, render_template, session, flash, redirect, url_for
from app.services.youtube_service import YouTubeService
from app.services.reddit_service import RedditService
from app.services.ml_service import MLService

views_bp = Blueprint('views', __name__)

@views_bp.route('/dashboard')
def dashboard():
    """Aggregates authenticated user data and passes it through the ML pipeline."""
    content_items = []

    # 1. Fetch Reddit Data
    if 'reddit_credentials' in session:
        try:
            print("DEBUG REDDIT CREDENTIALS:", session['reddit_credentials'], flush=True)
            reddit = RedditService(session['reddit_credentials'])
            content_items.extend(reddit.fetch_upvoted_posts())
        except Exception as e:
            flash(f"Failed to load Reddit data: {str(e)}", "warning")
            session.pop('reddit_credentials', None)

    # 2. Fetch YouTube Data
    if 'youtube_credentials' in session:
        try:
            yt_videos = YouTubeService.fetch_liked_videos(session['youtube_credentials'])
            content_items.extend(yt_videos)
        except Exception:
            flash("YouTube session expired. Please log in again.", "danger")
            session.pop('youtube_credentials', None)
            
    # 3. Process the merged datasets through the ML pipelines
    if content_items:
        categorized_data = MLService.categorize_content(content_items)
        recommendations = MLService.generate_recommendations(content_items)
    else:
        categorized_data = {}
        recommendations = []
        flash("No content available. Please connect your accounts.", "info")

    # 4. Render the frontend template
    return render_template('dashboard.html', data=categorized_data, recommendations=recommendations)