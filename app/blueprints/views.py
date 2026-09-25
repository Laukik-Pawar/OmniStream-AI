from flask import Blueprint, render_template, session, flash, redirect, url_for, request, jsonify
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
    time_keywords = {}
    if content_items:
        # categorize_content now returns BOTH the categorized data AND the K-Means keywords
        categorized_data, time_keywords = MLService.categorize_content(content_items)
    else:
        categorized_data = {}
        flash("No content available. Please connect your accounts.", "info")

    # 4. Render the frontend template, passing the time_keywords instead of recommendations
    return render_template('dashboard.html', data=categorized_data, time_keywords=time_keywords)

@views_bp.route('/api/recommendations')
def api_recommendations():
    """Async endpoint to fetch live CSE data for infinite scrolling."""
    query = request.args.get('q', '')
    offset = request.args.get('offset', 1, type=int)
    
    # Enforce maximum constraints (50 results total, 5 per page)
    if offset > 46:
        return jsonify([])
        
    results = MLService.fetch_paginated_cse(query, start_index=offset, num=5)
    return jsonify(results)