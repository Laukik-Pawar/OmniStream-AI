from flask import Blueprint, render_template, session

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    """Render the login page."""
    return render_template('index.html')

@views_bp.route('/dashboard')
def dashboard():
    """Render the main recommendation dashboard."""
    return render_template('dashboard.html')