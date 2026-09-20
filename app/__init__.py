import os
from flask import Flask
from dotenv import load_dotenv
from app.config import config_dict

# Load environment variables from .env
load_dotenv()

def create_app(config_name='default'):
    """Flask application factory."""
    app = Flask(__name__, instance_relative_config=True)
    
    # Load configuration
    app.config.from_object(config_dict[config_name])

    # Ensure the instance folder exists (useful for SQLite or local temp files)
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # A simple health check route to verify the factory is working
    @app.route('/health')
    def health_check():
        return {"status": "healthy", "environment": config_name}

    return app