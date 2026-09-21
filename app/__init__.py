import os
from flask import Flask
from dotenv import load_dotenv
from app.config import config_dict

load_dotenv()

def create_app(config_name='default'):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_dict[config_name])

    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # ---> ADD THIS BLOCK <---
    from app.blueprints.auth import auth_bp
    from app.blueprints.dashboard import dashboard_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    # ------------------------

    @app.route('/health')
    def health_check():
        return {"status": "healthy", "environment": config_name}

    return app