# import os

# class Config:
#     """Base configuration class."""
#     SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-dev-key')
#     GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
#     GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# class DevelopmentConfig(Config):
#     DEBUG = True
#     TESTING = False

# class ProductionConfig(Config):
#     DEBUG = False
#     TESTING = False

# config_dict = {
#     'development': DevelopmentConfig,
#     'production': ProductionConfig,
#     'default': DevelopmentConfig
# }
import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a-secure-development-key-123')

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# This dictionary maps directly to the config_dict imported in your __init__.py file
config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}