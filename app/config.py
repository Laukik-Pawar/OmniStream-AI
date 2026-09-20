import os

class Config:
    """Base configuration class."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-dev-key')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}