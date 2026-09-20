import os
from app import create_app

# Determine which config to load based on the environment
config_name = os.getenv('FLASK_ENV', 'default')
app = create_app(config_name)

if __name__ == '__main__':
    # host='0.0.0.0' exposes the server on your local network
    app.run(host='0.0.0.0', port=5000)