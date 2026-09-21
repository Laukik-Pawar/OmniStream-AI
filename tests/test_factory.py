from app import create_app

def test_health_check():
    """Test that the application factory creates a working app."""
    app = create_app('testing')
    client = app.test_client()
    
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'