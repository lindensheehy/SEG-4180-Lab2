import pytest
from app import app
import io

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    assert rv.json == {"status": "healthy"}

def test_predict_no_auth(client):
    # Expect 401 Unauthorized because the API key is missing
    rv = client.post('/predict', data={'file': (io.BytesIO(b"fake image data"), 'test.jpg')})
    assert rv.status_code == 401