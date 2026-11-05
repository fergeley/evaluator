import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
from routers.auth import create_access_token

client = TestClient(app)

def get_test_token():
    """Helper function to create a test token"""
    return create_access_token(data={"sub": "test@example.com"})

def test_endpoints_without_token():
    """Test that endpoints require authentication"""
    # Test initialize endpoint
    schema_data = {"schema": {"table": "test"}}
    response = client.post("/malay2sql/initialize", json=schema_data)
    assert response.status_code == 401

    # Test query endpoint
    query_data = {"query": "Show all users"}
    response = client.post("/malay2sql/query", json=query_data)
    assert response.status_code == 401

def test_endpoints_with_token():
    """Test endpoints with valid authentication"""
    token = get_test_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Test initialize endpoint (same shape the frontend uploads)
    schema_data = {"schema": {
        "table_name": "users",
        "columns": {
            "id": {"data_type": "Int64", "description": "Unique identifier for each user"},
            "name": {"data_type": "String", "description": "Full name of the user"},
        },
    }}
    response = client.post("/malay2sql/initialize", json=schema_data, headers=headers)
    assert response.status_code == 200

    # Test query endpoint
    query_data = {"query": "Show all users"}
    response = client.post("/malay2sql/query", json=query_data, headers=headers)
    assert response.status_code == 200