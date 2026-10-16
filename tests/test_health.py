import json


def test_landing_page(client):
    """Test that the root landing page loads successfully and identifies as TradeMind."""
    response = client.get("/")
    assert response.status_code == 200
    
    html = response.get_data(as_text=True)
    assert "TradeMind" in html
    assert "Phase 1: Project Setup" in html
    assert "System Health &amp; Runtime Status" in html or "System Health & Runtime Status" in html
    assert "main-heading" in html
    assert "btn-check-health" in html


def test_health_check_endpoint(client):
    """Test that the /health endpoint returns 200 OK with expected JSON payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.is_json
    assert response.content_type == "application/json"

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "TradeMind"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "testing"
    assert "timestamp_utc" in data
    assert "modules" in data
    assert data["modules"]["web_server"] == "operational"


def test_api_health_endpoint(client):
    """Test that the /api/health alias endpoint returns identical healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "TradeMind"


def test_static_assets_served(client):
    """Test that static CSS and JavaScript assets are accessible."""
    css_response = client.get("/static/css/style.css")
    assert css_response.status_code == 200
    assert "TradeMind Design System" in css_response.get_data(as_text=True)

    js_response = client.get("/static/js/main.js")
    assert js_response.status_code == 200
    assert "fetchHealthStatus" in js_response.get_data(as_text=True)


def test_404_handling(client):
    """Test handling of non-existent routes."""
    response = client.get("/non-existent-route")
    assert response.status_code == 404
