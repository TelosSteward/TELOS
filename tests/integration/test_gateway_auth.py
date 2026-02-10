"""
Gateway Authentication Tests
=============================

Tests for the TELOS Gateway authentication and rate limiting.
"""

import pytest
from fastapi.testclient import TestClient

from telos_gateway.main import app


@pytest.fixture
def client():
    """Create a test client for the gateway app."""
    return TestClient(app)


# ============================================================================
# Health endpoint (no auth required)
# ============================================================================


class TestHealthEndpoint:
    """Health check should work without authentication."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TELOS Gateway"
        assert "endpoints" in data

    def test_models_returns_200(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0


# ============================================================================
# Authentication
# ============================================================================


class TestAuthentication:
    """Chat completions endpoint requires Bearer token."""

    def test_missing_auth_returns_401(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["type"] == "authentication_error"

    def test_empty_bearer_returns_401(self, client):
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer "},
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401

    def test_valid_bearer_token_accepted(self, client):
        """A valid bearer token should not return 401."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-api-key-12345"},
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        # Should not be 401 (auth passes, may fail downstream for missing LLM key)
        assert response.status_code != 401


# ============================================================================
# Error format consistency
# ============================================================================


class TestErrorFormat:
    """All errors should use consistent JSON format."""

    def test_401_error_format(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        data = response.json()
        assert "detail" in data
        err = data["detail"]["error"]
        assert "message" in err
        assert "type" in err
        assert "code" in err

    def test_404_unknown_path(self, client):
        response = client.get("/v1/nonexistent-path")
        assert response.status_code == 404


# ============================================================================
# CORS
# ============================================================================


class TestCORS:
    """CORS should be properly configured (not wildcard)."""

    def test_cors_allows_configured_origin(self, client):
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        # Should include the allowed origin
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"

    def test_cors_rejects_unknown_origin(self, client):
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # Should NOT include the evil origin in allowed headers
        allowed = response.headers.get("access-control-allow-origin", "")
        assert "evil.example.com" not in allowed
