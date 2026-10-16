import pytest
from backend.app import create_app


@pytest.fixture
def app():
    """Create and configure an isolated Flask application instance for testing."""
    test_app = create_app("testing")
    yield test_app


@pytest.fixture
def client(app):
    """A test client for making HTTP requests to the application."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """A test CLI runner for testing Flask CLI commands."""
    return app.test_cli_runner()
