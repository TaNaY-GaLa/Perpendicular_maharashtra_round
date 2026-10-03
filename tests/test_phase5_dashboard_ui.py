"""
Phase 5 Test Suite: Web UI Dashboard Endpoints & Trace Exploration
Tests:
1. Test GET / serves the Black Box HTML dashboard (HTTP 200).
2. Test GET /dashboard serves the Black Box HTML dashboard (HTTP 200).
3. Test API runs list integrated with dashboard schema.
4. Test step reconstruction endpoint matches dashboard requirements.
"""

import pytest
from fastapi.testclient import TestClient
from blackbox.proxy.server import app


@pytest.fixture
def http_client():
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


def test_dashboard_root_serves_html(http_client):
    resp = http_client.get("/")
    assert resp.status_code == 200
    assert "BLACK BOX | AI Agent Flight Recorder" in resp.text
    assert "Replay from Checkpoint" in resp.text


def test_dashboard_url_serves_html(http_client):
    resp = http_client.get("/dashboard")
    assert resp.status_code == 200
    assert "BLACK BOX | AI Agent Flight Recorder" in resp.text
    assert "tailwind" in resp.text


def test_dashboard_api_stats(http_client):
    resp = http_client.get("/api/stats")
    assert resp.status_code == 200
    assert "total_runs" in resp.json()
