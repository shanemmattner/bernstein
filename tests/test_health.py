from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bernstein.api.health import router


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_status_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.json() == {"status": "ok", "version": "0.1.0"}


def test_health_content_type(client: TestClient) -> None:
    resp = client.get("/health")
    assert "application/json" in resp.headers["content-type"]
