from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

VERSION = "0.1.0"


@router.get("/health")
def health() -> dict[str, str]:
    """Return service health status and version."""
    return {"status": "ok", "version": VERSION}
