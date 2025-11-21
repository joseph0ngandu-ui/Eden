#!/usr/bin/env python3
"""AWS Bridge Router for the local FastAPI backend.

This module ensures that any auth or balance traffic hitting the local
backend is transparently proxied to the AWS serverless backend so that:
- JWTs are issued and validated by AWS only
- Balance reads come from DynamoDB via AWS lambdas
- The local SQLite DB is never used for these paths

Exposed endpoints (local):
- POST /auth/register  → POST {AWS_API_BASE}/auth/register
- POST /auth/login     → POST {AWS_API_BASE}/auth/login
- GET  /user/balance   → GET  {AWS_API_BASE}/user/balance

All Authorization headers are forwarded as-is to AWS (JWT passthrough).
"""

from typing import Any, Dict, Optional
import logging

import requests
from fastapi import APIRouter, Body, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .bridge_config import AWS_API_BASE, REQUEST_TIMEOUT_SECONDS, VERIFY_TLS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["aws-bridge"])


def _build_aws_url(path: str) -> str:
    """Join AWS_API_BASE with a relative path that may include a query string."""
    base = AWS_API_BASE.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def _forward_to_aws(method: str, path: str, *, request: Request, json_body: Optional[Any] = None) -> JSONResponse:
    """Forward an HTTP request to AWS, preserving Authorization header.

    Raises HTTPException on upstream failure and returns a JSONResponse on success.
    """
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    url = _build_aws_url(path)
    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=VERIFY_TLS,
        )
    except requests.RequestException as exc:  # pragma: no cover - network failure path
        logger.error("AWS bridge request failed (%s %s): %s", method, url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream AWS API unavailable",
        ) from exc

    # Try to decode JSON, but fall back to raw text
    try:
        data = resp.json()
    except ValueError:  # non-JSON body
        data = {"raw": resp.text}

    if resp.status_code >= 400:
        logger.warning("AWS bridge returned error %s for %s %s: %s", resp.status_code, method, url, data)
        raise HTTPException(status_code=resp.status_code, detail=data)

    return JSONResponse(status_code=resp.status_code, content=data)


@router.post("/auth/register")
def aws_auth_register(request: Request, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Proxy user registration to AWS.

    The payload shape is passed through untouched so the mobile app can send
    whatever AWS currently expects (e.g. username/email + password).
    """
    return _forward_to_aws("POST", "/auth/register", request=request, json_body=payload)


@router.post("/auth/login")
def aws_auth_login(request: Request, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Proxy user login to AWS so JWTs are issued by the serverless backend."""
    return _forward_to_aws("POST", "/auth/login", request=request, json_body=payload)


@router.get("/user/balance")
def aws_user_balance(request: Request) -> JSONResponse:
    """Proxy balance reads to AWS /user/balance endpoint.

    Query parameters (e.g. userId) and Authorization header are forwarded.
    """
    # Preserve query string by embedding it in the path we hand to _build_aws_url
    query = str(request.url.query)
    path = "/user/balance"
    if query:
        path = f"{path}?{query}"
    return _forward_to_aws("GET", path, request=request)
