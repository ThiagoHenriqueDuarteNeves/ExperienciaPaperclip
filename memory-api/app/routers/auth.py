"""Auth routes — multi-user profiles with server-validated PIN."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.auth import hash_pin, issue_token, user_from_authorization, verify_pin
from app.pgvector_client import create_user_profile, get_user_profile
from app.rate_limit import limiter
from app.schemas import LoginRequest, RegisterRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])


@router.post("/auth/register")
@limiter.limit("5/minute")
async def auth_register_endpoint(request: Request, req: RegisterRequest):
    existing = await get_user_profile(req.user_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="user_id already taken")
    pin_hash, pin_salt = hash_pin(req.pin)
    display_name = req.display_name or req.user_id
    await create_user_profile(req.user_id, display_name, pin_hash, pin_salt)
    logger.info("auth: registered user_id=%s", req.user_id)
    token = issue_token(req.user_id)
    return {"token": token, "user_id": req.user_id, "display_name": display_name}


@router.post("/auth/login")
@limiter.limit("6/minute")
async def auth_login_endpoint(request: Request, req: LoginRequest):
    profile = await get_user_profile(req.user_id)
    if profile is None or not verify_pin(req.pin, profile["pin_hash"], profile["pin_salt"]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = issue_token(req.user_id)
    return {
        "token": token,
        "user_id": req.user_id,
        "display_name": profile.get("display_name") or req.user_id,
    }


@router.get("/auth/me")
async def auth_me_endpoint(request: Request):
    user_id = user_from_authorization(request.headers.get("authorization"))
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing token")
    profile = await get_user_profile(user_id)
    return {
        "user_id": user_id,
        "display_name": (profile or {}).get("display_name") or user_id,
    }
