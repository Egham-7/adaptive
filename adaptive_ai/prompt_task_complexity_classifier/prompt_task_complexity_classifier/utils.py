"""Utility functions for the prompt task complexity classifier."""

import os
from typing import Dict, Any
import jwt
from fastapi import HTTPException, status, Request


def verify_jwt_token(request: Request) -> Dict[str, Any]:
    """Verify JWT token from Authorization header.

    Args:
        request: FastAPI Request object containing headers

    Returns:
        Dict containing the decoded JWT payload

    Raises:
        HTTPException: If token is missing, invalid format, expired, or invalid
    """
    # Extract Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authorization header missing",
        )

    # Check Bearer token format
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authorization header format",
        )

    token = auth_header.split(" ", 1)[1]

    # Get JWT secret from environment
    jwt_secret = os.environ.get("jwt_auth")
    if not jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured",
        )

    try:
        # Verify and decode JWT token
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        return payload  # type: ignore[no-any-return]
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
