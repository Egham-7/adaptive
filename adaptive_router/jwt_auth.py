"""JWT authentication utilities for Modal deployment."""

import os
from typing import Any, Dict

import jwt
from fastapi import HTTPException, Request, status


def verify_jwt_token(request: Request) -> Dict[str, Any]:
    """Verify JWT token from Authorization header.

    Used for authenticating requests to Modal-deployed endpoints.
    Extracts and validates JWT tokens using HS256 algorithm.

    Args:
        request: FastAPI Request object containing headers

    Returns:
        Dict containing the decoded JWT payload with user/service information

    Raises:
        HTTPException 403: If Authorization header is missing or malformed
        HTTPException 401: If token is expired or invalid
        HTTPException 500: If JWT secret is not configured
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
            detail="Invalid authorization header format. Expected: Bearer <token>",
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
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
