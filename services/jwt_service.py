"""
JWT Service for Neo4j GraphQL Authentication

Converts PAPR Memory's existing authentication (API keys, bearer tokens, session tokens)
into JWT tokens that Neo4j GraphQL can validate.

This is a translation layer - we keep using API keys for client auth,
but generate JWTs for Neo4j's @authorization directives.
"""

from datetime import datetime, timedelta, UTC
from typing import Optional
import jwt
from os import environ as env
import base64
from pathlib import Path


class JWTService:
    """
    Generate JWT tokens for Neo4j GraphQL from existing auth.

    Usage:
        jwt_service = JWTService()
        token = jwt_service.generate_token(
            user_id="user_abc123",
            workspace_id="ws_xyz789"
        )
    """

    def __init__(self):
        self.algorithm = "RS256"  # RSA signing
        
        # Use environment-specific issuer URL
        # This must match the JWKS URL that Neo4j is configured to use
        base_url = env.get("PARSE_SERVER_URL", "https://memory.papr.ai")
        # Remove trailing slash if present
        self.issuer = base_url.rstrip("/")
        self.audience = "neo4j-graphql"

        # Load private key for signing JWTs
        # Try environment variable first (base64-encoded), then file
        private_key_b64 = env.get("JWT_PRIVATE_KEY")
        if private_key_b64:
            try:
                self.private_key = base64.b64decode(private_key_b64).decode('utf-8')
            except Exception as e:
                raise ValueError(f"Failed to decode JWT_PRIVATE_KEY from environment: {e}")
        else:
            # Fall back to file path
            private_key_path = env.get(
                "JWT_PRIVATE_KEY_PATH",
                str(Path(__file__).parent.parent / "keys" / "jwt-private.pem")
            )
            try:
                with open(private_key_path, 'r') as f:
                    self.private_key = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"JWT private key not found at {private_key_path}. "
                    "Please set JWT_PRIVATE_KEY environment variable (base64-encoded) or "
                    "generate RSA keys using: openssl genrsa -out keys/jwt-private.pem 2048"
                )

    def generate_token(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
        end_user_id: Optional[str] = None,
        roles: Optional[list[str]] = None,
        expires_in_minutes: int = 60
    ) -> str:
        """
        Generate a JWT token for Neo4j GraphQL authorization.

        Args:
            user_id: User's unique identifier (required for @authorization filters)
            workspace_id: Workspace identifier (optional, for workspace-level data)
            end_user_id: End user identifier (optional)
            roles: User roles (optional, for role-based access)
            expires_in_minutes: Token expiration time

        Returns:
            Signed JWT token string

        Example:
            jwt_service = JWTService()
            token = jwt_service.generate_token(
                user_id="user_abc123",
                workspace_id="ws_xyz789"
            )

            # Token will have claims:
            # {
            #   "sub": "user_abc123",
            #   "user_id": "user_abc123",
            #   "workspace_id": "ws_xyz789",
            #   "exp": 1735564800,
            #   ...
            # }
        """
        now = datetime.now(UTC)
        expiration = now + timedelta(minutes=expires_in_minutes)

        # Build JWT payload with claims Neo4j will use in @authorization directives
        payload = {
            "sub": user_id,              # Standard JWT subject claim
            "user_id": user_id,          # Custom claim for @authorization
            "workspace_id": workspace_id, # Custom claim for @authorization
            "end_user_id": end_user_id,
            "roles": roles or [],
            "iss": self.issuer,          # Issuer
            "aud": self.audience,        # Audience
            "exp": int(expiration.timestamp()),  # Expiration
            "iat": int(now.timestamp())   # Issued at
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Sign and return token
        token = jwt.encode(
            payload,
            self.private_key,
            algorithm=self.algorithm
        )

        return token

    def verify_token(self, token: str) -> dict:
        """
        Verify a JWT token (for testing purposes).

        In production, Neo4j will verify tokens using our JWKS endpoint.
        This method is just for local testing.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict

        Raises:
            jwt.ExpiredSignatureError: If token is expired
            jwt.InvalidTokenError: If token is invalid
        """
        # Load public key - try env var first, then file
        public_key_b64 = env.get("JWT_PUBLIC_KEY")
        if public_key_b64:
            try:
                public_key = base64.b64decode(public_key_b64).decode('utf-8')
            except Exception as e:
                raise ValueError(f"Failed to decode JWT_PUBLIC_KEY from environment: {e}")
        else:
            public_key_path = env.get(
                "JWT_PUBLIC_KEY_PATH",
                str(Path(__file__).parent.parent / "keys" / "jwt-public.pem")
            )
            with open(public_key_path, 'r') as f:
                public_key = f.read()

        # Verify and decode
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[self.algorithm],
            audience=self.audience,
            issuer=self.issuer
        )

        return payload


# Singleton instance
_jwt_service = None


def get_jwt_service() -> JWTService:
    """Get singleton JWT service instance"""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService()
    return _jwt_service
