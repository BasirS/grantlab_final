"""
Authentication routes for user registration, login, and token management.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext
import structlog

from src.core.database import get_db, User
from src.api.models.schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    RefreshTokenRequest,
)
from src.config import settings


# initializing structured logger
logger = structlog.get_logger(__name__)

# initializing password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# initializing http bearer security scheme
security = HTTPBearer()

# creating router for authentication endpoints
router = APIRouter(prefix="/auth", tags=["Authentication"])


def hash_password(password: str) -> str:
    """
    hashing password using bcrypt
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    verifying plain password against hashed password
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int, email: str) -> str:
    """
    creating jwt access token for user
    """
    expires_delta = timedelta(minutes=settings.jwt_expiration_minutes)
    expire = datetime.utcnow() + expires_delta

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    logger.debug(
        "access_token_created",
        user_id=user_id,
        expires_at=expire.isoformat(),
    )

    return encoded_jwt


def create_refresh_token(user_id: int, email: str) -> str:
    """
    creating jwt refresh token for user
    """
    expires_delta = timedelta(days=settings.refresh_token_expiration_days)
    expire = datetime.utcnow() + expires_delta

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    logger.debug(
        "refresh_token_created",
        user_id=user_id,
        expires_at=expire.isoformat(),
    )

    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    getting current authenticated user from jwt token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials

        # decoding jwt token
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")

        if user_id is None or token_type != "access":
            logger.warning("invalid_token", reason="missing_user_id_or_wrong_type")
            raise credentials_exception

        # fetching user from database
        result = await db.execute(
            select(User).where(User.id == int(user_id))
        )
        user = result.scalar_one_or_none()

        if user is None:
            logger.warning("user_not_found", user_id=user_id)
            raise credentials_exception

        if not user.is_active:
            logger.warning("inactive_user", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive",
            )

        logger.debug("user_authenticated", user_id=user.id, email=user.email)
        return user

    except JWTError as e:
        logger.error("jwt_error", error=str(e))
        raise credentials_exception
    except Exception as e:
        logger.error("auth_error", error=str(e), error_type=type(e).__name__)
        raise credentials_exception


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    registering new user account
    """
    try:
        logger.info("registering_user", email=user_data.email)

        # checking if user already exists
        result = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.warning("registration_failed", email=user_data.email, reason="email_exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # creating new user
        hashed_password = hash_password(user_data.password)
        new_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_active=True,
            is_superuser=False,
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # creating tokens
        access_token = create_access_token(new_user.id, new_user.email)
        refresh_token = create_refresh_token(new_user.id, new_user.email)

        logger.info("user_registered", user_id=new_user.id, email=new_user.email)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expiration_minutes * 60,
            user=UserResponse.from_orm(new_user),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("registration_error", email=user_data.email, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user",
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    logging in user with email and password
    """
    try:
        logger.info("login_attempt", email=credentials.email)

        # fetching user by email
        result = await db.execute(
            select(User).where(User.email == credentials.email)
        )
        user = result.scalar_one_or_none()

        if not user:
            logger.warning("login_failed", email=credentials.email, reason="user_not_found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        # verifying password
        if not verify_password(credentials.password, user.hashed_password):
            logger.warning("login_failed", email=credentials.email, reason="invalid_password")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        # checking if user is active
        if not user.is_active:
            logger.warning("login_failed", email=credentials.email, reason="inactive_account")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive",
            )

        # creating tokens
        access_token = create_access_token(user.id, user.email)
        refresh_token = create_refresh_token(user.id, user.email)

        logger.info("login_successful", user_id=user.id, email=user.email)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expiration_minutes * 60,
            user=UserResponse.from_orm(user),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("login_error", email=credentials.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login",
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    refreshing access token using refresh token
    """
    try:
        logger.info("refreshing_token")

        # decoding refresh token
        payload = jwt.decode(
            request.refresh_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")

        if user_id is None or token_type != "refresh":
            logger.warning("refresh_failed", reason="invalid_token_type")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # fetching user from database
        result = await db.execute(
            select(User).where(User.id == int(user_id))
        )
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            logger.warning("refresh_failed", user_id=user_id, reason="user_not_found_or_inactive")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # creating new access token
        access_token = create_access_token(user.id, user.email)

        logger.info("token_refreshed", user_id=user.id)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expiration_minutes * 60,
            user=UserResponse.from_orm(user),
        )

    except JWTError as e:
        logger.error("jwt_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("refresh_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token",
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """
    getting current authenticated user information
    """
    logger.debug("fetching_user_info", user_id=current_user.id)
    return UserResponse.from_orm(current_user)


@router.get("/verify")
async def verify_token(
    current_user: User = Depends(get_current_user),
):
    """
    verifying if token is valid
    """
    logger.debug("token_verified", user_id=current_user.id)
    return {"valid": True, "user_id": current_user.id, "email": current_user.email}
