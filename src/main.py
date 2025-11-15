"""
FastAPI main application with middleware, routes, and startup/shutdown handlers.
"""

import time
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

from src.config import settings
from src.core.database import init_db, close_db, engine
from sqlalchemy import text
from src.core.llm import get_llm_client
from src.api.routes import auth, documents, grants
from src.api.models.schemas import HealthResponse, ErrorResponse


# configuring structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


# creating rate limiter storage (in-memory for simplicity, use Redis in production)
rate_limit_storage: Dict[str, list] = {}


def check_rate_limit(user_id: str, limit: int, window_seconds: int) -> bool:
    """
    checking if user has exceeded rate limit
    """
    now = time.time()
    key = f"{user_id}:{window_seconds}"

    # initializing user's request history if not exists
    if key not in rate_limit_storage:
        rate_limit_storage[key] = []

    # removing old requests outside the time window
    rate_limit_storage[key] = [
        timestamp for timestamp in rate_limit_storage[key]
        if now - timestamp < window_seconds
    ]

    # checking if limit exceeded
    if len(rate_limit_storage[key]) >= limit:
        return False

    # adding current request
    rate_limit_storage[key].append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    managing application lifecycle with startup and shutdown events
    """
    # startup: initializing database
    logger.info("application_starting", version=settings.app_version)

    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        raise

    # initializing llm client to warm up connections
    try:
        llm_client = get_llm_client()
        logger.info("llm_client_initialized")
    except Exception as e:
        logger.error("llm_client_initialization_failed", error=str(e))

    yield

    # shutdown: closing database connections
    logger.info("application_shutting_down")

    try:
        await close_db()
        logger.info("database_closed")
    except Exception as e:
        logger.error("database_close_failed", error=str(e))


# creating fastapi application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered grant writing assistant for Cambio Labs",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)


# adding cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# adding gzip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    logging all requests with timing information
    """
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}"

    # adding request id to logger context
    logger_ctx = logger.bind(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )

    logger_ctx.info("request_started")

    try:
        response = await call_next(request)

        # calculating request duration
        duration_ms = int((time.time() - start_time) * 1000)

        logger_ctx.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # adding custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(duration_ms)

        return response

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)

        logger_ctx.error(
            "request_failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=duration_ms,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(e) if settings.debug else None,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    implementing rate limiting per user based on api token
    """
    # skipping rate limiting for health check and auth endpoints
    if request.url.path in ["/health", "/auth/login", "/auth/register"]:
        return await call_next(request)

    # extracting user id from authorization header if present
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        user_id = token[:16]  # using first 16 chars as identifier

        # checking rate limits
        if not check_rate_limit(user_id, settings.rate_limit_per_minute, 60):
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                path=request.url.path,
                limit=settings.rate_limit_per_minute,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {settings.rate_limit_per_minute} requests per minute allowed",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    response = await call_next(request)
    return response


# adding exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    handling http exceptions with custom error response
    """
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": str(exc.detail),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    handling request validation errors
    """
    logger.warning(
        "validation_error",
        errors=exc.errors(),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc.errors()),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    handling all uncaught exceptions
    """
    logger.error(
        "unhandled_exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# including routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(grants.router)


# health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    checking application health and service status
    """
    try:
        # checking database connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error("health_check_db_failed", error=str(e))
        db_connected = False

    # getting circuit breaker status
    llm_client = get_llm_client()
    circuit_status = llm_client.get_circuit_status()

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        database_connected=db_connected,
        circuit_breaker_status=circuit_status,
    )


# root endpoint
@app.get("/")
async def root():
    """
    returning api information
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "documentation": "/docs" if settings.debug else "disabled",
    }


# running application with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
