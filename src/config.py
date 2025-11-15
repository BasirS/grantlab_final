"""
Configuration management for the Cambio Labs Grant Agent application.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # database configuration
    database_url: str = Field(..., alias="DATABASE_URL")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    # openai configuration
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(default=None, alias="OPENAI_ORG_ID")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_mini_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MINI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    # anthropic fallback configuration
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")

    # authentication configuration
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, alias="JWT_EXPIRATION_MINUTES")
    refresh_token_expiration_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRATION_DAYS")

    # rate limiting configuration
    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=100, alias="RATE_LIMIT_PER_HOUR")
    rate_limit_per_day: int = Field(default=500, alias="RATE_LIMIT_PER_DAY")

    # file upload configuration
    max_upload_size_mb: int = Field(default=10, alias="MAX_UPLOAD_SIZE_MB")
    allowed_file_extensions: str = Field(default=".txt,.docx,.pdf", alias="ALLOWED_FILE_EXTENSIONS")
    upload_directory: str = Field(default="./data/uploads", alias="UPLOAD_DIRECTORY")

    # monitoring configuration
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    enable_langfuse: bool = Field(default=True, alias="ENABLE_LANGFUSE")

    # application settings
    app_name: str = Field(default="Cambio Labs Grant Agent", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8080", alias="CORS_ORIGINS")

    # llm configuration
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    llm_timeout_seconds: int = Field(default=60, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # vector store configuration
    vector_dimension: int = Field(default=1536, alias="VECTOR_DIMENSION")
    vector_similarity_threshold: float = Field(default=0.7, alias="VECTOR_SIMILARITY_THRESHOLD")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")

    # circuit breaker configuration
    circuit_breaker_failure_threshold: int = Field(default=5, alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_timeout_seconds: int = Field(default=60, alias="CIRCUIT_BREAKER_TIMEOUT_SECONDS")
    circuit_breaker_half_open_attempts: int = Field(default=3, alias="CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS")

    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v):
        """
        validating that jwt secret key is at least 32 characters long for security
        """
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v

    @validator("cors_origins")
    def parse_cors_origins(cls, v) -> List[str]:
        """
        parsing comma-separated cors origins into a list
        """
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("allowed_file_extensions")
    def parse_file_extensions(cls, v) -> List[str]:
        """
        parsing comma-separated file extensions into a list
        """
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @property
    def max_upload_size_bytes(self) -> int:
        """
        converting max upload size from megabytes to bytes
        """
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def langfuse_enabled(self) -> bool:
        """
        checking if langfuse monitoring is enabled and configured properly
        """
        return (
            self.enable_langfuse and
            self.langfuse_public_key is not None and
            self.langfuse_secret_key is not None
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    getting cached settings instance to avoid reloading from environment multiple times
    """
    return Settings()


# exporting settings instance for easy import throughout the application
settings = get_settings()
