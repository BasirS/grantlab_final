"""
Pydantic schemas for request and response validation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, validator


# ==================== Authentication Schemas ====================

class UserBase(BaseModel):
    """
    base user schema with common fields
    """
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """
    schema for user registration
    """
    password: str = Field(..., min_length=8, max_length=100)

    @validator("password")
    def validate_password(cls, v):
        """
        validating password meets security requirements
        """
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        return v


class UserLogin(BaseModel):
    """
    schema for user login
    """
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """
    schema for user response
    """
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """
    schema for authentication token response
    """
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """
    schema for refresh token request
    """
    refresh_token: str


# ==================== Document Schemas ====================

class DocumentBase(BaseModel):
    """
    base document schema
    """
    filename: str
    document_type: str = Field(..., description="past_grant, rfp, org_info, template")
    program: Optional[str] = None
    year: Optional[int] = None
    funder: Optional[str] = None
    funder_type: Optional[str] = None
    outcome: Optional[str] = None
    award_amount: Optional[float] = None


class DocumentCreate(DocumentBase):
    """
    schema for creating document
    """
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentUpload(BaseModel):
    """
    schema for document upload metadata
    """
    document_type: str
    program: Optional[str] = None
    year: Optional[int] = None
    funder: Optional[str] = None
    funder_type: Optional[str] = None
    outcome: Optional[str] = None
    award_amount: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    """
    schema for document response
    """
    id: int
    content: Optional[str] = Field(None, description="Content excluded by default for performance")
    processed: bool
    processing_error: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentSearchRequest(BaseModel):
    """
    schema for document search request
    """
    query: str
    document_type: Optional[str] = None
    program: Optional[str] = None
    funder: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class DocumentSearchResult(BaseModel):
    """
    schema for document search result with similarity score
    """
    document: DocumentResponse
    similarity_score: float


class DocumentBatchProcess(BaseModel):
    """
    schema for batch document processing request
    """
    document_ids: List[int]


class DocumentBatchResult(BaseModel):
    """
    schema for batch processing result
    """
    total: int
    success: List[int]
    failed: List[Dict[str, Any]]


# ==================== Grant Application Schemas ====================

class GrantApplicationBase(BaseModel):
    """
    base grant application schema
    """
    title: str
    funder: Optional[str] = None
    funder_agency: Optional[str] = None
    opportunity_number: Optional[str] = None


class GrantApplicationCreate(GrantApplicationBase):
    """
    schema for creating grant application
    """
    metadata: Optional[Dict[str, Any]] = None


class GrantSectionUpdate(BaseModel):
    """
    schema for updating individual grant section
    """
    executive_summary: Optional[str] = None
    organizational_background: Optional[str] = None
    problem_statement: Optional[str] = None
    project_description: Optional[str] = None
    budget_narrative: Optional[str] = None
    evaluation_plan: Optional[str] = None


class GrantApplicationUpdate(GrantApplicationBase):
    """
    schema for updating grant application
    """
    executive_summary: Optional[str] = None
    organizational_background: Optional[str] = None
    problem_statement: Optional[str] = None
    project_description: Optional[str] = None
    budget_narrative: Optional[str] = None
    evaluation_plan: Optional[str] = None
    full_content: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GrantApplicationResponse(GrantApplicationBase):
    """
    schema for grant application response
    """
    id: int
    user_id: int
    executive_summary: Optional[str] = None
    organizational_background: Optional[str] = None
    problem_statement: Optional[str] = None
    project_description: Optional[str] = None
    budget_narrative: Optional[str] = None
    evaluation_plan: Optional[str] = None
    full_content: Optional[str] = None
    status: str
    version: int
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class GrantGenerationRequest(BaseModel):
    """
    schema for grant generation request
    """
    grant_id: int
    section: str = Field(..., description="executive_summary, organizational_background, etc.")
    context: Optional[str] = Field(None, description="Additional context for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, ge=100, le=4096)
    use_rag: bool = Field(default=True, description="Whether to use RAG for context")
    similar_documents_count: int = Field(default=3, ge=1, le=10)


class GrantGenerationResponse(BaseModel):
    """
    schema for grant generation response
    """
    grant_id: int
    section: str
    content: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: int
    similar_documents_used: int


class GrantSearchRequest(BaseModel):
    """
    schema for searching similar grants
    """
    query: str
    status: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class GrantSearchResult(BaseModel):
    """
    schema for grant search result with similarity score
    """
    grant: GrantApplicationResponse
    similarity_score: float


# ==================== Generation Log Schemas ====================

class GenerationLogResponse(BaseModel):
    """
    schema for generation log response
    """
    id: int
    user_id: int
    grant_id: Optional[int] = None
    section_type: Optional[str] = None
    model: str
    provider: str
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_total: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    success: bool
    error_message: Optional[str] = None
    user_rating: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class GenerationLogFilter(BaseModel):
    """
    schema for filtering generation logs
    """
    user_id: Optional[int] = None
    grant_id: Optional[int] = None
    provider: Optional[str] = None
    success: Optional[bool] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class GenerationStats(BaseModel):
    """
    schema for generation statistics
    """
    total_generations: int
    successful_generations: int
    failed_generations: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    provider_breakdown: Dict[str, int]
    model_breakdown: Dict[str, int]


# ==================== Audit Trail Schemas ====================

class AuditTrailResponse(BaseModel):
    """
    schema for audit trail response
    """
    id: int
    user_id: Optional[int] = None
    action: str
    resource_type: str
    resource_id: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    success: bool
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AuditTrailFilter(BaseModel):
    """
    schema for filtering audit trails
    """
    user_id: Optional[int] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    success: Optional[bool] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


# ==================== Health and Status Schemas ====================

class HealthResponse(BaseModel):
    """
    schema for health check response
    """
    status: str
    timestamp: datetime
    version: str
    database_connected: bool
    circuit_breaker_status: Dict[str, str]


class ErrorResponse(BaseModel):
    """
    schema for error response
    """
    error: str
    detail: Optional[str] = None
    timestamp: datetime


# ==================== Word Export Schemas ====================

class GrantExportResponse(BaseModel):
    """
    schema for word document export response metadata
    """
    grant_id: int
    filename: str
    export_format: str = "docx"
    file_size_bytes: Optional[int] = None
    sections_included: int
    generated_at: datetime
    include_metadata: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "grant_id": 1,
                "filename": "Cambio_Labs_Grant_Application.docx",
                "export_format": "docx",
                "file_size_bytes": 45678,
                "sections_included": 6,
                "generated_at": "2025-11-15T14:30:00",
                "include_metadata": True
            }
        }


# ==================== Pagination Schemas ====================

class PaginatedResponse(BaseModel):
    """
    generic schema for paginated responses
    """
    items: List[Any]
    total: int
    limit: int
    offset: int
    has_more: bool
