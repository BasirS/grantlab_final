"""
Database configuration and models for PostgreSQL with pgvector support.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Date, ForeignKey,
    Float, Boolean, Index, JSON, Numeric, text
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from pgvector.sqlalchemy import Vector

from src.config import settings


# creating base class for all database models
class Base(DeclarativeBase):
    """
    base class for all sqlalchemy orm models
    """
    pass


class User(Base):
    """
    user table storing authentication and profile information
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # relationships
    grant_applications = relationship("GrantApplication", back_populates="user", cascade="all, delete-orphan")
    generation_logs = relationship("GenerationLog", back_populates="user", cascade="all, delete-orphan")
    audit_trails = relationship("AuditTrail", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


class GrantApplication(Base):
    """
    grant applications table storing generated grant proposals with embeddings
    """
    __tablename__ = "grant_applications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    funder = Column(String(255), nullable=True)
    funder_agency = Column(String(255), nullable=True)
    opportunity_number = Column(String(100), nullable=True)

    # storing grant content sections
    executive_summary = Column(Text, nullable=True)
    organizational_background = Column(Text, nullable=True)
    problem_statement = Column(Text, nullable=True)
    project_description = Column(Text, nullable=True)
    budget_narrative = Column(Text, nullable=True)
    evaluation_plan = Column(Text, nullable=True)
    full_content = Column(Text, nullable=True)

    # storing vector embedding for full content
    content_embedding = Column(Vector(1536), nullable=True)

    # metadata and status tracking
    status = Column(String(50), default="draft", nullable=False)  # draft, in_progress, completed, submitted
    version = Column(Integer, default=1, nullable=False)
    grant_metadata = Column(JSONB, nullable=True)

    # timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    submitted_at = Column(DateTime, nullable=True)

    # relationships
    user = relationship("User", back_populates="grant_applications")
    generation_logs = relationship("GenerationLog", back_populates="grant", cascade="all, delete-orphan")

    # creating index for vector similarity search
    __table_args__ = (
        Index("idx_grant_embedding", "content_embedding", postgresql_using="ivfflat"),
        Index("idx_grant_status", "status"),
        Index("idx_grant_created", "created_at"),
    )

    def __repr__(self):
        return f"<GrantApplication(id={self.id}, title='{self.title}', status='{self.status}')>"


class Document(Base):
    """
    documents table storing historical grants, rfps, and organizational information
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(500), nullable=False)
    document_type = Column(String(50), nullable=False, index=True)  # past_grant, rfp, org_info, template

    # categorization fields
    program = Column(String(255), nullable=True, index=True)  # e.g., "Journey Platform", "StartUp NYCHA"
    year = Column(Integer, nullable=True, index=True)
    funder = Column(String(255), nullable=True, index=True)
    funder_type = Column(String(100), nullable=True)  # federal, state, local, foundation, corporate
    outcome = Column(String(50), nullable=True)  # awarded, declined, pending
    award_amount = Column(Float, nullable=True)

    # content and embeddings
    content = Column(Text, nullable=False)
    content_embedding = Column(Vector(1536), nullable=True)

    # additional metadata stored as json
    document_metadata = Column(JSONB, nullable=True)

    # file information
    file_path = Column(String(1000), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)

    # processing status
    processed = Column(Boolean, default=False, nullable=False)
    processing_error = Column(Text, nullable=True)

    # timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # creating indexes for vector similarity search and filtering
    __table_args__ = (
        Index("idx_document_embedding", "content_embedding", postgresql_using="ivfflat"),
        Index("idx_document_type", "document_type"),
        Index("idx_document_program", "program"),
        Index("idx_document_funder", "funder"),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', type='{self.document_type}')>"


class GenerationLog(Base):
    """
    generation logs table tracking all llm api calls for monitoring and cost analysis
    """
    __tablename__ = "generation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    grant_id = Column(Integer, ForeignKey("grant_applications.id"), nullable=True, index=True)

    # request details
    prompt = Column(Text, nullable=False)
    system_prompt = Column(Text, nullable=True)
    section_type = Column(String(100), nullable=True)  # executive_summary, problem_statement, etc.

    # response details
    response = Column(Text, nullable=False)
    model = Column(String(100), nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # openai, anthropic

    # usage metrics
    tokens_input = Column(Integer, nullable=True)
    tokens_output = Column(Integer, nullable=True)
    tokens_total = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)

    # quality and outcome
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    user_rating = Column(Integer, nullable=True)  # 1-5 rating from user

    # metadata
    document_metadata = Column(JSONB, nullable=True)

    # timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # relationships
    user = relationship("User", back_populates="generation_logs")
    grant = relationship("GrantApplication", back_populates="generation_logs")

    # creating indexes for analytics queries
    __table_args__ = (
        Index("idx_generation_model", "model"),
        Index("idx_generation_created", "created_at"),
        Index("idx_generation_success", "success"),
    )

    def __repr__(self):
        return f"<GenerationLog(id={self.id}, model='{self.model}', success={self.success})>"


class AuditTrail(Base):
    """
    audit trail table tracking all user actions for security and compliance
    """
    __tablename__ = "audit_trail"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # action details
    action = Column(String(100), nullable=False, index=True)  # create, read, update, delete, login, logout
    resource_type = Column(String(100), nullable=False, index=True)  # user, grant, document, etc.
    resource_id = Column(Integer, nullable=True)

    # request context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    endpoint = Column(String(255), nullable=True)
    method = Column(String(10), nullable=True)  # GET, POST, PUT, DELETE

    # additional details stored as json
    details = Column(JSONB, nullable=True)

    # outcome
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)

    # timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # relationships
    user = relationship("User", back_populates="audit_trails")

    # creating indexes for audit queries
    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_created", "created_at"),
    )

    def __repr__(self):
        return f"<AuditTrail(id={self.id}, action='{self.action}', resource='{self.resource_type}')>"


class GrantBudget(Base):
    """
    structured budget information from past grants for preventing hallucinations
    """
    __tablename__ = "grant_budgets"

    id = Column(Integer, primary_key=True, index=True)
    grant_id = Column(Integer, ForeignKey("grant_applications.id"), nullable=True)

    # grant details
    program = Column(String(255), nullable=False, index=True)  # Journey, StartUp NYCHA, etc.
    funder = Column(String(255), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    outcome = Column(String(50), nullable=False, index=True)  # awarded, rejected, pending

    # budget information (structured for exact retrieval)
    total_amount = Column(Numeric(10, 2), nullable=False)
    budget_items = Column(JSONB, nullable=False)  # [{category, amount, description}]
    indirect_rate = Column(Numeric(5, 2), nullable=True)  # percentage
    cost_sharing = Column(Numeric(10, 2), nullable=True)

    # metadata
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # relationships
    grant = relationship("GrantApplication", backref="budgets")

    __table_args__ = (
        Index("idx_budget_program_year", "program", "year"),
        Index("idx_budget_outcome", "outcome"),
    )

    def __repr__(self):
        return f"<GrantBudget(id={self.id}, program='{self.program}', amount=${self.total_amount})>"


class TeamMember(Base):
    """
    structured team member information for accurate credential citation
    """
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True, index=True)

    # personal information
    name = Column(String(255), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    role = Column(String(100), nullable=False, index=True)  # executive, instructor, developer, etc.

    # credentials (structured for exact retrieval)
    education = Column(JSONB, nullable=True)  # [{degree, institution, year}]
    certifications = Column(JSONB, nullable=True)  # [{name, issuer, year}]
    experience_years = Column(Integer, nullable=True)
    expertise_areas = Column(ARRAY(String), nullable=True)  # programming languages, teaching methods, etc.

    # professional history
    bio = Column(Text, nullable=True)
    linkedin_url = Column(String(500), nullable=True)
    publications = Column(JSONB, nullable=True)  # [{title, venue, year}]

    # availability
    is_active = Column(Boolean, default=True, nullable=False)
    start_date = Column(Date, nullable=True)

    # metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_team_member_role", "role"),
        Index("idx_team_member_active", "is_active"),
    )

    def __repr__(self):
        return f"<TeamMember(id={self.id}, name='{self.name}', role='{self.role}')>"


class ProgramOutcome(Base):
    """
    structured program outcomes and metrics for accurate impact reporting
    """
    __tablename__ = "program_outcomes"

    id = Column(Integer, primary_key=True, index=True)

    # program details
    program = Column(String(255), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    cohort = Column(String(100), nullable=True)

    # participation metrics (structured for exact retrieval)
    total_participants = Column(Integer, nullable=False)
    completion_rate = Column(Numeric(5, 2), nullable=True)  # percentage
    demographics = Column(JSONB, nullable=True)  # {bipoc_percentage, age_range, etc.}

    # outcome metrics
    outcomes = Column(JSONB, nullable=False)  # {job_placements, businesses_launched, avg_income_increase}
    success_stories = Column(JSONB, nullable=True)  # [{participant, outcome, details}]

    # impact data
    economic_impact = Column(Numeric(12, 2), nullable=True)  # total dollars
    community_impact = Column(Text, nullable=True)

    # verification
    data_source = Column(String(255), nullable=True)  # survey, report, tracking system
    verified = Column(Boolean, default=False, nullable=False)
    verification_date = Column(Date, nullable=True)

    # metadata
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_program_outcome_year", "program", "year"),
        Index("idx_program_outcome_verified", "verified"),
    )

    def __repr__(self):
        return f"<ProgramOutcome(id={self.id}, program='{self.program}', year={self.year})>"


class QualityReview(Base):
    """
    quality review records for generated content tracking hallucinations and compliance
    """
    __tablename__ = "quality_reviews"

    id = Column(Integer, primary_key=True, index=True)
    grant_id = Column(Integer, ForeignKey("grant_applications.id"), nullable=False)
    generation_log_id = Column(Integer, ForeignKey("generation_logs.id"), nullable=True)

    # review details
    section = Column(String(100), nullable=False, index=True)
    review_type = Column(String(50), nullable=False, index=True)  # faithfulness, compliance, format

    # quality scores
    faithfulness_score = Column(Numeric(4, 3), nullable=True)  # 0.000-1.000
    compliance_score = Column(Numeric(4, 3), nullable=True)
    similarity_score = Column(Numeric(4, 3), nullable=True)

    # detected issues
    hallucinations_detected = Column(JSONB, nullable=True)  # [{claim, type, evidence}]
    compliance_issues = Column(JSONB, nullable=True)  # [{requirement, status, details}]
    flagged_for_review = Column(Boolean, default=False, nullable=False)

    # review outcome
    approved = Column(Boolean, nullable=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    # metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # relationships
    grant = relationship("GrantApplication", backref="quality_reviews")
    generation_log = relationship("GenerationLog", backref="quality_reviews")
    reviewer = relationship("User", backref="quality_reviews")

    __table_args__ = (
        Index("idx_quality_review_section", "section"),
        Index("idx_quality_review_flagged", "flagged_for_review"),
        Index("idx_quality_review_type", "review_type"),
    )

    def __repr__(self):
        return f"<QualityReview(id={self.id}, section='{self.section}', flagged={self.flagged_for_review})>"


class ComplianceRequirement(Base):
    """
    extracted requirements from rfps for automated compliance checking
    """
    __tablename__ = "compliance_requirements"

    id = Column(Integer, primary_key=True, index=True)
    grant_id = Column(Integer, ForeignKey("grant_applications.id"), nullable=False)

    # requirement details
    requirement_text = Column(Text, nullable=False)
    requirement_type = Column(String(50), nullable=False, index=True)  # eligibility, content, format, submission
    section = Column(String(100), nullable=False, index=True)
    is_mandatory = Column(Boolean, default=True, nullable=False)

    # compliance tracking
    addressed = Column(Boolean, default=False, nullable=False)
    evidence_location = Column(String(255), nullable=True)  # section where addressed
    compliance_notes = Column(Text, nullable=True)

    # metadata
    extracted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    verified_at = Column(DateTime, nullable=True)

    # relationships
    grant = relationship("GrantApplication", backref="requirements")

    __table_args__ = (
        Index("idx_requirement_type", "requirement_type"),
        Index("idx_requirement_addressed", "addressed"),
    )

    def __repr__(self):
        return f"<ComplianceRequirement(id={self.id}, type='{self.requirement_type}', addressed={self.addressed})>"


# creating async engine with connection pooling
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.debug,
    future=True,
)

# creating async session maker for database operations
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """
    getting database session for dependency injection in fastapi routes
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """
    initializing database by creating all tables and extensions
    """
    async with engine.begin() as conn:
        # enabling pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # creating all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """
    closing database connection pool gracefully
    """
    await engine.dispose()
