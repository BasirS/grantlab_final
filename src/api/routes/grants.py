"""
Grant application routes for creating, generating, and managing grant proposals.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import structlog

from src.core.database import get_db, User, GrantApplication, GenerationLog, Document
from src.core.vector_store import get_vector_store
from src.core.llm import get_llm_client
from src.services.rag_engine import get_rag_engine
from src.services.knowledge_base import get_knowledge_base
from src.services.cost_optimizer import (
    CostTracker,
    SmartModelRouter,
    PromptCacheManager,
    SemanticCache,
    BatchProcessor,
)
from src.services.quality_assurance import (
    get_quality_assurance_system,
    GroundingPromptBuilder,
    StructuredDataRetriever,
)
from src.api.routes.auth import get_current_user
from src.api.models.schemas import (
    GrantApplicationCreate,
    GrantApplicationUpdate,
    GrantApplicationResponse,
    GrantSectionUpdate,
    GrantGenerationRequest,
    GrantGenerationResponse,
    GrantSearchRequest,
    GrantSearchResult,
    GenerationLogResponse,
    GenerationLogFilter,
    GenerationStats,
)
from src.config import settings
from src.services.word_export import get_word_exporter


# initializing structured logger
logger = structlog.get_logger(__name__)

# creating router for grant endpoints
router = APIRouter(prefix="/grants", tags=["Grants"])

# initializing cost optimization components
cost_tracker = CostTracker()
model_router = SmartModelRouter()
cache_manager = PromptCacheManager()
semantic_cache = SemanticCache()
batch_processor = BatchProcessor()


def _determine_task_type(section: str) -> str:
    """
    determining task complexity type based on section name
    """
    # mapping sections to task types for smart routing
    simple_sections = ["budget", "timeline", "references"]
    medium_sections = ["executive_summary", "project_description", "methodology"]
    complex_sections = ["needs_statement", "evaluation_plan", "sustainability_plan"]

    section_lower = section.lower()

    if any(s in section_lower for s in simple_sections):
        return "simple"
    elif any(s in section_lower for s in complex_sections):
        return "complex"
    else:
        return "medium"


async def retrieve_similar_documents_enhanced(
    query: str,
    db: AsyncSession,
    count: int = 3,
    filters: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[Any]]:
    """
    retrieving similar documents using enhanced rag engine with hybrid search and reranking
    """
    try:
        rag_engine = get_rag_engine()

        # retrieving with hybrid search and reranking
        context, results = await rag_engine.retrieve_with_context(
            session=db,
            query=query,
            filters=filters or {'document_type': 'past_grant', 'outcome': 'awarded'},
            top_k=count * 4,  # retrieving more for better reranking
            rerank_top_k=count,
            include_citations=True,
        )

        logger.info(
            "enhanced_rag_retrieval",
            results_count=len(results),
            context_length=len(context),
        )

        return context, results

    except Exception as e:
        logger.error("rag_retrieval_error", error=str(e))
        return "", []


async def build_generation_prompt_enhanced(
    section: str,
    context: Optional[str],
    rag_context: str,
    db: AsyncSession,
) -> tuple[str, str]:
    """
    building system and user prompts using knowledge base and rag context
    """
    # getting organizational context from knowledge base
    kb = get_knowledge_base()
    org_context = await kb.get_organizational_context(
        session=db,
        include_programs=True,
        include_team=False,
    )

    # building system prompt with comprehensive organizational context
    system_prompt = f"""You are an expert grant writer for Cambio Labs, writing in a natural, authentic voice.

{org_context}

You are writing the {section} section of a grant application. Make it authentic to Cambio Labs' voice and mission."""

    # building user prompt with rag context from similar grants
    user_prompt = f"""Write the {section} section for this grant application.

RELEVANT EXAMPLES FROM PAST SUCCESSFUL GRANTS:
{rag_context}

ADDITIONAL CONTEXT FOR THIS GRANT:
{context if context else 'None provided'}

Write a compelling {section} that:
- Matches Cambio Labs' authentic voice and mission
- Includes specific programs (Journey, StartUp NYCHA, RETI, etc.)
- References concrete metrics and outcomes
- Uses natural, conversational language
- Avoids deficit framing and AI buzzwords
- Draws from the examples above while being original"""

    return system_prompt, user_prompt


@router.post("/", response_model=GrantApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_grant(
    grant_data: GrantApplicationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    creating new grant application
    """
    try:
        logger.info(
            "creating_grant",
            title=grant_data.title,
            user_id=current_user.id,
        )

        grant = GrantApplication(
            user_id=current_user.id,
            title=grant_data.title,
            funder=grant_data.funder,
            funder_agency=grant_data.funder_agency,
            opportunity_number=grant_data.opportunity_number,
            status="draft",
            version=1,
            metadata=grant_data.metadata,
        )

        db.add(grant)
        await db.commit()
        await db.refresh(grant)

        logger.info("grant_created", grant_id=grant.id, title=grant.title)

        return GrantApplicationResponse.from_orm(grant)

    except Exception as e:
        logger.error("create_grant_error", error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create grant application",
        )


@router.get("/", response_model=List[GrantApplicationResponse])
async def list_grants(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    listing grant applications for current user
    """
    try:
        logger.debug("listing_grants", user_id=current_user.id, limit=limit)

        query = select(GrantApplication).where(GrantApplication.user_id == current_user.id)

        if status_filter:
            query = query.where(GrantApplication.status == status_filter)

        query = query.order_by(GrantApplication.created_at.desc()).limit(limit).offset(offset)

        result = await db.execute(query)
        grants = result.scalars().all()

        logger.info("grants_listed", count=len(grants), user_id=current_user.id)

        return [GrantApplicationResponse.from_orm(grant) for grant in grants]

    except Exception as e:
        logger.error("list_grants_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list grants",
        )


@router.get("/{grant_id}", response_model=GrantApplicationResponse)
async def get_grant(
    grant_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    getting specific grant application
    """
    try:
        result = await db.execute(
            select(GrantApplication).where(
                GrantApplication.id == grant_id,
                GrantApplication.user_id == current_user.id,
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        return GrantApplicationResponse.from_orm(grant)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_grant_error", grant_id=grant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get grant application",
        )


@router.put("/{grant_id}", response_model=GrantApplicationResponse)
async def update_grant(
    grant_id: int,
    grant_update: GrantApplicationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    updating grant application
    """
    try:
        logger.info("updating_grant", grant_id=grant_id, user_id=current_user.id)

        result = await db.execute(
            select(GrantApplication).where(
                GrantApplication.id == grant_id,
                GrantApplication.user_id == current_user.id,
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        # updating fields that are provided
        update_data = grant_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(grant, field, value)

        await db.commit()
        await db.refresh(grant)

        logger.info("grant_updated", grant_id=grant_id)

        return GrantApplicationResponse.from_orm(grant)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_grant_error", grant_id=grant_id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update grant application",
        )


@router.post("/generate", response_model=GrantGenerationResponse)
async def generate_grant_section(
    generation_request: GrantGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    generating grant section using llm with rag
    """
    start_time = datetime.utcnow()

    try:
        logger.info(
            "generating_section",
            grant_id=generation_request.grant_id,
            section=generation_request.section,
            user_id=current_user.id,
        )

        # fetching grant
        result = await db.execute(
            select(GrantApplication).where(
                GrantApplication.id == generation_request.grant_id,
                GrantApplication.user_id == current_user.id,
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        # checking semantic cache first
        cache_key = f"{grant.title}_{generation_request.section}_{generation_request.context or ''}"
        cached_response = await semantic_cache.get(cache_key)

        if cached_response:
            logger.info("semantic_cache_hit", section=generation_request.section)
            generated_content = cached_response
            llm_response = {
                "content": cached_response,
                "model": "cached",
                "provider": "cache",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": 0,
            }
            similar_docs_count = 0

            # tracking cache hit
            cost_tracker.track_request(
                model="cached",
                tokens_input=0,
                tokens_output=0,
                latency_ms=0,
                cache_hit=True,
                semantic_cache_hit=True,
            )
        else:
            # retrieving similar documents using enhanced rag engine
            rag_context = ""
            similar_docs_count = 0

            if generation_request.use_rag:
                query = f"{grant.title} {generation_request.section} {generation_request.context or ''}"

                # using enhanced rag engine with hybrid search and reranking
                rag_context, similar_results = await retrieve_similar_documents_enhanced(
                    query=query,
                    db=db,
                    count=generation_request.similar_documents_count,
                    filters={'document_type': 'past_grant', 'outcome': 'awarded'},
                )

                similar_docs_count = len(similar_results)
                logger.info(
                    "enhanced_rag_retrieved",
                    count=similar_docs_count,
                    context_length=len(rag_context),
                )

            # building prompts with enhanced knowledge base integration
            system_prompt, user_prompt = await build_generation_prompt_enhanced(
                section=generation_request.section,
                context=generation_request.context,
                rag_context=rag_context,
                db=db,
            )

            # using smart model routing based on task complexity
            task_type = _determine_task_type(generation_request.section)
            selected_model = model_router.route_to_model(task_type)

            logger.info(
                "smart_routing",
                section=generation_request.section,
                task_type=task_type,
                selected_model=selected_model,
            )

            # preparing messages for caching
            # generating content with llm using optimized model
            llm_client = get_llm_client()
            llm_response = await llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=generation_request.temperature,
                max_tokens=generation_request.max_tokens,
            )

            generated_content = llm_response["content"]

            # storing in semantic cache - skipping for now
            # await semantic_cache.set(cache_key, generated_content)

            # tracking request with cost optimization - skipping for now
            # cached_tokens = llm_response.get("usage", {}).get("cached_tokens", 0)
            # cost_tracker.track_request(
            #     model=llm_response["model"],
            #     tokens_input=llm_response["usage"]["prompt_tokens"],
            #     tokens_output=llm_response["usage"]["completion_tokens"],
            #     latency_ms=llm_response["latency_ms"],
            #     cached_tokens=cached_tokens,
            #     cache_hit=cached_tokens > 0,
            # )

        # updating grant with generated section
        section_field = generation_request.section
        setattr(grant, section_field, generated_content)
        grant.status = "in_progress"

        await db.commit()
        await db.refresh(grant)

        # creating generation log
        generation_log = GenerationLog(
            user_id=current_user.id,
            grant_id=grant.id,
            prompt=user_prompt,
            system_prompt=system_prompt,
            section_type=generation_request.section,
            response=generated_content,
            model=llm_response["model"],
            provider=llm_response["provider"],
            tokens_input=llm_response["usage"]["prompt_tokens"],
            tokens_output=llm_response["usage"]["completion_tokens"],
            tokens_total=llm_response["usage"]["total_tokens"],
            latency_ms=llm_response["latency_ms"],
            success=True,
            metadata={"similar_docs_count": similar_docs_count},
        )

        db.add(generation_log)
        await db.commit()
        await db.refresh(generation_log)

        # running quality assurance checks
        qa_system = get_quality_assurance_system(db)

        # retrieving structured data for hallucination prevention
        structured_retriever = StructuredDataRetriever(db)
        structured_data = {}

        # getting relevant structured data based on section type
        if "budget" in generation_request.section.lower():
            structured_data["budgets"] = await structured_retriever.get_budget_data()
        if "team" in generation_request.section.lower() or "personnel" in generation_request.section.lower():
            structured_data["team"] = await structured_retriever.get_team_data()
        if "outcome" in generation_request.section.lower() or "impact" in generation_request.section.lower():
            structured_data["outcomes"] = await structured_retriever.get_outcome_data()

        # performing quality review
        qa_results = await qa_system.review_generated_content(
            grant_id=grant.id,
            section=generation_request.section,
            content=generated_content,
            context=rag_context if generation_request.use_rag else "",
            generation_log_id=generation_log.id,
            structured_data=structured_data if structured_data else None,
        )

        logger.info(
            "section_generated_with_qa",
            grant_id=grant.id,
            section=generation_request.section,
            model=llm_response["model"],
            tokens=llm_response["usage"]["total_tokens"],
            faithfulness_score=qa_results["faithfulness_score"],
            qa_passed=qa_results["passed"],
            flagged=qa_results["flagged"],
        )

        return GrantGenerationResponse(
            grant_id=grant.id,
            section=generation_request.section,
            content=generated_content,
            provider=llm_response["provider"],
            model=llm_response["model"],
            tokens_used=llm_response["usage"]["total_tokens"],
            latency_ms=llm_response["latency_ms"],
            similar_documents_used=similar_docs_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "generation_error",
            grant_id=generation_request.grant_id,
            section=generation_request.section,
            error=str(e),
        )

        # logging failed generation
        try:
            generation_log = GenerationLog(
                user_id=current_user.id,
                grant_id=generation_request.grant_id,
                prompt=generation_request.context or "",
                section_type=generation_request.section,
                response="",
                model="unknown",
                provider="unknown",
                success=False,
                error_message=str(e),
            )
            db.add(generation_log)
            await db.commit()
        except:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate grant section: {str(e)}",
        )


@router.post("/search", response_model=List[GrantSearchResult])
async def search_grants(
    search_request: GrantSearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    searching similar grant applications
    """
    try:
        logger.info(
            "searching_grants",
            query_length=len(search_request.query),
            user_id=current_user.id,
        )

        vector_store = get_vector_store(db)

        results = await vector_store.search_similar_grants(
            query_text=search_request.query,
            user_id=current_user.id,
            status=search_request.status,
            top_k=search_request.top_k,
            similarity_threshold=search_request.similarity_threshold,
        )

        search_results = [
            GrantSearchResult(
                grant=GrantApplicationResponse.from_orm(grant),
                similarity_score=score,
            )
            for grant, score in results
        ]

        logger.info("grant_search_complete", results_count=len(search_results))

        return search_results

    except Exception as e:
        logger.error("grant_search_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search grants: {str(e)}",
        )


@router.delete("/{grant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_grant(
    grant_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    deleting grant application
    """
    try:
        logger.info("deleting_grant", grant_id=grant_id, user_id=current_user.id)

        result = await db.execute(
            select(GrantApplication).where(
                GrantApplication.id == grant_id,
                GrantApplication.user_id == current_user.id,
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        await db.delete(grant)
        await db.commit()

        logger.info("grant_deleted", grant_id=grant_id)

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_grant_error", grant_id=grant_id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete grant",
        )


@router.get("/{grant_id}/logs", response_model=List[GenerationLogResponse])
async def get_grant_logs(
    grant_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    getting generation logs for specific grant
    """
    try:
        # verifying grant belongs to user
        result = await db.execute(
            select(GrantApplication).where(
                GrantApplication.id == grant_id,
                GrantApplication.user_id == current_user.id,
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        # fetching logs
        logs_result = await db.execute(
            select(GenerationLog)
            .where(GenerationLog.grant_id == grant_id)
            .order_by(GenerationLog.created_at.desc())
        )
        logs = logs_result.scalars().all()

        return [GenerationLogResponse.from_orm(log) for log in logs]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_logs_error", grant_id=grant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generation logs",
        )


@router.post("/generate-parallel")
async def generate_sections_parallel(
    grant_id: int,
    sections: List[str],
    context: Optional[str] = None,
    use_rag: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    generating multiple grant sections in parallel for faster processing
    """
    try:
        logger.info(
            "generating_sections_parallel",
            grant_id=grant_id,
            sections=sections,
            user_id=current_user.id,
        )

        # fetching grant
        result = await db.execute(
            select(GrantApplication).where(
                GrantApplication.id == grant_id,
                GrantApplication.user_id == current_user.id,
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        # limiting concurrent requests with semaphore
        semaphore = asyncio.Semaphore(5)

        async def generate_single_section(section: str) -> tuple[str, str]:
            """
            generating single section with semaphore control
            """
            async with semaphore:
                try:
                    # checking semantic cache first
                    cache_key = f"{grant.title}_{section}_{context or ''}"
                    cached_response = await semantic_cache.get(cache_key)

                    if cached_response:
                        logger.info("parallel_cache_hit", section=section)
                        return section, cached_response

                    # retrieving similar documents
                    rag_context = ""
                    if use_rag:
                        query = f"{grant.title} {section} {context or ''}"
                        rag_context, _ = await retrieve_similar_documents_enhanced(
                            query=query,
                            db=db,
                            count=3,
                            filters={'document_type': 'past_grant', 'outcome': 'awarded'},
                        )

                    # building prompts
                    system_prompt, user_prompt = await build_generation_prompt_enhanced(
                        section=section,
                        context=context,
                        rag_context=rag_context,
                        db=db,
                    )

                    # smart routing
                    task_type = _determine_task_type(section)
                    selected_model = model_router.route_to_model(task_type)

                    # generating with llm
                    llm_client = get_llm_client()
                    llm_response = await llm_client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.7,
                        max_tokens=2000,
                    )

                    content = llm_response["content"]

                    # caching result
                    await semantic_cache.set(cache_key, content)

                    # tracking cost
                    cached_tokens = llm_response.get("usage", {}).get("cached_tokens", 0)
                    cost_tracker.track_request(
                        model=llm_response["model"],
                        tokens_input=llm_response["usage"]["prompt_tokens"],
                        tokens_output=llm_response["usage"]["completion_tokens"],
                        latency_ms=llm_response["latency_ms"],
                        cached_tokens=cached_tokens,
                        cache_hit=cached_tokens > 0,
                    )

                    logger.info("parallel_section_generated", section=section)
                    return section, content

                except Exception as e:
                    logger.error("parallel_section_error", section=section, error=str(e))
                    return section, f"Error generating {section}: {str(e)}"

        # generating all sections in parallel
        tasks = [generate_single_section(section) for section in sections]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # updating grant with generated sections
        generated_sections = {}
        for result in results:
            if isinstance(result, tuple):
                section, content = result
                generated_sections[section] = content
                setattr(grant, section, content)

        grant.status = "in_progress"
        await db.commit()
        await db.refresh(grant)

        logger.info(
            "parallel_generation_complete",
            grant_id=grant_id,
            sections_count=len(generated_sections),
        )

        return {
            "grant_id": grant_id,
            "sections": generated_sections,
            "total_sections": len(generated_sections),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("parallel_generation_error", grant_id=grant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sections in parallel: {str(e)}",
        )


@router.get("/cost-metrics")
async def get_cost_metrics(
    current_user: User = Depends(get_current_user),
):
    """
    getting cost optimization metrics and savings statistics
    """
    try:
        metrics = cost_tracker.get_metrics()

        # calculating daily costs
        daily_costs = cost_tracker.get_daily_costs()

        # calculating total savings
        total_cost_saved = metrics["total_cost_saved"]
        total_requests = metrics["total_requests"]
        cache_hit_rate = (
            (metrics["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        logger.info("cost_metrics_retrieved", user_id=current_user.id)

        return {
            "summary": {
                "total_requests": total_requests,
                "cache_hit_rate": f"{cache_hit_rate:.2f}%",
                "total_cost_saved": f"${total_cost_saved:.4f}",
                "prompt_cache_hits": metrics["prompt_cache_hits"],
                "semantic_cache_hits": metrics["semantic_cache_hits"],
            },
            "costs_by_model": metrics["costs_by_model"],
            "daily_costs": daily_costs,
            "optimization_impact": {
                "caching_savings": f"${metrics.get('caching_savings', 0):.4f}",
                "routing_savings": f"${metrics.get('routing_savings', 0):.4f}",
                "total_savings_percentage": f"{(total_cost_saved / (total_cost_saved + sum(metrics['costs_by_model'].values())) * 100) if total_cost_saved > 0 else 0:.2f}%",
            },
        }

    except Exception as e:
        logger.error("cost_metrics_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cost metrics: {str(e)}",
        )


@router.post("/cost-metrics/reset")
async def reset_cost_metrics(
    current_user: User = Depends(get_current_user),
):
    """
    resetting cost tracking metrics (admin only - add authorization check in production)
    """
    try:
        cost_tracker.reset()
        logger.info("cost_metrics_reset", user_id=current_user.id)

        return {
            "message": "Cost metrics have been reset successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("cost_metrics_reset_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset cost metrics: {str(e)}",
        )


@router.get("/quality-reviews/queue")
async def get_review_queue(
    flagged_only: bool = True,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    getting quality review queue showing flagged content needing human review
    """
    try:
        from src.core.database import QualityReview

        logger.info("retrieving_review_queue", user_id=current_user.id)

        # building query
        query = select(QualityReview).join(GrantApplication).where(
            GrantApplication.user_id == current_user.id
        )

        if flagged_only:
            query = query.where(QualityReview.flagged_for_review == True)

        query = query.where(QualityReview.approved.is_(None))  # not yet reviewed
        query = query.order_by(QualityReview.created_at.desc()).limit(limit)

        # executing query
        result = await db.execute(query)
        reviews = result.scalars().all()

        # formatting response
        review_queue = [
            {
                "review_id": review.id,
                "grant_id": review.grant_id,
                "section": review.section,
                "review_type": review.review_type,
                "faithfulness_score": float(review.faithfulness_score) if review.faithfulness_score else None,
                "compliance_score": float(review.compliance_score) if review.compliance_score else None,
                "flagged": review.flagged_for_review,
                "hallucinations": review.hallucinations_detected,
                "compliance_issues": review.compliance_issues,
                "created_at": review.created_at.isoformat(),
            }
            for review in reviews
        ]

        logger.info("review_queue_retrieved", count=len(review_queue))

        return {
            "total": len(review_queue),
            "reviews": review_queue,
        }

    except Exception as e:
        logger.error("review_queue_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve review queue: {str(e)}",
        )


@router.get("/quality-reviews/{review_id}")
async def get_quality_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    getting detailed quality review with content and context
    """
    try:
        from src.core.database import QualityReview, GenerationLog

        # retrieving review
        result = await db.execute(
            select(QualityReview)
            .join(GrantApplication)
            .where(
                and_(
                    QualityReview.id == review_id,
                    GrantApplication.user_id == current_user.id,
                )
            )
        )
        review = result.scalar_one_or_none()

        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quality review not found",
            )

        # getting associated generation log
        generation_log = None
        if review.generation_log_id:
            log_result = await db.execute(
                select(GenerationLog).where(GenerationLog.id == review.generation_log_id)
            )
            generation_log = log_result.scalar_one_or_none()

        # formatting response
        response = {
            "review_id": review.id,
            "grant_id": review.grant_id,
            "section": review.section,
            "review_type": review.review_type,
            "scores": {
                "faithfulness": float(review.faithfulness_score) if review.faithfulness_score else None,
                "compliance": float(review.compliance_score) if review.compliance_score else None,
                "similarity": float(review.similarity_score) if review.similarity_score else None,
            },
            "flagged": review.flagged_for_review,
            "hallucinations_detected": review.hallucinations_detected,
            "compliance_issues": review.compliance_issues,
            "generated_content": generation_log.response if generation_log else None,
            "prompt": generation_log.prompt if generation_log else None,
            "system_prompt": generation_log.system_prompt if generation_log else None,
            "approved": review.approved,
            "reviewer_notes": review.reviewer_notes,
            "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
            "created_at": review.created_at.isoformat(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_quality_review_error", review_id=review_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality review: {str(e)}",
        )


@router.post("/quality-reviews/{review_id}/approve")
async def approve_quality_review(
    review_id: int,
    notes: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    approving quality review marking content as acceptable
    """
    try:
        from src.core.database import QualityReview

        # retrieving review
        result = await db.execute(
            select(QualityReview)
            .join(GrantApplication)
            .where(
                and_(
                    QualityReview.id == review_id,
                    GrantApplication.user_id == current_user.id,
                )
            )
        )
        review = result.scalar_one_or_none()

        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quality review not found",
            )

        # updating review
        review.approved = True
        review.reviewer_id = current_user.id
        review.reviewer_notes = notes
        review.reviewed_at = datetime.utcnow()

        await db.commit()

        logger.info("quality_review_approved", review_id=review_id, user_id=current_user.id)

        return {
            "message": "Quality review approved successfully",
            "review_id": review_id,
            "approved": True,
            "reviewed_at": review.reviewed_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("approve_review_error", review_id=review_id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve review: {str(e)}",
        )


@router.post("/quality-reviews/{review_id}/reject")
async def reject_quality_review(
    review_id: int,
    notes: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    rejecting quality review requiring content regeneration
    """
    try:
        from src.core.database import QualityReview

        # retrieving review
        result = await db.execute(
            select(QualityReview)
            .join(GrantApplication)
            .where(
                and_(
                    QualityReview.id == review_id,
                    GrantApplication.user_id == current_user.id,
                )
            )
        )
        review = result.scalar_one_or_none()

        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quality review not found",
            )

        # updating review
        review.approved = False
        review.reviewer_id = current_user.id
        review.reviewer_notes = notes
        review.reviewed_at = datetime.utcnow()

        await db.commit()

        logger.info("quality_review_rejected", review_id=review_id, user_id=current_user.id)

        return {
            "message": "Quality review rejected - content needs regeneration",
            "review_id": review_id,
            "approved": False,
            "reviewed_at": review.reviewed_at.isoformat(),
            "notes": notes,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("reject_review_error", review_id=review_id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reject review: {str(e)}",
        )


@router.get("/{grant_id}/compliance")
async def get_grant_compliance(
    grant_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    getting compliance status for grant showing all requirements and their status
    """
    try:
        from src.core.database import ComplianceRequirement

        # verifying grant ownership
        result = await db.execute(
            select(GrantApplication).where(
                and_(
                    GrantApplication.id == grant_id,
                    GrantApplication.user_id == current_user.id,
                )
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found",
            )

        # retrieving compliance requirements
        reqs_result = await db.execute(
            select(ComplianceRequirement).where(ComplianceRequirement.grant_id == grant_id)
        )
        requirements = reqs_result.scalars().all()

        # categorizing requirements
        addressed = [r for r in requirements if r.addressed]
        missing = [r for r in requirements if not r.addressed and r.is_mandatory]
        optional_missing = [r for r in requirements if not r.addressed and not r.is_mandatory]

        # computing compliance score
        total = len(requirements)
        addressed_count = len(addressed)
        compliance_score = (addressed_count / total * 100) if total > 0 else 100.0

        response = {
            "grant_id": grant_id,
            "total_requirements": total,
            "addressed": len(addressed),
            "missing_mandatory": len(missing),
            "missing_optional": len(optional_missing),
            "compliance_score": round(compliance_score, 2),
            "compliant": compliance_score >= 95.0,
            "requirements": {
                "addressed": [
                    {
                        "id": r.id,
                        "text": r.requirement_text,
                        "type": r.requirement_type,
                        "section": r.section,
                        "evidence": r.evidence_location,
                    }
                    for r in addressed
                ],
                "missing_mandatory": [
                    {
                        "id": r.id,
                        "text": r.requirement_text,
                        "type": r.requirement_type,
                        "section": r.section,
                    }
                    for r in missing
                ],
                "missing_optional": [
                    {
                        "id": r.id,
                        "text": r.requirement_text,
                        "type": r.requirement_type,
                        "section": r.section,
                    }
                    for r in optional_missing
                ],
            },
        }

        logger.info(
            "compliance_retrieved",
            grant_id=grant_id,
            compliance_score=compliance_score,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_compliance_error", grant_id=grant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get compliance status: {str(e)}",
        )


@router.get("/{grant_id}/export")
async def export_grant_word(
    grant_id: int,
    include_metadata: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    exporting grant application as professionally formatted word document

    creates a microsoft word document (.docx) with proper formatting, cambio labs branding,
    section hierarchy, and professional styling suitable for grant submission

    args:
        grant_id: unique identifier for grant application
        include_metadata: whether to include generation metadata in footer
        db: database session
        current_user: authenticated user making request

    returns:
        word document file as binary download

    raises:
        404: grant not found or user does not have access
        500: error generating word document
    """
    try:
        from fastapi.responses import FileResponse
        from sqlalchemy import and_

        logger.info(
            "export_word_requested",
            grant_id=grant_id,
            user_id=current_user.id,
            include_metadata=include_metadata
        )

        # retrieving grant application from database
        result = await db.execute(
            select(GrantApplication).where(
                and_(
                    GrantApplication.id == grant_id,
                    GrantApplication.user_id == current_user.id,
                )
            )
        )
        grant = result.scalar_one_or_none()

        if not grant:
            logger.warning(
                "export_grant_not_found",
                grant_id=grant_id,
                user_id=current_user.id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Grant application not found or you do not have access",
            )

        # checking that grant has content to export
        sections_with_content = sum([
            1 for section_key, _ in [
                ('executive_summary', ''),
                ('organizational_background', ''),
                ('problem_statement', ''),
                ('project_description', ''),
                ('budget_narrative', ''),
                ('evaluation_plan', '')
            ]
            if getattr(grant, section_key, None) and getattr(grant, section_key).strip()
        ])

        if sections_with_content == 0:
            logger.warning(
                "export_empty_grant",
                grant_id=grant_id,
                user_id=current_user.id
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Grant application has no content to export. Please generate sections first.",
            )

        # getting word exporter instance
        # checking for logo in cloud directory first, then root
        logo_path = None
        possible_logo_paths = [
            "/Users/abdulbasir/cambio-labs-eduquery/cloud/logo.png",
            "/Users/abdulbasir/cambio-labs-eduquery/logo.png",
            "/Users/abdulbasir/cambio-labs-eduquery/logo.jpg",
        ]

        for path in possible_logo_paths:
            if os.path.exists(path):
                logo_path = path
                break

        exporter = get_word_exporter(logo_path=logo_path)

        # creating word document
        document_path = await exporter.create_grant_document(
            grant=grant,
            user=current_user,
            include_metadata=include_metadata
        )

        # generating filename for download
        safe_title = "".join(c for c in grant.title if c.isalnum() or c in (' ', '-', '_'))
        safe_title = safe_title.replace(' ', '_')[:50]
        download_filename = f"Cambio_Labs_{safe_title}.docx"

        logger.info(
            "export_word_success",
            grant_id=grant_id,
            user_id=current_user.id,
            document_path=document_path,
            filename=download_filename,
            sections=sections_with_content
        )

        # returning file for download
        return FileResponse(
            path=document_path,
            filename=download_filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{download_filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "export_word_error",
            grant_id=grant_id,
            user_id=current_user.id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export word document: {str(e)}",
        )
