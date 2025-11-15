"""
Vector store operations using PostgreSQL pgvector for semantic search.
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.database import Document, GrantApplication
from src.core.embeddings import generate_embedding
from src.config import settings


# initializing structured logger
logger = structlog.get_logger(__name__)


class VectorStore:
    """
    managing vector operations for semantic search using pgvector
    """

    def __init__(self, session: AsyncSession):
        """
        initializing vector store with database session
        """
        self.session = session
        self.similarity_threshold = settings.vector_similarity_threshold
        self.top_k = settings.top_k_results

    async def add_document_embedding(
        self,
        document_id: int,
        text: str,
    ) -> None:
        """
        generating and storing embedding for a document
        """
        try:
            logger.info("adding_document_embedding", document_id=document_id)

            # generating embedding for document content
            embedding = await generate_embedding(text)

            # updating document with embedding
            result = await self.session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if document is None:
                raise ValueError(f"Document {document_id} not found")

            document.content_embedding = embedding
            document.processed = True
            document.processing_error = None

            await self.session.commit()

            logger.info(
                "document_embedding_added",
                document_id=document_id,
                dimension=len(embedding),
            )

        except Exception as e:
            logger.error(
                "document_embedding_error",
                document_id=document_id,
                error=str(e),
            )
            await self.session.rollback()

            # marking document with error
            result = await self.session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()
            if document:
                document.processing_error = str(e)
                await self.session.commit()

            raise

    async def add_grant_embedding(
        self,
        grant_id: int,
        text: str,
    ) -> None:
        """
        generating and storing embedding for a grant application
        """
        try:
            logger.info("adding_grant_embedding", grant_id=grant_id)

            # generating embedding for grant content
            embedding = await generate_embedding(text)

            # updating grant with embedding
            result = await self.session.execute(
                select(GrantApplication).where(GrantApplication.id == grant_id)
            )
            grant = result.scalar_one_or_none()

            if grant is None:
                raise ValueError(f"Grant {grant_id} not found")

            grant.content_embedding = embedding
            await self.session.commit()

            logger.info(
                "grant_embedding_added",
                grant_id=grant_id,
                dimension=len(embedding),
            )

        except Exception as e:
            logger.error(
                "grant_embedding_error",
                grant_id=grant_id,
                error=str(e),
            )
            await self.session.rollback()
            raise

    async def search_similar_documents(
        self,
        query_text: str,
        document_type: Optional[str] = None,
        program: Optional[str] = None,
        funder: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        searching for similar documents using cosine similarity
        """
        try:
            k = top_k or self.top_k
            threshold = similarity_threshold or self.similarity_threshold

            logger.info(
                "searching_documents",
                document_type=document_type,
                program=program,
                funder=funder,
                top_k=k,
            )

            # generating query embedding
            query_embedding = await generate_embedding(query_text)

            # building base query with cosine similarity
            query = select(
                Document,
                (1 - Document.content_embedding.cosine_distance(query_embedding)).label("similarity")
            ).where(
                Document.content_embedding.isnot(None),
                Document.processed == True,
            )

            # applying filters if provided
            if document_type:
                query = query.where(Document.document_type == document_type)
            if program:
                query = query.where(Document.program == program)
            if funder:
                query = query.where(Document.funder == funder)

            # filtering by similarity threshold and ordering by similarity
            query = query.where(
                (1 - Document.content_embedding.cosine_distance(query_embedding)) >= threshold
            ).order_by(
                text("similarity DESC")
            ).limit(k)

            # executing query
            result = await self.session.execute(query)
            results = result.all()

            logger.info(
                "documents_found",
                count=len(results),
                top_similarity=results[0][1] if results else None,
            )

            return [(row[0], row[1]) for row in results]

        except Exception as e:
            logger.error("document_search_error", error=str(e))
            raise

    async def search_similar_grants(
        self,
        query_text: str,
        user_id: Optional[int] = None,
        status: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[GrantApplication, float]]:
        """
        searching for similar grant applications using cosine similarity
        """
        try:
            k = top_k or self.top_k
            threshold = similarity_threshold or self.similarity_threshold

            logger.info(
                "searching_grants",
                user_id=user_id,
                status=status,
                top_k=k,
            )

            # generating query embedding
            query_embedding = await generate_embedding(query_text)

            # building base query with cosine similarity
            query = select(
                GrantApplication,
                (1 - GrantApplication.content_embedding.cosine_distance(query_embedding)).label("similarity")
            ).where(
                GrantApplication.content_embedding.isnot(None),
            )

            # applying filters if provided
            if user_id:
                query = query.where(GrantApplication.user_id == user_id)
            if status:
                query = query.where(GrantApplication.status == status)

            # filtering by similarity threshold and ordering by similarity
            query = query.where(
                (1 - GrantApplication.content_embedding.cosine_distance(query_embedding)) >= threshold
            ).order_by(
                text("similarity DESC")
            ).limit(k)

            # executing query
            result = await self.session.execute(query)
            results = result.all()

            logger.info(
                "grants_found",
                count=len(results),
                top_similarity=results[0][1] if results else None,
            )

            return [(row[0], row[1]) for row in results]

        except Exception as e:
            logger.error("grant_search_error", error=str(e))
            raise

    async def batch_add_document_embeddings(
        self,
        document_ids: List[int],
    ) -> Dict[str, Any]:
        """
        generating and storing embeddings for multiple documents in batch
        """
        results = {
            "success": [],
            "failed": [],
            "total": len(document_ids),
        }

        logger.info("batch_processing_documents", total=len(document_ids))

        for doc_id in document_ids:
            try:
                # fetching document
                result = await self.session.execute(
                    select(Document).where(Document.id == doc_id)
                )
                document = result.scalar_one_or_none()

                if document is None:
                    results["failed"].append({
                        "id": doc_id,
                        "error": "Document not found",
                    })
                    continue

                if not document.content:
                    results["failed"].append({
                        "id": doc_id,
                        "error": "No content to embed",
                    })
                    continue

                # adding embedding
                await self.add_document_embedding(doc_id, document.content)
                results["success"].append(doc_id)

            except Exception as e:
                logger.error(
                    "batch_document_error",
                    document_id=doc_id,
                    error=str(e),
                )
                results["failed"].append({
                    "id": doc_id,
                    "error": str(e),
                })

        logger.info(
            "batch_processing_complete",
            success=len(results["success"]),
            failed=len(results["failed"]),
        )

        return results

    async def get_document_by_similarity_score(
        self,
        grant_text: str,
        min_score: float = 0.8,
    ) -> Optional[Document]:
        """
        finding most similar document above a minimum similarity score
        """
        results = await self.search_similar_documents(
            query_text=grant_text,
            top_k=1,
            similarity_threshold=min_score,
        )

        if results:
            return results[0][0]
        return None

    async def get_documents_by_filters(
        self,
        query_text: str,
        filters: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        searching documents with multiple filter criteria
        """
        return await self.search_similar_documents(
            query_text=query_text,
            document_type=filters.get("document_type"),
            program=filters.get("program"),
            funder=filters.get("funder"),
            top_k=top_k,
            similarity_threshold=filters.get("similarity_threshold"),
        )


def get_vector_store(session: AsyncSession) -> VectorStore:
    """
    creating vector store instance with database session
    """
    return VectorStore(session)
