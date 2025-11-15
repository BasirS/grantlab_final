"""
Hybrid search combining BM25 sparse retrieval with vector similarity using reciprocal rank fusion.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
from collections import defaultdict

from sqlalchemy import select, text, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.database import Document
from src.core.embeddings import generate_embedding
from src.config import settings


# initializing structured logger
logger = structlog.get_logger(__name__)


class BM25Retriever:
    """
    implementing bm25 sparse retrieval using postgresql full-text search
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        initializing bm25 retriever with tuning parameters
        """
        self.k1 = k1  # term frequency saturation
        self.b = b    # length normalization

    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        searching documents using bm25 ranking on full-text search
        """
        try:
            logger.debug("bm25_search", query=query[:50], top_k=top_k)

            # building base query with full-text search
            query_tsquery = text(
                "plainto_tsquery('english', :query)"
            ).bindparams(query=query)

            # creating ts_rank query for scoring
            base_query = select(
                Document,
                func.ts_rank(
                    func.to_tsvector('english', Document.content),
                    query_tsquery,
                ).label('rank')
            ).where(
                func.to_tsvector('english', Document.content).op('@@')(query_tsquery),
                Document.processed == True,
            )

            # applying filters if provided
            if filters:
                if filters.get('document_type'):
                    base_query = base_query.where(
                        Document.document_type == filters['document_type']
                    )
                if filters.get('program'):
                    base_query = base_query.where(
                        Document.program == filters['program']
                    )
                if filters.get('funder'):
                    base_query = base_query.where(
                        Document.funder == filters['funder']
                    )
                if filters.get('year'):
                    base_query = base_query.where(
                        Document.year == filters['year']
                    )
                if filters.get('outcome'):
                    base_query = base_query.where(
                        Document.outcome == filters['outcome']
                    )

            # ordering by rank and limiting results
            base_query = base_query.order_by(text('rank DESC')).limit(top_k)

            # executing query
            result = await session.execute(base_query)
            results = result.all()

            logger.info("bm25_search_complete", results_count=len(results))

            return [(row[0], float(row[1])) for row in results]

        except Exception as e:
            logger.error("bm25_search_error", error=str(e))
            return []


class VectorRetriever:
    """
    implementing vector similarity search using pgvector
    """

    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        searching documents using cosine similarity on embeddings
        """
        try:
            logger.debug("vector_search", query=query[:50], top_k=top_k)

            # generating query embedding
            query_embedding = await generate_embedding(query)

            # building base query with cosine similarity
            base_query = select(
                Document,
                (1 - Document.content_embedding.cosine_distance(query_embedding)).label("similarity")
            ).where(
                Document.content_embedding.isnot(None),
                Document.processed == True,
            )

            # applying filters if provided
            if filters:
                if filters.get('document_type'):
                    base_query = base_query.where(
                        Document.document_type == filters['document_type']
                    )
                if filters.get('program'):
                    base_query = base_query.where(
                        Document.program == filters['program']
                    )
                if filters.get('funder'):
                    base_query = base_query.where(
                        Document.funder == filters['funder']
                    )
                if filters.get('year'):
                    base_query = base_query.where(
                        Document.year == filters['year']
                    )
                if filters.get('outcome'):
                    base_query = base_query.where(
                        Document.outcome == filters['outcome']
                    )

            # filtering by similarity threshold
            if similarity_threshold > 0:
                base_query = base_query.where(
                    (1 - Document.content_embedding.cosine_distance(query_embedding)) >= similarity_threshold
                )

            # ordering by similarity and limiting results
            base_query = base_query.order_by(text("similarity DESC")).limit(top_k)

            # executing query
            result = await session.execute(base_query)
            results = result.all()

            logger.info("vector_search_complete", results_count=len(results))

            return [(row[0], float(row[1])) for row in results]

        except Exception as e:
            logger.error("vector_search_error", error=str(e))
            return []


class ReciprocalRankFusion:
    """
    combining multiple ranked lists using reciprocal rank fusion
    """

    def __init__(self, k: int = 60):
        """
        initializing rrf with constant k for score calculation
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: List[List[Tuple[Document, float]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        fusing multiple ranked lists into single ranking using rrf
        """
        if not ranked_lists:
            return []

        # defaulting to equal weights if not provided
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        # normalizing weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # calculating rrf scores
        doc_scores = defaultdict(float)
        doc_objects = {}

        for rank_list, weight in zip(ranked_lists, weights):
            for rank, (doc, original_score) in enumerate(rank_list, start=1):
                # rrf formula: score = weight / (k + rank)
                rrf_score = weight / (self.k + rank)
                doc_id = doc.id
                doc_scores[doc_id] += rrf_score
                doc_objects[doc_id] = doc

        # sorting by fused score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # creating result list with documents and scores
        results = [
            (doc_objects[doc_id], score)
            for doc_id, score in sorted_docs
        ]

        logger.debug(
            "rrf_fusion_complete",
            num_lists=len(ranked_lists),
            total_unique_docs=len(results),
        )

        return results


class HybridSearchEngine:
    """
    combining bm25 and vector search using reciprocal rank fusion
    """

    def __init__(
        self,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        """
        initializing hybrid search engine with retriever weights
        """
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        searching documents using hybrid bm25 + vector retrieval
        """
        try:
            logger.info(
                "hybrid_search_starting",
                query=query[:50],
                top_k=top_k,
                use_hybrid=use_hybrid,
            )

            if not use_hybrid:
                # using only vector search
                return await self.vector_retriever.search(
                    session=session,
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    filters=filters,
                )

            # retrieving more results for fusion (2x top_k)
            fetch_k = min(top_k * 2, 50)

            # performing both searches concurrently
            import asyncio
            bm25_task = self.bm25_retriever.search(
                session=session,
                query=query,
                top_k=fetch_k,
                filters=filters,
            )
            vector_task = self.vector_retriever.search(
                session=session,
                query=query,
                top_k=fetch_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )

            bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

            # fusing results using rrf
            fused_results = self.rrf.fuse(
                ranked_lists=[bm25_results, vector_results],
                weights=[self.bm25_weight, self.vector_weight],
            )

            # limiting to top_k
            final_results = fused_results[:top_k]

            logger.info(
                "hybrid_search_complete",
                bm25_results=len(bm25_results),
                vector_results=len(vector_results),
                fused_results=len(fused_results),
                final_results=len(final_results),
            )

            return final_results

        except Exception as e:
            logger.error("hybrid_search_error", error=str(e))
            # falling back to vector search only
            return await self.vector_retriever.search(
                session=session,
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )


# creating singleton instance
_hybrid_search_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    """
    getting singleton hybrid search engine
    """
    global _hybrid_search_engine
    if _hybrid_search_engine is None:
        _hybrid_search_engine = HybridSearchEngine()
    return _hybrid_search_engine
