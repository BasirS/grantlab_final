"""
Embeddings generation using OpenAI API with error handling and caching.
"""

from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import hashlib
import json

from openai import AsyncOpenAI, OpenAIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import structlog

from src.config import settings


# initializing structured logger
logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """
    simple in-memory cache for embeddings to reduce api calls
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        initializing cache with time-to-live for entries
        """
        self.cache: Dict[str, tuple[List[float], datetime]] = {}
        self.ttl_seconds = ttl_seconds

    def _generate_key(self, text: str, model: str) -> str:
        """
        generating cache key from text content and model name
        """
        content = f"{text}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        getting cached embedding if it exists and is not expired
        """
        key = self._generate_key(text, model)
        if key in self.cache:
            embedding, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl_seconds):
                logger.debug("cache_hit", key=key[:8])
                return embedding
            else:
                # removing expired entry
                del self.cache[key]
                logger.debug("cache_expired", key=key[:8])
        return None

    def set(self, text: str, model: str, embedding: List[float]):
        """
        storing embedding in cache with current timestamp
        """
        key = self._generate_key(text, model)
        self.cache[key] = (embedding, datetime.utcnow())
        logger.debug("cache_set", key=key[:8])

    def clear(self):
        """
        clearing all cached embeddings
        """
        self.cache.clear()
        logger.info("cache_cleared")


class EmbeddingGenerator:
    """
    generating embeddings using openai api with retry logic and caching
    """

    def __init__(self):
        """
        initializing openai client and embedding cache
        """
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            organization=settings.openai_org_id,
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.openai_embedding_model
        self.cache = EmbeddingCache()
        self.dimension = settings.vector_dimension

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        reraise=True,
    )
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
    ) -> List[float]:
        """
        generating single embedding for text with retry logic and caching
        """
        # checking cache first if enabled
        if use_cache:
            cached_embedding = self.cache.get(text, self.model)
            if cached_embedding is not None:
                return cached_embedding

        try:
            # preprocessing text by removing excessive whitespace
            clean_text = " ".join(text.split())

            # truncating text if too long (openai limit is 8191 tokens, roughly 32k chars)
            max_chars = 30000
            if len(clean_text) > max_chars:
                logger.warning(
                    "text_truncated",
                    original_length=len(clean_text),
                    truncated_length=max_chars,
                )
                clean_text = clean_text[:max_chars]

            # generating embedding via openai api
            logger.debug(
                "generating_embedding",
                model=self.model,
                text_length=len(clean_text),
            )

            response = await self.client.embeddings.create(
                input=clean_text,
                model=self.model,
            )

            embedding = response.data[0].embedding

            # validating embedding dimension
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.dimension}, got {len(embedding)}"
                )

            # caching the result
            if use_cache:
                self.cache.set(text, self.model, embedding)

            logger.info(
                "embedding_generated",
                model=self.model,
                dimension=len(embedding),
                usage=response.usage.total_tokens,
            )

            return embedding

        except RateLimitError as e:
            logger.error("rate_limit_error", error=str(e))
            raise
        except APITimeoutError as e:
            logger.error("timeout_error", error=str(e))
            raise
        except OpenAIError as e:
            logger.error("openai_error", error=str(e), error_type=type(e).__name__)
            raise
        except Exception as e:
            logger.error(
                "unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        generating embeddings for multiple texts in batches with concurrent processing
        """
        if not texts:
            return []

        logger.info(
            "generating_batch_embeddings",
            total_texts=len(texts),
            batch_size=batch_size,
        )

        embeddings = []

        # processing texts in batches to avoid overwhelming the api
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # generating embeddings concurrently within each batch
            batch_tasks = [
                self.generate_embedding(text, use_cache=use_cache)
                for text in batch
            ]

            try:
                batch_embeddings = await asyncio.gather(*batch_tasks)
                embeddings.extend(batch_embeddings)

                logger.debug(
                    "batch_completed",
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                )

            except Exception as e:
                logger.error(
                    "batch_error",
                    batch_number=i // batch_size + 1,
                    error=str(e),
                )
                raise

        logger.info(
            "batch_embeddings_complete",
            total_generated=len(embeddings),
        )

        return embeddings

    def clear_cache(self):
        """
        clearing embedding cache
        """
        self.cache.clear()


# creating singleton instance for reuse across the application
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    getting singleton embedding generator instance
    """
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


async def generate_embedding(text: str, use_cache: bool = True) -> List[float]:
    """
    convenience function for generating single embedding
    """
    generator = get_embedding_generator()
    return await generator.generate_embedding(text, use_cache=use_cache)


async def generate_embeddings_batch(
    texts: List[str],
    use_cache: bool = True,
    batch_size: int = 100,
) -> List[List[float]]:
    """
    convenience function for generating multiple embeddings
    """
    generator = get_embedding_generator()
    return await generator.generate_embeddings_batch(
        texts, use_cache=use_cache, batch_size=batch_size
    )
