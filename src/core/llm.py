"""
LLM client with multi-provider fallback support and circuit breaker pattern.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime, timedelta
import asyncio

from openai import AsyncOpenAI, OpenAIError, RateLimitError, APITimeoutError
from anthropic import AsyncAnthropic, APIError, RateLimitError as AnthropicRateLimitError
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


class LLMProvider(str, Enum):
    """
    enum for supported llm providers
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class CircuitState(str, Enum):
    """
    enum for circuit breaker states
    """
    CLOSED = "closed"  # normal operation
    OPEN = "open"      # failing, rejecting requests
    HALF_OPEN = "half_open"  # testing if service recovered


class CircuitBreaker:
    """
    implementing circuit breaker pattern to prevent cascade failures
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 3,
    ):
        """
        initializing circuit breaker with configurable thresholds
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts

        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[datetime] = None

    def record_success(self):
        """
        recording successful request
        """
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_attempts:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("circuit_closed", reason="successful_recovery")

    def record_failure(self):
        """
        recording failed request and updating circuit state
        """
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "circuit_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def can_attempt(self) -> bool:
        """
        checking if request can be attempted based on circuit state
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # checking if timeout has elapsed to try half-open
            if self.last_failure_time:
                elapsed = datetime.utcnow() - self.last_failure_time
                if elapsed.total_seconds() >= self.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("circuit_half_open", elapsed_seconds=elapsed.total_seconds())
                    return True
            return False

        # half-open state allows attempts
        return True

    def reset(self):
        """
        resetting circuit breaker to closed state
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class LLMClient:
    """
    unified llm client with multi-provider fallback and circuit breakers
    """

    def __init__(self):
        """
        initializing llm clients for openai and anthropic
        """
        # initializing openai client
        self.openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            organization=settings.openai_org_id,
            timeout=settings.llm_timeout_seconds,
        )

        # initializing anthropic client if api key is available
        self.anthropic_client = None
        if settings.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_seconds,
            )

        # initializing circuit breakers for each provider
        self.openai_circuit = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            timeout_seconds=settings.circuit_breaker_timeout_seconds,
            half_open_attempts=settings.circuit_breaker_half_open_attempts,
        )
        self.anthropic_circuit = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            timeout_seconds=settings.circuit_breaker_timeout_seconds,
            half_open_attempts=settings.circuit_breaker_half_open_attempts,
        )

        # tracking provider usage
        self.primary_model = settings.openai_model
        self.fallback_model = settings.openai_mini_model
        self.anthropic_model = settings.anthropic_model

    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        calling openai api with error handling
        """
        try:
            if not self.openai_circuit.can_attempt():
                raise Exception("OpenAI circuit breaker is open")

            temp = temperature if temperature is not None else settings.llm_temperature
            tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

            logger.debug(
                "calling_openai",
                model=model,
                temperature=temp,
                max_tokens=tokens,
            )

            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )

            self.openai_circuit.record_success()

            result = {
                "provider": LLMProvider.OPENAI,
                "model": model,
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }

            logger.info(
                "openai_success",
                model=model,
                tokens=response.usage.total_tokens,
            )

            return result

        except (RateLimitError, APITimeoutError) as e:
            self.openai_circuit.record_failure()
            logger.warning("openai_retryable_error", error=str(e), error_type=type(e).__name__)
            raise
        except OpenAIError as e:
            self.openai_circuit.record_failure()
            logger.error("openai_error", error=str(e), error_type=type(e).__name__)
            raise
        except Exception as e:
            self.openai_circuit.record_failure()
            logger.error("openai_unexpected_error", error=str(e), error_type=type(e).__name__)
            raise

    async def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        calling anthropic api with error handling
        """
        if not self.anthropic_client:
            raise Exception("Anthropic client not configured")

        try:
            if not self.anthropic_circuit.can_attempt():
                raise Exception("Anthropic circuit breaker is open")

            temp = temperature if temperature is not None else settings.llm_temperature
            tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

            # separating system message from user messages for anthropic
            system_message = None
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            logger.debug(
                "calling_anthropic",
                model=self.anthropic_model,
                temperature=temp,
                max_tokens=tokens,
            )

            response = await self.anthropic_client.messages.create(
                model=self.anthropic_model,
                messages=user_messages,
                system=system_message,
                temperature=temp,
                max_tokens=tokens,
            )

            self.anthropic_circuit.record_success()

            result = {
                "provider": LLMProvider.ANTHROPIC,
                "model": self.anthropic_model,
                "content": response.content[0].text,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                "finish_reason": response.stop_reason,
            }

            logger.info(
                "anthropic_success",
                model=self.anthropic_model,
                tokens=result["usage"]["total_tokens"],
            )

            return result

        except AnthropicRateLimitError as e:
            self.anthropic_circuit.record_failure()
            logger.warning("anthropic_rate_limit", error=str(e))
            raise
        except APIError as e:
            self.anthropic_circuit.record_failure()
            logger.error("anthropic_error", error=str(e), error_type=type(e).__name__)
            raise
        except Exception as e:
            self.anthropic_circuit.record_failure()
            logger.error("anthropic_unexpected_error", error=str(e), error_type=type(e).__name__)
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        generating text with automatic fallback between providers
        """
        start_time = datetime.utcnow()

        # building messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("generating_text", prompt_length=len(prompt))

        # trying primary model first (gpt-4o)
        try:
            result = await self._call_openai(
                messages=messages,
                model=self.primary_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result["latency_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return result

        except RateLimitError:
            logger.warning("rate_limit_primary", model=self.primary_model)

            # trying fallback model (gpt-4o-mini) for rate limit
            if use_fallback:
                try:
                    result = await self._call_openai(
                        messages=messages,
                        model=self.fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result["latency_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    return result
                except Exception as e:
                    logger.error("fallback_model_failed", model=self.fallback_model, error=str(e))

        except Exception as e:
            logger.error("primary_model_failed", model=self.primary_model, error=str(e))

        # trying anthropic as final fallback
        if use_fallback and self.anthropic_client:
            try:
                logger.info("trying_anthropic_fallback")
                result = await self._call_anthropic(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result["latency_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                return result
            except Exception as e:
                logger.error("anthropic_fallback_failed", error=str(e))
                raise

        # all providers failed
        raise Exception("All LLM providers failed")

    def get_circuit_status(self) -> Dict[str, str]:
        """
        getting current status of all circuit breakers
        """
        return {
            "openai": self.openai_circuit.state.value,
            "anthropic": self.anthropic_circuit.state.value,
        }

    def reset_circuits(self):
        """
        resetting all circuit breakers
        """
        self.openai_circuit.reset()
        self.anthropic_circuit.reset()
        logger.info("circuits_reset")


# creating singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    getting singleton llm client instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    """
    convenience function for generating text with llm
    """
    client = get_llm_client()
    return await client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        use_fallback=use_fallback,
    )
