import logging
import time
from typing import List, Optional
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from ..config import Config

logger = logging.getLogger(__name__)

_openai_client = None

def _get_openai_client() -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client, ensuring API key is set."""
    global _openai_client
    if _openai_client is None:
        api_key = Config.OPENAI_API_KEY
        if api_key:
            try:
                _openai_client = OpenAI(api_key=api_key, timeout=20.0)
                logger.info("OpenAI client initialized for embedding generation.")
            except Exception as e:
                logger.exception(f"Failed to initialize OpenAI client: {e}")
                _openai_client = None
        else:
            logger.error("OpenAI API key not configured. Embedding generation will fail.")
    return _openai_client

def get_embedding(
    text: str,
    model: str = Config.OPENAI_EMBEDDING_MODEL,
    retries: int = 2,
    initial_delay: float = 1.0
) -> Optional[List[float]]:
    """
    Generates an embedding for the given text using the configured OpenAI model.
    Includes retry logic for transient API errors.

    Args:
        text: The input text (e.g., Damasco product description).
        model: OpenAI embedding model ID.
        retries: Number of retries on API errors.
        initial_delay: Initial retry delay in seconds (exponential backoff).

    Returns:
        A list of floats (embedding vector) or None if failed.
    """
    client = _get_openai_client()
    if not client:
        logger.error("Cannot generate embedding: OpenAI client unavailable.")
        return None

    if not text or not isinstance(text, str):
        logger.warning("Invalid or empty text for embedding generation. Returning None.")
        return None

    processed_text = text.replace("\n", " ").strip()
    if not processed_text:
        logger.warning("Text became empty after cleaning. Skipping embedding.")
        return None

    current_retries = 0
    delay = initial_delay
    while current_retries <= retries:
        try:
            response = client.embeddings.create(
                input=[processed_text],
                model=model
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text: '{processed_text[:50]}...'")
            return embedding
        except (RateLimitError, APITimeoutError) as e:
            logger.warning(f"Retrying due to API limit/timeout. Delay {delay}s (Attempt {current_retries + 1}/{retries + 1})")
        except APIError as e:
            if e.status_code >= 500:
                logger.warning(f"OpenAI server error {e.status_code}. Retrying in {delay}s...")
            else:
                logger.error(f"Non-retryable OpenAI API error: {e} (Status: {e.status_code})")
                return None
        except Exception as e:
            logger.exception(f"Unexpected error during embedding generation: {e}")
            return None

        current_retries += 1
        if current_retries <= retries:
            time.sleep(delay)
            delay *= 2

    logger.error(f"Failed to generate embedding after {retries} retries for text: '{processed_text[:50]}...'")
    return None
