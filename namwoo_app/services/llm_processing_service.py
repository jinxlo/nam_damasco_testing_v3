# namwoo_app/services/llm_processing_service.py
import logging
from typing import Optional

from ..config import Config
# Import the specific summarization functions from your existing LLM service files
# We'll define these specific functions in the next steps.
from . import openai_service
from . import google_service
from ..utils.text_utils import strip_html_to_text # For pre-stripping HTML

logger = logging.getLogger(__name__)

def generate_llm_product_summary(
    html_description: Optional[str],
    item_name: Optional[str] = None
) -> Optional[str]:
    """
    Generates a product summary using the configured LLM provider.
    It first strips HTML from the description before sending to the LLM.
    """
    if not html_description:
        logger.debug("No HTML description provided for summarization.")
        return None

    plain_text_description = strip_html_to_text(html_description)
    if not plain_text_description:
        logger.debug(f"HTML description for '{item_name or 'Unknown'}' stripped to empty text; skipping summarization.")
        return None

    provider = Config.LLM_PROVIDER.lower()
    logger.info(f"Generating product summary for '{item_name or 'Unknown'}' using LLM provider: {provider}")

    summary = None
    try:
        if provider == 'openai':
            # We'll create/ensure this function exists in openai_service.py
            summary = openai_service.get_openai_product_summary(plain_text_description, item_name)
        elif provider == 'google':
            # We'll create/ensure this function exists in google_service.py
            summary = google_service.get_google_product_summary(plain_text_description, item_name)
        else:
            logger.error(f"Unsupported LLM_PROVIDER for summarization: {provider}. Cannot generate summary.")
            return None # Or fallback to just plain_text_description if preferred

        if summary:
            logger.info(f"Successfully generated summary via {provider} for '{item_name or 'Unknown'}'.")
        else:
            logger.warning(f"Summarization via {provider} for '{item_name or 'Unknown'}' returned no summary.")
        
        return summary

    except Exception as e:
        logger.exception(f"Error during LLM summarization call via {provider} for '{item_name or 'Unknown'}': {e}")
        return None # Fallback on error