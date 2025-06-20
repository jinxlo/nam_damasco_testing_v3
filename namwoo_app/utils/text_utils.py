# namwoo_app/utils/text_utils.py
import logging
from bs4 import BeautifulSoup
from typing import Optional # Optional is good practice for type hints

logger = logging.getLogger(__name__)

def strip_html_to_text(html_content: Optional[str]) -> str:
    """
    Strips HTML tags from a string and returns plain text.
    Handles None or empty input gracefully, returning an empty string.
    Normalizes whitespace in the resulting plain text.
    """
    if not html_content or not isinstance(html_content, str):
        # If input is None, or not a string (e.g. if an int was accidentally passed)
        return "" 
    
    try:
        # Use html.parser, it's built-in and usually sufficient.
        # lxml is faster but an external dependency.
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Get all text content. The 'separator=" "' helps by putting a space
        # where block tags (like <p>, <div>, <li>) were, preventing words
        # from different blocks from mashing together.
        text = soup.get_text(separator=" ")
        
        # Normalize whitespace:
        # 1. Replace multiple whitespace characters (including newlines, tabs) with a single space.
        # 2. Strip leading/trailing whitespace.
        if text: # Only process if text is not None or empty
            normalized_text = " ".join(text.split()).strip()
            return normalized_text
        else:
            return "" # Return empty string if soup.get_text() results in None or empty
            
    except Exception as e:
        # Log the error and the beginning of the problematic HTML for debugging.
        logger.warning(
            f"Error stripping HTML: {e}. "
            f"Input snippet (first 100 chars): '{str(html_content)[:100]}...' - Returning empty string."
        )
        # Fallback to an empty string to ensure clean text output,
        # especially if this text is used for embeddings or LLM prompts.
        return ""


def split_full_name(full_name: str) -> tuple[str, str]:
    """Split a full name string into first name and last name parts."""
    if not full_name or not isinstance(full_name, str):
        return "", ""
    parts = full_name.strip().split()
    nombre = parts[0] if parts else ""
    apellido = " ".join(parts[1:]) if len(parts) > 1 else ""
    return nombre, apellido

