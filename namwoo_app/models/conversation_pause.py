# models/conversation_pause.py
import logging
from sqlalchemy import Column, String, DateTime # Removed Integer, String(255) is fine for IDs
# from sqlalchemy.dialects.postgresql import VARCHAR, TIMESTAMP # Standard DateTime(timezone=True) is generally preferred and cross-DB compatible
from . import Base # Import Base from the models package's __init__.py
import datetime # Keep this for datetime.datetime within the class if preferred
from datetime import timezone # Or import timezone directly for cleaner calls

logger = logging.getLogger(__name__)

class ConversationPause(Base):
    """
    SQLAlchemy ORM model representing the 'conversation_pauses' table.
    Tracks when a conversation should be paused for the bot due to human takeover.
    """
    __tablename__ = 'conversation_pauses'

    # Corresponds to conversation_id VARCHAR(255) PRIMARY KEY in schema.sql
    # Support Board IDs are usually strings, even if they look numeric.
    conversation_id = Column(String(255), primary_key=True, index=True) # Added index=True for faster lookups

    # Corresponds to paused_until TIMESTAMP WITH TIME ZONE NOT NULL in schema.sql
    # Storing timezone-aware datetime is crucial for comparisons
    paused_until = Column(DateTime(timezone=True), nullable=False, index=True) # Added index=True

    def __repr__(self):
        """Provides a developer-friendly representation of the ConversationPause object."""
        # Format timestamp for better readability in logs/debug
        paused_str = self.paused_until.isoformat() if self.paused_until else 'None'
        return f"<ConversationPause(conversation_id='{self.conversation_id}', paused_until='{paused_str}')>"

    def is_active(self) -> bool:
        """
        Checks if the pause is currently active (paused_until is in the future).
        Note: Prefer using db_utils.is_conversation_paused for session-managed checks.
        """
        if not self.paused_until:
            return False # Should not happen due to NOT NULL constraint, but safe check
        # Ensure comparison is timezone-aware
        # Using timezone directly from datetime module for clarity
        return self.paused_until > datetime.datetime.now(timezone.utc)