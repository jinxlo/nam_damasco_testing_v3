# namwoo_app/utils/db_utils.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session as SQLAlchemySession # Renamed to avoid conflict
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
from typing import Optional, Generator, List, Dict

# Import datetime and timezone for pause logic
import datetime
from datetime import timezone

from ..models import Base
from ..models.conversation_pause import ConversationPause # Ensure this model is correctly defined
from ..config import Config # Import Config to use for logging or other settings if needed

logger = logging.getLogger(__name__)

# --- FIX: Expose the ScopedSessionFactory as a public 'db_session' object. ---
# This is the standard pattern and allows other modules (like celery_tasks)
# to access the .remove() method for guaranteed cleanup.
db_session: scoped_session = None

# Internal "private" globals (never access these outside this file)
_engine = None
_SessionFactory = None


def init_db(app) -> bool:
    """
    Initialize the SQLAlchemy engine and session factories using app config.
    """
    global _engine, _SessionFactory, db_session

    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    if not db_uri:
        logger.error("SQLALCHEMY_DATABASE_URI not configured. Database features will fail.")
        return False

    try:
        db_uri_parts = db_uri.split('@')
        loggable_db_uri = db_uri_parts[-1] if len(db_uri_parts) > 1 else db_uri
        logger.info(f"Attempting to connect to database: {loggable_db_uri}")

        _engine = create_engine(
            db_uri,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=app.config.get('SQLALCHEMY_ECHO', False)
        )
        with _engine.connect() as connection:
            logger.info("Database connection test successful.")

        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        db_session = scoped_session(_SessionFactory)
        logger.info("SQLAlchemy SessionFactory and ScopedSessionFactory (db_session) initialized successfully.")
        return True

    except OperationalError as oe:
        logger.error(f"Database connection failed (OperationalError): {oe}", exc_info=True)
        _engine = None; _SessionFactory = None; db_session = None
        return False
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        _engine = None; _SessionFactory = None; db_session = None
        return False

@contextmanager
def get_db_session() -> Generator[Optional[SQLAlchemySession], None, None]: # Use SQLAlchemySession type hint
    """
    Yields a SQLAlchemy Session, handles commit/rollback, and always removes session from scope.
    Always use via: `with get_db_session() as session:`
    """
    if not db_session:
        logger.error("db_session (ScopedSessionFactory) not initialized. Cannot create DB session.")
        yield None # Make sure to yield None if there's an issue
        return

    session: SQLAlchemySession = db_session()
    logger.debug(f"DB Session {id(session)} acquired from ScopedSessionFactory.")
    try:
        yield session
        session.commit()
        logger.debug(f"DB Session {id(session)} committed.")
    except SQLAlchemyError as e:
        logger.error(f"DB Session {id(session)} SQLAlchemy error: {e}", exc_info=True)
        session.rollback()
        logger.debug(f"DB Session {id(session)} rolled back due to SQLAlchemyError.")
        raise
    except Exception as e:
        logger.error(f"DB Session {id(session)} unexpected error: {e}", exc_info=True)
        session.rollback()
        logger.debug(f"DB Session {id(session)} rolled back due to unexpected error.")
        raise
    finally:
        logger.debug(f"DB Session {id(session)} scope ending. Calling db_session.remove().")
        db_session.remove()
        logger.debug(f"DB Session {id(session)} removed from current scope by ScopedSessionFactory.")

def create_all_tables(app) -> bool:
    """
    Create all tables from SQLAlchemy models and ensure pgvector is present.
    """
    if not _engine:
        logger.error("Database engine not initialized. Cannot create tables.")
        return False
    try:
        logger.info("Attempting to create tables from SQLAlchemy models (if they don't already exist)...")
        # Ensure your ConversationPause model is imported and part of Base.metadata
        # from ..models.conversation_pause import ConversationPause # This should be at the top
        Base.metadata.create_all(bind=_engine)
        logger.info("SQLAlchemy Base.metadata.create_all() executed.")

        with _engine.connect() as connection:
            with connection.begin(): # Ensure transaction for DDL if needed by DB
                logger.info("Ensuring pgvector extension exists in the database...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("pgvector extension check complete.")
        return True
    except Exception as e:
        logger.error(f"Error during create_all_tables: {e}", exc_info=True)
        return False

def fetch_history(session_id: str) -> Optional[List[Dict]]:
    """
    Fetches conversation history for a given session_id.
    """
    with get_db_session() as session:
        if not session: return None
        from ..models.history import ConversationHistory # Keep local import if preferred
        try:
            record = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if record and record.history_data:
                return record.history_data
            return []
        except Exception as e:
            logger.exception(f"Error fetching history for session {session_id}: {e}")
            return None

def save_history(session_id: str, history_list: List[Dict]) -> bool:
    """
    Saves or updates conversation history for a given session_id.
    """
    with get_db_session() as session:
        if not session: return False
        from ..models.history import ConversationHistory # Keep local import if preferred
        try:
            record = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if record:
                record.history_data = history_list
            else:
                record = ConversationHistory(session_id=session_id, history_data=history_list)
                session.add(record)
            # Commit is handled by the context manager
            return True
        except Exception as e:
            logger.exception(f"Error saving history for session {session_id}: {e}")
            return False

# --- NEW FUNCTIONS FOR CONVERSATION PAUSE MANAGEMENT ---

def is_conversation_paused(conversation_id: str) -> bool:
    """Checks if a conversation is currently paused."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot check pause status for conv {conversation_id}: DB session not available.")
            return False # Or raise an error, depending on desired handling
        try:
            now_utc = datetime.datetime.now(timezone.utc)
            pause_record = session.query(ConversationPause)\
                .filter(ConversationPause.conversation_id == conversation_id)\
                .filter(ConversationPause.paused_until > now_utc)\
                .first()
            if pause_record:
                logger.debug(f"Conv {conversation_id} IS paused until {pause_record.paused_until.isoformat()}")
                return True
            else:
                logger.debug(f"Conv {conversation_id} is NOT actively paused.")
                return False
        except Exception as e:
            logger.exception(f"Error checking pause status for conversation {conversation_id}: {e}")
            return False # Default to not paused on error to avoid blocking bot indefinitely

def get_pause_record(conversation_id: str) -> Optional[ConversationPause]:
    """Retrieves the current active pause record for a conversation, if any."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot get pause record for conv {conversation_id}: DB session not available.")
            return None
        try:
            now_utc = datetime.datetime.now(timezone.utc)
            pause_record = session.query(ConversationPause)\
                .filter(ConversationPause.conversation_id == conversation_id)\
                .filter(ConversationPause.paused_until > now_utc)\
                .first()
            return pause_record
        except Exception as e:
            logger.exception(f"Error getting pause record for conversation {conversation_id}: {e}")
            return None

def pause_conversation_for_duration(conversation_id: str, duration_seconds: int):
    """Sets or updates a pause for a conversation."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot pause conv {conversation_id}: DB session not available.")
            return
        try:
            pause_until_time = datetime.datetime.now(timezone.utc) + datetime.timedelta(seconds=duration_seconds)
            
            pause_record = session.query(ConversationPause).filter_by(conversation_id=conversation_id).first()
            
            if pause_record:
                pause_record.paused_until = pause_until_time
                logger.info(f"Updating pause for conversation {conversation_id} until {pause_until_time.isoformat()}.")
            else:
                pause_record = ConversationPause(conversation_id=conversation_id, paused_until=pause_until_time)
                session.add(pause_record)
                logger.info(f"Setting new pause for conversation {conversation_id} until {pause_until_time.isoformat()}.")
            # Commit is handled by the context manager
            logger.debug(f"Pause set/updated in DB for conversation {conversation_id}.")
        except Exception as e:
            logger.exception(f"Error pausing conversation {conversation_id}: {e}")
            # Rollback is handled by the context manager

def unpause_conversation(conversation_id: str):
    """Removes any active pause for a conversation by setting paused_until to a past time or deleting."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot unpause conv {conversation_id}: DB session not available.")
            return
        try:
            # Option 1: Set paused_until to now (or slightly in the past) to effectively unpause
            # now_utc = datetime.datetime.now(timezone.utc)
            # updated_count = session.query(ConversationPause)\
            #     .filter(ConversationPause.conversation_id == conversation_id)\
            #     .update({"paused_until": now_utc - datetime.timedelta(seconds=1)}) # Mark as expired
            # if updated_count > 0:
            #    logger.info(f"Effectively unpaused conversation {conversation_id} by updating paused_until.")
            # else:
            #    logger.info(f"No active pause record found to unpause for conversation {conversation_id} by update.")

            # Option 2: Delete the pause record (simpler if you don't need to keep history of pauses)
            pause_record = session.query(ConversationPause).filter_by(conversation_id=conversation_id).first()
            if pause_record:
                session.delete(pause_record)
                logger.info(f"Deleted pause record for conversation {conversation_id}, effectively unpausing it.")
            else:
                logger.info(f"No pause record found to delete for conversation {conversation_id}.")

            # Commit is handled by the context manager
        except Exception as e:
            logger.exception(f"Error unpausing conversation {conversation_id}: {e}")
            # Rollback is handled by the context manager