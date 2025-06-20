# /home/ec2-user/namwoo_app/namwoo_app/__init__.py
import os
import logging
from logging.config import dictConfig
from flask import Flask
from .config.config import Config
from .utils.logging_utils import JsonFormatter

# --- Logging Configuration (Your original logging setup - Kept as is) ---
log_level_env = os.environ.get('LOG_LEVEL', 'INFO').upper()
_default_basedir_for_logs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir_path = os.path.join(getattr(Config, 'basedir', _default_basedir_for_logs) , 'logs')
os.makedirs(log_dir_path, exist_ok=True)

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': JsonFormatter,
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': log_level_env,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'app_file': {
            'level': log_level_env,
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': os.path.join(log_dir_path, 'app.log'),
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8',
        },
        'sync_file': {
            'level': log_level_env,
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': os.path.join(log_dir_path, 'sync.log'),
            'maxBytes': 5242880,
            'backupCount': 3,
            'encoding': 'utf8',
        },
        'json_file': {
            'level': log_level_env,
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json',
            'filename': getattr(Config, 'LOG_JSON_FILE', os.path.join(log_dir_path, 'app.json')),
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8',
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'app_file', 'json_file'],
            'level': log_level_env,
            'propagate': True
        },
        'werkzeug': {'handlers': ['console', 'app_file', 'json_file'], 'level': 'INFO', 'propagate': False,},
        'sqlalchemy.engine': {'handlers': ['console', 'app_file', 'json_file'], 'level': 'WARNING', 'propagate': False,},
        'apscheduler': {'handlers': ['console', 'app_file', 'json_file'], 'level': 'INFO', 'propagate': False,},
        'sync': {'handlers': ['console', 'sync_file', 'json_file'], 'level': log_level_env, 'propagate': False,},
        'celery': {'handlers': ['console', 'app_file', 'json_file'], 'level': log_level_env, 'propagate': False,},
        'namwoo_app': {'handlers': ['console', 'app_file', 'json_file'], 'level': log_level_env, 'propagate': False}
    }
}
dictConfig(logging_config)
logger = logging.getLogger(__name__) # Logger for this __init__.py module

# Removed _flask_app_instance_for_celery and get_flask_app_for_celery
# as celery_app.py will now manage its own app instance for context.

def create_app(config_class=Config):
    logger.info("--- Creating Flask Application Instance (for Gunicorn/Web or Celery context) ---")
    app = Flask(__name__)
    app.config.from_object(config_class)

    logger.info(f"Flask Environment: {app.config.get('FLASK_ENV', 'not_set')}")
    logger.info(f"Debug Mode: {app.config.get('DEBUG', False)}")

    # Initialize components like database for THIS app instance
    from .utils import db_utils
    db_utils.init_db(app) # Ensure init_db is idempotent or handles being called for multiple app instances
                         # (e.g., one for web, one for Celery's context via its own create_app call)

    logger.info("Dependent services (like OpenAI client) will be initialized as needed or at module level within their respective service files.")

    # --- BLUEPRINT REGISTRATION ---
    # This assumes namwoo_app/api/__init__.py defines 'api_bp' as a global variable
    # within that package, and imports its route modules (routes.py, receiver_routes.py)
    # which then decorate that 'api_bp'.
    from .api import api_bp as api_module_blueprint # Import the blueprint instance
    app.register_blueprint(api_module_blueprint) # Register it
    logger.info(f"Main API Blueprint '{api_module_blueprint.name}' registered under url_prefix: {api_module_blueprint.url_prefix}")
    
    # --- CELERY CONFIGURATION LINKING ---
    # This part updates the Celery app instance's configuration with settings from the Flask app.
    # The actual Flask app context for Celery tasks is handled within celery_app.py by its FlaskTask.
    try:
        from .celery_app import celery_app as celery_application_instance
        
        # Define which Flask config keys should be passed to Celery config
        celery_config_keys_to_pass = [
            'CELERY_BROKER_URL', 
            'CELERY_RESULT_BACKEND', 
            'CELERY_TASK_SERIALIZER', # Example, if you set it in Flask config
            # Add any other FLASK_CONFIG_KEY that Celery needs
        ]
        celery_flask_config = {key: app.config[key] for key in celery_config_keys_to_pass if key in app.config}
        
        if celery_flask_config:
            celery_application_instance.conf.update(celery_flask_config)
            logger.info(f"Celery instance config updated from Flask app config for keys: {list(celery_flask_config.keys())}")
        else:
            logger.info("No specific Celery configurations found in Flask app.config to update Celery instance.")
            
        # The FlaskTask._flask_app setting is now removed from here.
        # celery_app.py's get_celery_flask_app will handle creating its own app instance.

    except ImportError:
        logger.warning("celery_app not found or importable. Celery specific configurations in create_app skipped.")
    except Exception as e_celery_conf:
        logger.error(f"Error during Celery configuration linking in create_app: {e_celery_conf}", exc_info=True)


    # Background Scheduler (APScheduler) - Your original logic
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        if app.config.get('SYNC_INTERVAL_MINUTES', 0) > 0:
            logger.info("Initializing background scheduler (APScheduler)...")
            from .scheduler import tasks as scheduler_tasks 
            scheduler_instance = getattr(app, 'scheduler', None)
            if scheduler_instance is None or not scheduler_instance.running:
                app.scheduler = scheduler_tasks.start_scheduler(app) 
                if app.scheduler and getattr(app.scheduler, 'running', False): 
                    logger.info(f"APScheduler started. Sync interval: {app.config['SYNC_INTERVAL_MINUTES']} minutes.")
                else:
                    logger.warning("APScheduler failed to start or was disabled after attempt.")
            else:
                logger.info("APScheduler already running.")
        else:
            logger.info("Automatic background sync (APScheduler) is disabled (SYNC_INTERVAL_MINUTES <= 0).")
    else:
        logger.debug("APScheduler initialization skipped in Flask debug reloader process.")

    register_cli_commands(app) # Your original CLI registration

    logger.info("--- Namwoo Application Initialization Complete ---")
    return app

# Your original register_cli_commands function - Kept as is
def register_cli_commands(app):
    @app.cli.command("run-sync")
    def run_sync_command():
        logger.info("Manual sync triggered via CLI.")
        print("--- Starting Manual Product Sync ---")
        try:
            with app.app_context(): 
                from .scheduler import tasks as scheduler_tasks
                scheduler_tasks.run_sync_logic(app, full_resync=True) 
            print("--- Manual Sync Finished ---")
            print("Check logs/sync.log for details.")
            logger.info("Manual sync finished successfully via CLI.")
        except Exception as e:
            logger.exception("Error during manual sync via CLI.")
            print(f"An error occurred during manual sync: {e}")

    @app.cli.command("create-db")
    def create_db_command():
        logger.info("Database table creation triggered via CLI.")
        print("--- Creating Database Tables ---")
        try:
            with app.app_context(): 
                from .utils import db_utils
                from .models import Base 
                if db_utils.engine:
                    print("Creating tables from SQLAlchemy models...")
                    Base.metadata.create_all(bind=db_utils.engine)
                    print("Database tables (from models) created successfully.")
                    logger.info("Database tables (from models) created successfully via CLI.")

                    from sqlalchemy import text
                    with db_utils.engine.connect() as connection:
                        with connection.begin(): 
                            logger.info("Ensuring pgvector extension exists in the database (CLI)...")
                            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                            logger.info("pgvector extension check complete (CLI).")
                else:
                    print("Error: Database engine not initialized.")
                    logger.error("Database engine not initialized in create-db command.")
        except Exception as e:
            logger.exception("Error during database table creation via CLI.")
            print(f"An error occurred during table creation: {e}")

    logger.info("Custom CLI commands registered.")

# ---- THIS IS THE ONLY CHANGE: Ensure Celery Tasks Are Imported ----
import namwoo_app.celery_tasks
