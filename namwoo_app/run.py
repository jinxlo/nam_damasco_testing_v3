import os
import sys # <-- Added sys import
import logging

# --- Explicitly add the project root to sys.path --- START ---
# This ensures that the 'namwoo_app' directory can be found as a package
# when this script is run by Gunicorn/systemd, regardless of how the
# initial Python path is configured by the environment.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Explicitly add the project root to sys.path --- END ---

# Now, this import should reliably find the 'namwoo_app' package
from namwoo_app import create_app # Import the app factory function
# from config import Config # Config is usually loaded via create_app

# Get a logger for this entry point script
logger = logging.getLogger(__name__)

# Create the Flask app instance using the factory.
# This also triggers the initialization of logging, extensions, blueprints, etc.
# defined within create_app().
try:
    # Note: The path fix above must happen *before* this line
    app = create_app()
    logger.info("Flask application created successfully via factory.")
except Exception as e:
    # Catch critical errors during app creation (e.g., config loading, early init failures)
    logger.exception("!!! CRITICAL ERROR DURING FLASK APP CREATION !!!")
    # Optionally print to stderr as logging might not be fully set up
    print(f"FATAL: Failed to create Flask app: {e}")
    # Exit here if the app cannot even be created
    # import sys # sys is already imported above
    sys.exit(1)


# --- Main Execution Block ---
# This block runs only when the script is executed directly (e.g., `python run.py`)
# It's typically used for starting the Flask development server.
if __name__ == '__main__':
    # Get host and port from environment variables or use defaults
    # Use 0.0.0.0 to make the server accessible externally (e.g., to Caddy proxy)
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    try:
        port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    except ValueError:
        logger.warning("Invalid FLASK_RUN_PORT value. Using default port 5000.")
        port = 5000

    # Get debug setting from app config (determined by FLASK_ENV)
    debug_mode = app.config.get('DEBUG', False)

    logger.info(f"Starting Flask development server on {host}:{port} (Debug Mode: {debug_mode})")
    print(f"[*] Flask app '{app.name}' running on http://{host}:{port}/")
    print(f"[*] Debug mode is: {'on' if debug_mode else 'off'}")
    print("[!] WARNING: This is a development server. Do not use it in a production deployment.")
    print("[!] Use a production WSGI server like Gunicorn or uWSGI instead.")
    print("    Example (Gunicorn): gunicorn --bind 127.0.0.1:5000 'run:app'")
    print("Press CTRL+C to quit")


    # Run the Flask development server
    # use_reloader=debug_mode handles automatic restarts on code changes in debug mode
    # use_debugger=debug_mode enables the Werkzeug web-based debugger on errors
    try:
        app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
    except Exception as e:
         logger.exception("Flask development server failed to start or crashed.")
         print(f"Error starting development server: {e}")


# --- Gunicorn Integration ---
# When running with Gunicorn (e.g., `gunicorn 'run:app'`), Gunicorn imports this file
# and looks for the `app` variable (which we created above using `create_app`).
# The path modification added at the top ensures the 'from namwoo_app...' import works.
# Gunicorn itself handles the WSGI server part, so the `if __name__ == '__main__':`
# block is *not* executed when running under Gunicorn.

# You can add Gunicorn-specific configurations here if needed, but usually,
# Gunicorn settings are provided via command-line arguments or a config file (gunicorn.conf.py).
# logger.info("Application module loaded (potentially by Gunicorn). 'app' instance is ready.")