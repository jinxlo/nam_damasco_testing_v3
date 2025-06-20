# NAMWOO/scheduler/tasks.py

import logging
import time
import traceback
import requests  # <-- ADDED for making API calls
import json      # <-- ADDED for handling JSON data

from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Flask

# --- IMPORTANT: We NO LONGER import sync_service ---
# from ..services import sync_service

# --- We now need to import a function to get the product data ---
# This function should live somewhere central, maybe in a new 'data_fetcher.py' service,
# or for now, we can define a placeholder for it here.
# Let's assume you create a new service for this.
from ..services import data_fetcher_service  # <-- ASSUMING YOU CREATE THIS FILE

logger = logging.getLogger(__name__)
sync_logger = logging.getLogger('sync')

# Global scheduler instance
scheduler = None
SYNC_JOB_ID = 'product_sync_trigger' # Renamed to reflect its new role

_sync_running = False

def run_sync_logic(app: Flask):
    """
    The core logic that triggers the sync process by calling the API endpoint.
    This replaces the old method of running a long process.
    """
    global _sync_running
    if _sync_running:
        sync_logger.warning(f"Sync trigger job '{SYNC_JOB_ID}' attempted to start while already running. Skipping.")
        return

    _sync_running = True
    sync_logger.info(f"--- Starting sync trigger job '{SYNC_JOB_ID}' ---")
    start_time = time.time()
    
    try:
        # We still need the app context for config values
        with app.app_context():
            # 1. Fetch the product data from the primary source
            sync_logger.info("Fetching latest product data from Damasco API...")
            
            # This function needs to be created. It's the logic that was probably
            # hidden inside your external "Fetcher Service" before.
            product_list = data_fetcher_service.get_all_products_from_source()
            
            if not product_list:
                sync_logger.warning("No products returned from the data source. Ending sync trigger.")
                _sync_running = False
                return

            sync_logger.info(f"Fetched {len(product_list)} products. Sending to API endpoint for processing.")

            # 2. Get API URL and Key from Flask config
            # Ensure your app config has these values
            api_url = app.config.get('INTERNAL_API_URL', 'http://127.0.0.1:5100/api/receive-products')
            api_key = app.config.get('DAMASCO_API_SECRET')

            if not api_key:
                sync_logger.error("DAMASCO_API_SECRET is not configured. Cannot trigger sync.")
                _sync_running = False
                return

            # 3. Call your own API to enqueue the Celery tasks
            headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
            
            # Using a session for potential keep-alive benefits, though not strictly necessary
            with requests.Session() as s:
                response = s.post(
                    api_url,
                    headers=headers,
                    data=json.dumps(product_list, default=str) # Use default=str to handle Decimals, etc.
                )

            if response.status_code == 202:
                sync_logger.info(f"Successfully sent {len(product_list)} products to the Celery pipeline. API response: {response.status_code}")
            else:
                sync_logger.error(f"Failed to send products to the API endpoint. Status: {response.status_code}, Response: {response.text[:200]}")

    except Exception as e:
        sync_logger.error(f"!!! Sync trigger job '{SYNC_JOB_ID}' failed !!!", exc_info=True)
    finally:
        duration = time.time() - start_time
        sync_logger.info(f"--- Finished sync trigger job '{SYNC_JOB_ID}' in {duration:.2f}s ---")
        _sync_running = False # Release the lock

def scheduled_sync_job(app: Flask):
    """
    Wrapper function specifically designed to be called by APScheduler.
    """
    sync_logger.info(f"Scheduler triggered for job '{SYNC_JOB_ID}'.")
    run_sync_logic(app)


def start_scheduler(app: Flask) -> Optional[BackgroundScheduler]:
    """
    Initializes and starts the APScheduler for background tasks.
    """
    global scheduler
    if scheduler and scheduler.running:
        logger.warning("Scheduler is already running.")
        return scheduler

    interval_minutes = app.config.get('SYNC_INTERVAL_MINUTES', 0)
    if interval_minutes <= 0:
        logger.info("Background sync scheduler is disabled via config (SYNC_INTERVAL_MINUTES <= 0).")
        return None

    try:
        logger.info(f"Initializing APScheduler (BackgroundScheduler). Sync interval: {interval_minutes} minutes.")
        scheduler = BackgroundScheduler(daemon=True)

        scheduler.add_job(
            func=scheduled_sync_job,
            args=[app],
            trigger=IntervalTrigger(minutes=interval_minutes),
            id=SYNC_JOB_ID,
            name='Product Sync Trigger', # Renamed job
            replace_existing=True,
            misfire_grace_time=300
        )

        scheduler.start()
        logger.info(f"APScheduler started successfully. Job '{SYNC_JOB_ID}' scheduled.")

        import atexit
        atexit.register(lambda: stop_scheduler())

        return scheduler

    except Exception as e:
        logger.exception(f"Failed to initialize or start APScheduler: {e}")
        scheduler = None
        return None


def stop_scheduler():
    """Stops the APScheduler gracefully if it is running."""
    global scheduler
    if scheduler and scheduler.running:
        logger.info("Attempting to shut down APScheduler...")
        try:
            scheduler.shutdown()
            logger.info("APScheduler shut down successfully.")
        except Exception as e:
            logger.exception(f"Error shutting down APScheduler: {e}")
    elif scheduler:
        logger.info("APScheduler was initialized but not running.")
    else:
        logger.info("APScheduler was not initialized.")


def get_scheduler_status() -> dict:
    """Returns the current status of the scheduler and its jobs."""
    status = {"scheduler_running": False, "jobs": []}
    if scheduler and scheduler.running:
        status["scheduler_running"] = True
        try:
            jobs = scheduler.get_jobs()
            for job in jobs:
                status["jobs"].append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
        except Exception as e:
            logger.error(f"Failed to get job details from scheduler: {e}")
            status["error"] = str(e)
    return status