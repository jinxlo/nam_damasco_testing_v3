# /home/ec2-user/namwoo_app/namwoo_app/celery_app.py

from celery import Celery
from .config import Config

# Initialize Celery with lowercase config keys (Celery 5+ best practice)
celery_app = Celery(
    'namwoo_tasks',
    broker=Config.broker_url,
    backend=Config.result_backend,
    include=['namwoo_app.celery_tasks'],
)

# Basic Celery config (still ok to override explicitly if needed)
celery_app.conf.update(
    task_serializer=getattr(Config, 'task_serializer', 'json'),
    accept_content=getattr(Config, 'accept_content', ['json']),
    result_serializer=getattr(Config, 'result_serializer', 'json'),
    timezone=getattr(Config, 'timezone', 'UTC'),
    enable_utc=getattr(Config, 'enable_utc', True),
    broker_connection_retry_on_startup=True,
)

# --- FLASK APP CONTEXT FOR TASKS (PER TASK, NOT GLOBAL) ---

_flask_app_for_celery_context = None

def get_celery_flask_app():
    """Create or return the Flask app instance for Celery task context."""
    global _flask_app_for_celery_context
    if _flask_app_for_celery_context is None:
        from namwoo_app import create_app  # Import here to avoid circular imports
        _flask_app_for_celery_context = create_app()
    return _flask_app_for_celery_context

class FlaskTask(celery_app.Task):
    """Custom Task base to push Flask app context per task (if needed)."""
    def __call__(self, *args, **kwargs):
        flask_app = get_celery_flask_app()
        with flask_app.app_context():
            return self.run(*args, **kwargs)

# NOTE: Do NOT set celery_app.Task = FlaskTask globally! Only use per-task.
# Example usage in celery_tasks.py:
# @celery_app.task(bind=True, base=FlaskTask)
# def my_task(self, ...):

if __name__ == '__main__':
    print("To start Celery worker: celery -A namwoo_app.celery_app worker -l info")
