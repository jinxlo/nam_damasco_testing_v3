# namwoo_app/api/__init__.py
from flask import Blueprint

api_bp = Blueprint(
    'api',
    __name__,
    url_prefix='/api'
)

# Import all route modules that use this blueprint HERE
from . import routes          # Assumes routes.py contains "from . import api_bp"
from . import receiver_routes # Assumes receiver_routes.py contains "from . import api_bp"