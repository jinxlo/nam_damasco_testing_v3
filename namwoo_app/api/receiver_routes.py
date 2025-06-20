# NAMWOO/api/receiver_routes.py

import logging
from flask import request, jsonify, current_app
# Removed SQLAlchemyError as direct DB operations are moved to Celery
# from sqlalchemy.exc import SQLAlchemyError

# Removed direct service/model imports not used by this enqueuing route
# from ..utils import db_utils
# from ..services import product_service
# from ..services.openai_service import generate_product_embedding
# from ..models.product import Product

# Import the Celery task
from ..celery_tasks import process_product_item_task # Assuming this is the correct task

from . import api_bp  # Use the main API blueprint

logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# Helper to convert incoming camelCase from Fetcher to snake_case for Celery task
# This ensures the Celery task receives data in the format its Pydantic model expects.
def _convert_api_input_to_snake_case_for_task(data_camel: dict) -> dict:
    """
    Converts a dictionary with camelCase keys (from API input)
    to snake_case keys (for Celery task's Pydantic model).
    """
    if not data_camel or not isinstance(data_camel, dict): # Added type check for safety
        logger.warning("Received non-dict or empty data for camel_to_snake conversion.")
        return {}
    
    # Define the mapping from camelCase (API input from Fetcher)
    # to snake_case (as expected by DamascoProductDataSnake Pydantic model in celery_tasks.py)
    key_map = {
        "itemCode": "item_code",
        "itemName": "item_name",
        "description": "description", # Pass raw HTML description
        "specifitacion": "specifitacion", # Pass specification field
        "stock": "stock",
        "price": "price", # This is the primary price, e.g., USD
        "priceBolivar": "price_bolivar", # <<< --- ADDED THIS MAPPING ---
        "category": "category",
        "subCategory": "sub_category",
        "brand": "brand",
        "line": "line",
        "itemGroupName": "item_group_name",
        "whsName": "warehouse_name", # Pydantic model expects 'warehouse_name'
        "branchName": "branch_name",
        # Add any other fields that your Fetcher might send and your Pydantic model expects
    }
    
    data_snake = {}
    for camel_key, value in data_camel.items():
        snake_key = key_map.get(camel_key)
        if snake_key: # Only include keys that are defined in our map
            data_snake[snake_key] = value
        else:
            # Log unmapped keys if they are unexpected and you want to track them
            # This helps identify if Fetcher is sending new fields not yet handled.
            # Be careful not to log sensitive data if `value` could contain it.
            # For known, intentionally unmapped keys, this log can be noisy.
            # Example: if 'someOtherFetcherField' is sent but not needed.
            if camel_key not in ["someOtherKnownButUnusedField1", "anotherKnownUnusedField"]: # Example condition
                 logger.debug(f"Unmapped key '{camel_key}' in product data from fetcher. Value: '{str(value)[:50]}...'. Will not be passed to Celery task unless Pydantic 'extra=allow' and Celery task handles it directly.")
            pass # Current behavior: ignoring unmapped keys not in key_map
    
    # Log what is being prepared for Celery for easier debugging
    logger.debug(f"Converted API input for itemCode '{data_camel.get('itemCode', 'N/A')}' to snake_case for Celery: {data_snake}")
    return data_snake


@api_bp.route('/receive-products', methods=['POST'])
def receive_data():
    """
    Receives JSON payload of product entries from Fetcher EC2.
    Validates API token.
    For each product entry, converts to snake_case and enqueues a Celery task for processing.
    Returns an HTTP 202 Accepted response.
    """
    auth_token = request.headers.get('X-API-KEY')
    expected_token = current_app.config.get('DAMASCO_API_SECRET')

    if not expected_token:
        logger.critical("DAMASCO_API_SECRET not configured in app settings. Cannot authenticate request.")
        return jsonify({"status": "error", "message": "Server misconfiguration - API Secret missing"}), 500

    if not auth_token or auth_token != expected_token:
        logger.warning("Unauthorized /receive-products request. Invalid or missing API token.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        # force=True can be risky if content-type isn't application/json.
        # Consider request.is_json and then request.get_json()
        if not request.is_json:
            logger.error("Invalid /receive-products request: Content-Type not application/json.")
            return jsonify({"status": "error", "message": "Invalid request: Content-Type must be application/json."}), 415
        
        damasco_payload_camel_case = request.get_json() 
    except Exception as e_json: # Catch potential JSON parsing errors
        logger.error(f"Invalid JSON received for /receive-products: {e_json}", exc_info=True)
        return jsonify({"status": "error", "message": f"Invalid JSON format: {e_json}"}), 400


    if not isinstance(damasco_payload_camel_case, list):
        logger.error("Invalid data received for /receive-products. Expected a JSON list of product entries.")
        return jsonify({"status": "error", "message": "Invalid JSON format: Expected a list."}), 400

    if not damasco_payload_camel_case: # Empty list
        logger.info("Received an empty list of products. No action taken.")
        return jsonify({
            "status": "accepted", 
            "message": "Received empty product list. No tasks enqueued.",
            "tasks_enqueued": 0
        }), 202 # 202 is still appropriate as the request was accepted, even if no work was done.

    logger.info(f"Received {len(damasco_payload_camel_case)} product entries from fetcher. Attempting to enqueue for Celery processing.")

    enqueued_count = 0
    failed_to_enqueue_count = 0
    items_skipped_validation_count = 0

    for i, product_entry_camel in enumerate(damasco_payload_camel_case):
        item_code_log = product_entry_camel.get('itemCode', 'N/A_in_payload')
        whs_name_log = product_entry_camel.get('whsName', 'N/A_in_payload')
        log_prefix = f"Receiver Entry [{i+1}/{len(damasco_payload_camel_case)}] ({item_code_log} @ {whs_name_log}):"

        if not isinstance(product_entry_camel, dict):
            logger.warning(f"{log_prefix} Item in payload is not a dictionary. Skipping enqueue.")
            items_skipped_validation_count += 1
            continue

        # Convert incoming camelCase product data to snake_case for the Celery task
        # This product_data_snake is what celery_tasks.DamascoProductDataSnake will receive.
        # damasco_service.py is NOT called here; this route prepares data FOR Celery,
        # and damasco_service.py might be used by sync_service.py or other direct ingestion paths.
        # The README implied this route calls Celery directly after transformation.
        product_data_snake = _convert_api_input_to_snake_case_for_task(product_entry_camel)

        # Basic check: Ensure essential keys for task identification and Pydantic model are present
        if not product_data_snake.get("item_code") or not product_data_snake.get("warehouse_name"):
            logger.warning(f"{log_prefix} Missing essential 'item_code' or 'warehouse_name' after conversion to snake_case. Skipping enqueue. Original: {product_entry_camel}, Converted: {product_data_snake}")
            items_skipped_validation_count += 1
            continue
        
        try:
            # Enqueue the task with the snake_case data dictionary
            # The Celery task (process_product_item_task) will then use its Pydantic model
            # to validate and convert types (e.g., price_bolivar from string/float to Decimal).
            process_product_item_task.delay(product_data_snake)
            logger.info(f"{log_prefix} Successfully enqueued for Celery processing.")
            enqueued_count += 1
        except Exception as e_celery: # Catch errors during .delay() if broker is down etc.
            logger.error(f"{log_prefix} Failed to enqueue task for Celery: {e_celery}", exc_info=True)
            failed_to_enqueue_count +=1
            
    response_summary = {
        "status": "accepted",
        "message": "Product data received and tasks enqueued for processing.",
        "total_payload_items": len(damasco_payload_camel_case),
        "tasks_successfully_enqueued": enqueued_count,
        "items_skipped_pre_enqueue_validation": items_skipped_validation_count,
        "tasks_failed_to_enqueue": failed_to_enqueue_count
    }
    logger.info(f"Enqueue summary for /receive-products: {response_summary}")
    return jsonify(response_summary), 202

# --- /health endpoint REMOVED (as per your original file) ---