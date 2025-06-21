# NAMWOO/api/receiver_routes.py

import logging
from flask import request, jsonify, current_app

# Import the NEW, EFFICIENT batch processing Celery task
from ..celery_tasks import process_products_batch_task

from . import api_bp

logger = logging.getLogger(__name__)

# This helper function is still useful for converting keys, so we keep it.
def _convert_api_input_to_snake_case_for_task(data_camel: dict) -> dict:
    """
    Converts a dictionary with camelCase keys (from API input)
    to snake_case keys (for the Celery task's Pydantic model).
    """
    if not data_camel or not isinstance(data_camel, dict):
        logger.warning("Received non-dict or empty data for camel_to_snake conversion.")
        return {}

    key_map = {
        "itemCode": "item_code",
        "itemName": "item_name",
        "description": "description",
        "specifitacion": "specifitacion",
        "stock": "stock",
        "price": "price",
        "priceBolivar": "price_bolivar",
        "category": "category",
        "subCategory": "sub_category",
        "brand": "brand",
        "line": "line",
        "itemGroupName": "item_group_name",
        "whsName": "warehouse_name",
        "branchName": "branch_name",
    }

    data_snake = {}
    for camel_key, value in data_camel.items():
        if snake_key := key_map.get(camel_key):
            data_snake[snake_key] = value
        else:
            logger.debug(f"Unmapped key '{camel_key}' in product data from fetcher. Ignoring.")

    return data_snake


@api_bp.route('/receive-products', methods=['POST'])
def receive_data():
    """
    Receives a JSON list of product entries from the Fetcher service.
    Validates the API token.
    Enqueues ONE SINGLE Celery task for the entire batch of products.
    Returns an HTTP 202 Accepted response.
    """
    auth_token = request.headers.get('X-API-KEY')
    expected_token = current_app.config.get('DAMASCO_API_SECRET')

    if not expected_token:
        logger.critical("DAMASCO_API_SECRET not configured. Cannot authenticate request.")
        return jsonify({"status": "error", "message": "Server misconfiguration"}), 500

    if not auth_token or auth_token != expected_token:
        logger.warning("Unauthorized /receive-products request. Invalid or missing API token.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if not request.is_json:
        logger.error("Invalid request: Content-Type not application/json.")
        return jsonify({"status": "error", "message": "Content-Type must be application/json."}), 415

    try:
        damasco_payload_camel_case = request.get_json()
    except Exception as e_json:
        logger.error(f"Invalid JSON received: {e_json}", exc_info=True)
        return jsonify({"status": "error", "message": f"Invalid JSON format: {e_json}"}), 400

    if not isinstance(damasco_payload_camel_case, list):
        logger.error("Invalid data received. Expected a JSON list of product entries.")
        return jsonify({"status": "error", "message": "Invalid format: Expected a list."}), 400

    if not damasco_payload_camel_case:
        logger.info("Received an empty list of products. No action taken.")
        return jsonify({
            "status": "accepted",
            "message": "Received empty product list. No tasks enqueued.",
            "tasks_enqueued": 0
        }), 202

    logger.info(f"Received {len(damasco_payload_camel_case)} product entries. Preparing batch for Celery.")

    # --- REFACTORED BATCH PROCESSING LOGIC ---
    # 1. Convert all items in the batch to snake_case in one go.
    products_batch_snake = []
    for item_camel in damasco_payload_camel_case:
        if isinstance(item_camel, dict):
            products_batch_snake.append(_convert_api_input_to_snake_case_for_task(item_camel))
        else:
            logger.warning(f"Item in payload is not a dictionary. Skipping: {str(item_camel)[:100]}")
    
    # 2. Enqueue ONE task for the entire batch.
    if not products_batch_snake:
        logger.warning("No valid product dictionaries found in payload after filtering.")
        return jsonify({"status": "accepted", "message": "No valid items to process.", "tasks_enqueued": 0}), 202
    
    try:
        # Pass the entire list to our new, robust batch task.
        process_products_batch_task.delay(products_batch_snake)
        
        enqueued_count = len(products_batch_snake)
        response_summary = {
            "status": "accepted",
            "message": "Product data batch received and one task enqueued for processing.",
            "total_payload_items": len(damasco_payload_camel_case),
            "items_in_enqueued_batch": enqueued_count
        }
        logger.info(f"Successfully enqueued one batch task for {enqueued_count} products.")
        return jsonify(response_summary), 202

    except Exception as e_celery:
        logger.critical(f"CRITICAL: Failed to enqueue batch task for Celery: {e_celery}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Failed to enqueue task to the message broker. Check broker connectivity."
        }), 503 # Service Unavailable is appropriate if the broker is down.