# /home/ec2-user/namwoo_app/namwoo_app/celery_tasks.py

import logging
from typing import List, Optional, Dict, Any
import re
import numpy as np # Ensure numpy is imported
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ValidationError, validator

from celery.exceptions import Ignore, MaxRetriesExceededError
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from celery.exceptions import OperationalError as CeleryOperationalError


from .celery_app import celery_app, FlaskTask
from .services import product_service, openai_service
from .services import llm_processing_service
# --- FIX: Import db_utils to access the db_session for cleanup. ---
from .utils import db_utils, text_utils # text_utils is used by Product model
from .models.product import Product
from .config import Config

logger = logging.getLogger(__name__)

# --- Pydantic Model for Validating Incoming Snake_Case Product Data ---
class DamascoProductDataSnake(BaseModel):
    item_code: str
    item_name: str
    description: Optional[str] = None
    specifitacion: Optional[str] = None # <<< NEW FIELD ADDED
    stock: int
    price: Optional[Decimal] = None # Will be Decimal after validation
    price_bolivar: Optional[Decimal] = None # <<< NEW FIELD: Will be Decimal after validation
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand: Optional[str] = None
    line: Optional[str] = None
    item_group_name: Optional[str] = None
    warehouse_name: str # Required for ID generation
    branch_name: Optional[str] = None
    original_input_data: Optional[Dict[str, Any]] = None # Set by the task

    @validator('price', 'price_bolivar', pre=True, allow_reuse=True)
    def validate_prices_to_decimal(cls, v: Any) -> Optional[Decimal]:
     if v is None:
        return None
     if isinstance(v, (int, float, str)): # Handles data from various sources
        try:
           return Decimal(str(v)) # Convert to string first for exact Decimal representation
        except InvalidOperation:
           logger.warning(f"Pydantic validator: Could not convert price value '{v}' to Decimal. Setting to None.")
           return None
     if isinstance(v, Decimal): # If already Decimal (e.g. from damasco_service)
        return v
     logger.warning(f"Pydantic validator: Unexpected type '{type(v)}' for price field. Value: {v}. Setting to None.")
     return None

    class Config:
     extra = 'allow' # Allows 'original_input_data' to be added later
     validate_assignment = True


def _convert_snake_to_camel_case(data_snake: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts snake_case keys from Pydantic model output to camelCase.
    Decimal price values are converted to float for the output camelCase dictionary.
    """
    if not data_snake:
     return {}
    key_map = {
     "item_code": "itemCode",
     "item_name": "itemName",
     "description": "description",
     "specifitacion": "specifitacion", # <<< NEW FIELD ADDED
     "stock": "stock",
     "price": "price", # USD price
     "price_bolivar": "priceBolivar", # <<< CRITICAL: Ensure this mapping is present
     "category": "category",
     "sub_category": "subCategory",
     "brand": "brand",
     "line": "line",
     "item_group_name": "itemGroupName",
     "warehouse_name": "whsName", # Pydantic model has 'warehouse_name'
     "branch_name": "branchName",
    }
    data_camel = {}
    for snake_key, value in data_snake.items():
     # Skip internal fields like 'original_input_data' if they are part of data_snake
     if snake_key == "original_input_data":
        continue

     camel_key = key_map.get(snake_key)
     
     val_to_store = value
     # Convert Decimal (from Pydantic model) to float for the damasco_product_data_camel dict
     # because product_service.py (as last modified for minimal changes) expects to
     # get float-like numbers from this dict and then converts them back to Decimal.
     if isinstance(value, Decimal):
        val_to_store = float(value)

     if camel_key:
        data_camel[camel_key] = val_to_store
     # else: # Optionally log unmapped keys if needed for debugging, but only for non-internal fields
        # if snake_key not in ["original_input_data"]:
        # logger.debug(f"Snake_case key '{snake_key}' not in key_map, not included in camelCase dict during _convert_snake_to_camel_case.")
    
    # Ensure description is present if it was in data_snake (though key_map should handle it)
    if "description" in data_snake and "description" not in data_camel and key_map.get("description"):
     data_camel[key_map.get("description")] = data_snake["description"]
        
    return data_camel

def _normalize_string_for_id_parts(value: Any) -> Optional[str]:
    if value is None:
     return None
    s = str(value).strip()
    return s if s else None

def _generate_product_location_id(item_code: str, warehouse_name: str) -> Optional[str]:
    norm_item_code = _normalize_string_for_id_parts(item_code)
    norm_whs_name = _normalize_string_for_id_parts(warehouse_name)

    if not norm_item_code or not norm_whs_name: # Pydantic should ensure these are non-empty strings
     logger.error(f"ID Generation Error: item_code ('{item_code}') or whs_name ('{warehouse_name}') became empty. This is unexpected after Pydantic validation.")
     return None
    
    sanitized_whs_name = re.sub(r'[^a-zA-Z0-9_-]', '_', norm_whs_name)
    product_location_id = f"{norm_item_code}_{sanitized_whs_name}".lower()
    
    max_len = 512 
    if len(product_location_id) > max_len:
     logger.warning(f"Generated product_location_id '{product_location_id}' for item '{norm_item_code}' exceeds max length {max_len}. Truncating.")
     product_location_id = product_location_id[:max_len]
     
    return product_location_id

# --- TASKS ---

@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.process_product_item_task',
    max_retries=Config.CELERY_TASK_MAX_RETRIES if hasattr(Config, 'CELERY_TASK_MAX_RETRIES') else 3,
    default_retry_delay=Config.CELERY_TASK_RETRY_DELAY if hasattr(Config, 'CELERY_TASK_RETRY_DELAY') else 300,
    acks_late=True
)
def process_product_item_task(self, product_data_dict_snake: Dict[str, Any]):
    task_id = self.request.id
    item_code_log_initial = product_data_dict_snake.get('item_code', 'N/A')
    whs_name_log_initial = product_data_dict_snake.get('warehouse_name', 'N/A') # Key from receiver
    item_identifier_for_log = f"{item_code_log_initial}_{whs_name_log_initial}"
    
    logger.info(f"Task {task_id}: Starting processing for initial item identifier: {item_identifier_for_log}")

    processing_summary_logs = {
     "task_id": str(task_id),
     "item_identifier": item_identifier_for_log,
     "status": "pending",
     "validation": "pending",
     "summarization_action": "not_applicable",
     "embedding_action": "not_applicable",
     "db_operation": "pending",
     "final_message": ""
    }

    try:
     with db_utils.get_db_session() as session:
        try:
           validated_product_snake = DamascoProductDataSnake(**product_data_dict_snake)
           item_identifier_for_log = f"{validated_product_snake.item_code}_{validated_product_snake.warehouse_name}" # Update with validated data
           processing_summary_logs["item_identifier"] = item_identifier_for_log
           
           validated_product_snake.original_input_data = product_data_dict_snake.copy() 
           
           data_for_conversion_snake = validated_product_snake.model_dump(exclude_unset=True, exclude_none=True) 
           logger.debug(f"Task {task_id} ({item_identifier_for_log}): Pydantic validation successful.")
           processing_summary_logs["validation"] = "success"
        except ValidationError as val_err:
           error_details = val_err.errors()
           logger.error(f"Task {task_id} ({item_identifier_for_log}): Pydantic validation error: {error_details}")
           processing_summary_logs["validation"] = f"failed: {error_details}"
           processing_summary_logs["status"] = "ignored_validation_error"
           logger.debug(f"Task {task_id} ({item_identifier_for_log}): Data causing Pydantic validation error: {product_data_dict_snake}")
           raise Ignore("Pydantic validation failed")

        product_data_camel = _convert_snake_to_camel_case(data_for_conversion_snake)
        logger.debug(f"Task {task_id} ({item_identifier_for_log}): Converted to camelCase keys for product_service: {list(product_data_camel.keys())}")


        item_code_for_id = validated_product_snake.item_code
        whs_name_for_id = validated_product_snake.warehouse_name
        
        product_location_id = _generate_product_location_id(item_code_for_id, whs_name_for_id)

        if not product_location_id:
           logger.error(f"Task {task_id} ({item_identifier_for_log}): Failed to generate product_location_id. Cannot proceed.")
           processing_summary_logs["status"] = "ignored_id_generation_failed"
           raise Ignore("ID generation failed")
        
        processing_summary_logs["product_location_id"] = product_location_id

        existing_product_details = None
        try:
           existing_product_db_entry = session.query(
               Product.description,
               Product.specifitacion,
               Product.llm_summarized_description,
               Product.searchable_text_content,
               Product.embedding
           ).filter_by(id=product_location_id).first()

           if existing_product_db_entry:
               existing_product_details = {
                  "description": existing_product_db_entry.description,
                  "specifitacion": existing_product_db_entry.specifitacion,
                  "llm_summarized_description": existing_product_db_entry.llm_summarized_description,
                  "searchable_text_content": existing_product_db_entry.searchable_text_content,
                  "embedding": existing_product_db_entry.embedding
               }
               logger.debug(f"Task {task_id} ({product_location_id}): Found existing entry details.")
           else:
               logger.debug(f"Task {task_id} ({product_location_id}): No existing entry found in DB.")
        except (SQLAlchemyOperationalError, CeleryOperationalError) as e_db_op_read:
           logger.error(f"Task {task_id} ({product_location_id}): Retriable DB/Broker error reading existing entry: {e_db_op_read}", exc_info=True)
           raise self.retry(exc=e_db_op_read)
        except Exception as e_read:
           logger.error(f"Task {task_id} ({product_location_id}): Non-retriable error reading existing entry: {e_read}", exc_info=True)
           processing_summary_logs["status"] = "failed_db_read_error"
           raise Ignore(f"Failed to read existing product details due to non-retriable error: {e_read}")

        llm_summary_to_use: Optional[str] = None
        raw_html_incoming = validated_product_snake.description
        item_name_for_log = validated_product_snake.item_name

        needs_new_summary = False
        if existing_product_details:
           llm_summary_to_use = existing_product_details["llm_summarized_description"]
           incoming_plain = text_utils.strip_html_to_text(raw_html_incoming or "")
           existing_plain = text_utils.strip_html_to_text(existing_product_details["description"] or "")
           if incoming_plain != existing_plain:
               needs_new_summary = True
               processing_summary_logs["summarization_action"] = "needed_html_changed"
               logger.info(
                   f"Task {task_id} ({product_location_id}): Plain text description changed for '{item_name_for_log}'. New summary needed."
               )
           elif not existing_product_details["llm_summarized_description"] and raw_html_incoming:
               needs_new_summary = True
               processing_summary_logs["summarization_action"] = "needed_summary_missing"
               logger.info(f"Task {task_id} ({product_location_id}): LLM summary missing for '{item_name_for_log}'. New summary needed.")
           else:
               processing_summary_logs["summarization_action"] = "reused_existing"
               logger.info(f"Task {task_id} ({product_location_id}): Re-using stored LLM summary for '{item_name_for_log}'.")
        elif raw_html_incoming:
           needs_new_summary = True
           processing_summary_logs["summarization_action"] = "needed_new_product_with_html"
           logger.info(f"Task {task_id} ({product_location_id}): New product with HTML description for '{item_name_for_log}'. New summary needed.")
        else:
           processing_summary_logs["summarization_action"] = "skipped_no_html"
           logger.info(f"Task {task_id} ({product_location_id}): No HTML description for '{item_name_for_log}'. Summarization not applicable.")

        if needs_new_summary and raw_html_incoming:
           logger.info(f"Task {task_id} ({product_location_id}): Attempting LLM summarization for '{item_name_for_log}'.")
           try:
               llm_summary_to_use = llm_processing_service.generate_llm_product_summary(
                  html_description=raw_html_incoming,
                  item_name=item_name_for_log
               )
               if llm_summary_to_use:
                  logger.info(f"Task {task_id} ({product_location_id}): New LLM summary generated. Preview: '{llm_summary_to_use[:100]}...'")
                  processing_summary_logs["summarization_action"] += "_success"
               else:
                  logger.warning(f"Task {task_id} ({product_location_id}): LLM summarization returned no content for '{item_name_for_log}'.")
                  llm_summary_to_use = None
                  processing_summary_logs["summarization_action"] += "_failed_empty_result"
           except Exception as e_summ:
               logger.error(f"Task {task_id} ({product_location_id}): LLM summarization failed for '{item_name_for_log}'. Error: {e_summ}", exc_info=True)
               processing_summary_logs["summarization_action"] += f"_exception: {str(e_summ)[:100]}"
               if existing_product_details: 
                  llm_summary_to_use = existing_product_details["llm_summarized_description"] 
               else:
                  llm_summary_to_use = None
        elif not raw_html_incoming: 
           llm_summary_to_use = None

        text_to_embed = Product.prepare_text_for_embedding(
           damasco_product_data=product_data_camel,
           llm_generated_summary=llm_summary_to_use,
           raw_html_description_for_fallback=raw_html_incoming
        )

        if not text_to_embed:
           logger.warning(f"Task {task_id} ({product_location_id}): No text content for embedding. Skipping item.")
           processing_summary_logs["embedding_action"] = "skipped_no_text_to_embed"
           processing_summary_logs["status"] = "ignored_no_text_for_embedding"
           raise Ignore("No text to embed") 
        
        logger.debug(f"Task {task_id} ({product_location_id}): Text prepared for embedding (first 100 chars): '{text_to_embed[:100]}...'")

        embedding_vector_to_pass: Optional[List[float]] = None
        generate_new_embedding = True

        if existing_product_details and \
          existing_product_details["searchable_text_content"] == text_to_embed and \
          existing_product_details["embedding"] is not None:
           
           if isinstance(existing_product_details["embedding"], np.ndarray):
               embedding_vector_to_pass = existing_product_details["embedding"].tolist()
           elif isinstance(existing_product_details["embedding"], list):
               embedding_vector_to_pass = existing_product_details["embedding"]
           else:
               logger.warning(f"Task {task_id} ({product_location_id}): Existing embedding type {type(existing_product_details['embedding'])} unexpected. Generating new.")
               
           if embedding_vector_to_pass is not None:
               generate_new_embedding = False
               logger.info(f"Task {task_id} ({product_location_id}): Re-using existing embedding.")
               processing_summary_logs["embedding_action"] = "reused_existing"

        if generate_new_embedding:
           reason = "new_product_or_no_valid_existing_embedding"
           if existing_product_details:
               if existing_product_details["searchable_text_content"] != text_to_embed: reason = "searchable_text_changed"
               elif existing_product_details["embedding"] is None: reason = "existing_embedding_missing"
               elif not (isinstance(existing_product_details["embedding"], list) or isinstance(existing_product_details["embedding"], np.ndarray)): reason = "existing_embedding_invalid_type"
           
           logger.info(f"Task {task_id} ({product_location_id}): Generating new embedding. Reason: {reason}.")
           try:
               newly_generated_embedding = openai_service.generate_product_embedding(text_to_embed)
               if newly_generated_embedding is None:
                  logger.error(f"Task {task_id} ({product_location_id}): Embedding service returned None. Retrying.")
                  processing_summary_logs["embedding_action"] = f"generated_failed_service_returned_none ({reason})"
                  raise self.retry(exc=Exception("Embedding service returned None"))
               
               embedding_vector_to_pass = newly_generated_embedding
               logger.info(f"Task {task_id} ({product_location_id}): New embedding generated successfully.")
               processing_summary_logs["embedding_action"] = f"generated_new ({reason})"
           except Exception as e_embed:
               logger.error(f"Task {task_id} ({product_location_id}): Embedding generation failed: {e_embed}", exc_info=True)
               processing_summary_logs["embedding_action"] = f"generated_exception ({reason}): {str(e_embed)[:100]}"
               raise self.retry(exc=e_embed)

        if embedding_vector_to_pass is None and text_to_embed:
           logger.error(f"Task {task_id} ({product_location_id}): Embedding vector is None after attempts. Ignoring item.")
           processing_summary_logs["embedding_action"] = "final_state_none_vector_unexpected"
           processing_summary_logs["status"] = "ignored_embedding_failed_critically"
           raise Ignore("Critical failure obtaining embedding vector.")

        success, op_type_or_error_msg = product_service.add_or_update_product_in_db(
           session=session,
           product_location_id=product_location_id,
           damasco_product_data_camel=product_data_camel,
           embedding_vector=embedding_vector_to_pass,
           text_used_for_embedding=text_to_embed,
           llm_summarized_description_to_store=llm_summary_to_use
        )

        processing_summary_logs["db_operation"] = op_type_or_error_msg
        if success:
           logger.info(f"Task {task_id} ({product_location_id}): DB operation successful: {op_type_or_error_msg}.")
           processing_summary_logs["status"] = "success"
           processing_summary_logs["final_message"] = f"Operation: {op_type_or_error_msg}."
        else:
           logger.error(f"Task {task_id} ({product_location_id}): DB operation failed. Reason: {op_type_or_error_msg}")
           processing_summary_logs["status"] = f"failed_db_operation"
           processing_summary_logs["final_message"] = f"DB Error: {op_type_or_error_msg}"
           non_retriable_db_errors = [
               "ConstraintViolation", "DataError", "InvalidTextRepresentation",
               "Missing", "dimension mismatch", "Invalid embedding vector type"
           ]
           if any(err_keyword in op_type_or_error_msg for err_keyword in non_retriable_db_errors):
               logger.warning(f"Task {task_id} ({product_location_id}): Non-retriable DB error. Ignoring. Reason: {op_type_or_error_msg}")
               raise Ignore(f"Non-retriable DB error: {op_type_or_error_msg}")
           raise self.retry(exc=Exception(f"DB operation failed: {op_type_or_error_msg}"))

        logger.info(f"Task {task_id} ({product_location_id}) Processing Summary: {processing_summary_logs}")
        return processing_summary_logs

    except Ignore as e_ignore:
     ignore_reason = e_ignore.args[0] if e_ignore.args else "Unknown Ignore reason"
     logger.warning(f"Task {task_id} ({item_identifier_for_log}): Task ignored. Reason: {ignore_reason}")
     if not processing_summary_logs.get("status") or processing_summary_logs.get("status") == "pending":
        processing_summary_logs["status"] = "ignored"
     processing_summary_logs["final_message"] = processing_summary_logs.get("final_message") or f"Task Ignored: {ignore_reason}"
     logger.info(f"Task {task_id} ({item_identifier_for_log}) Processing Summary (Ignored): {processing_summary_logs}")
     return processing_summary_logs
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e_retriable_op:
     logger.error(f"Task {task_id} ({item_identifier_for_log}): Retriable OperationalError: {e_retriable_op}", exc_info=True)
     processing_summary_logs["status"] = "retrying_operational_error"
     processing_summary_logs["final_message"] = f"Retrying due to OperationalError: {str(e_retriable_op)[:100]}"
     try:
        raise self.retry(exc=e_retriable_op)
     except MaxRetriesExceededError:
        logger.critical(f"Task {task_id} ({item_identifier_for_log}): Max retries exceeded for OperationalError. Error: {e_retriable_op}", exc_info=True)
        processing_summary_logs["status"] = "failed_max_retries_operational"
        processing_summary_logs["final_message"] += " Max retries exceeded."
        logger.info(f"Task {task_id} ({item_identifier_for_log}) Processing Summary (Retrying/MaxRetries): {processing_summary_logs}")
        return processing_summary_logs
    except Exception as exc:
        logger.exception(f"Task {task_id} ({item_identifier_for_log}): Unhandled non-operational exception: {exc}")
        processing_summary_logs["status"] = "failed_unhandled_exception"
        processing_summary_logs["final_message"] = f"Unhandled Exception: {str(exc)[:200]}"
        if not self.request.called_directly:
            try:
                countdown = (self.request.retries + 1) * (Config.CELERY_TASK_RETRY_DELAY if hasattr(Config, 'CELERY_TASK_RETRY_DELAY') else 300)
                logger.info(f"Task {task_id} ({item_identifier_for_log}): Retrying task due to unhandled exception. Attempt {self.request.retries + 1}. Countdown: {countdown}s.")
                raise self.retry(exc=exc, countdown=countdown)
            except MaxRetriesExceededError:
                logger.critical(f"Task {task_id} ({item_identifier_for_log}): Max retries exceeded for unhandled exception. Error: {exc}", exc_info=True)
                processing_summary_logs["status"] = "failed_max_retries_unhandled"
                processing_summary_logs["final_message"] += " Max retries exceeded."
            except Exception as retry_exc:
                logger.error(f"Task {task_id} ({item_identifier_for_log}): Error during retry mechanism itself: {retry_exc}", exc_info=True)
                processing_summary_logs["status"] = "failed_retry_mechanism_error"
                processing_summary_logs["final_message"] += " Error in retry mechanism."
        else:
            logger.info(f"Task {task_id} ({item_identifier_for_log}): Task called directly, not retrying unhandled exception.")
        logger.info(f"Task {task_id} ({item_identifier_for_log}) Processing Summary (Failed Exception): {processing_summary_logs}")
        return processing_summary_logs
@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.deactivate_product_task',
    max_retries=Config.CELERY_TASK_MAX_RETRIES_SHORT if hasattr(Config, 'CELERY_TASK_MAX_RETRIES_SHORT') else 3,
    default_retry_delay=Config.CELERY_TASK_RETRY_DELAY_SHORT if hasattr(Config, 'CELERY_TASK_RETRY_DELAY_SHORT') else 60,
    acks_late=True
)
def deactivate_product_task(self, product_id: str):
    task_id = self.request.id
    product_id_lower = product_id.lower()
    logger.info(f"Task {task_id}: Starting deactivation for product_id: {product_id_lower}")
    processing_summary_logs = {
     "task_id": str(task_id),
     "product_id": product_id_lower,
     "status": "pending",
     "db_operation_status": "pending",
     "final_message": ""
    }
    try:
     with db_utils.get_db_session() as session:
        entry = session.query(Product).filter_by(id=product_id_lower).first()
        if entry:
           if entry.stock != 0:
               entry.stock = 0
               # --- FIX: Removed session.commit(). The context manager handles it. ---
               logger.info(f"Task {task_id}: Product_id: {product_id_lower} stock set to 0 for deactivation.")
               processing_summary_logs["db_operation_status"] = "stock_set_to_0"
           else:
               logger.info(f"Task {task_id}: Product_id: {product_id_lower} already has stock 0. No change needed.")
               processing_summary_logs["db_operation_status"] = "already_stock_0"

           processing_summary_logs["status"] = "success"
           processing_summary_logs["final_message"] = "Deactivation processed."
        else:
           logger.warning(f"Task {task_id}: Product_id {product_id_lower} not found for deactivation. No action taken.")
           processing_summary_logs["status"] = "ignored_not_found"
           processing_summary_logs["final_message"] = "Product not found."
           raise Ignore("Product not found for deactivation")

    except Ignore as e_ignore:
     ignore_reason = e_ignore.args[0] if e_ignore.args else "Unknown Ignore reason"
     logger.warning(f"Task {task_id}: Deactivation task for {product_id_lower} ignored. Reason: {ignore_reason}")
     processing_summary_logs["final_message"] = processing_summary_logs.get("final_message") or f"Ignored: {ignore_reason}"
     processing_summary_logs["status"] = "ignored"
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e_op_deactivate:
     logger.error(f"Task {task_id}: Retriable OperationalError during deactivation of {product_id_lower}: {e_op_deactivate}", exc_info=True)
     processing_summary_logs["status"] = "retrying_operational_error"
     processing_summary_logs["final_message"] = f"Retrying deactivation due to OperationalError: {str(e_op_deactivate)[:100]}"
     try:
        raise self.retry(exc=e_op_deactivate)
     except MaxRetriesExceededError:
        logger.critical(f"Task {task_id} (deactivate_product_task): Max retries exceeded for {product_id_lower} after OperationalError. Error: {e_op_deactivate}", exc_info=True)
        processing_summary_logs["status"] = "failed_max_retries_operational"
        processing_summary_logs["final_message"] += " Max retries exceeded."
    except Exception as exc:
     logger.exception(f"Task {task_id}: Unexpected error during deactivation of product_id {product_id_lower}: {exc}")
     processing_summary_logs["status"] = "failed_exception"
     processing_summary_logs["final_message"] = f"Exception: {str(exc)[:200]}"
     if not self.request.called_directly:
        try:
           raise self.retry(exc=exc)
        except MaxRetriesExceededError:
             logger.error(f"Task {task_id} (deactivate_product_task): Max retries exceeded for {product_id_lower} after generic exception. Error: {exc}", exc_info=True)
             processing_summary_logs["status"] = "failed_max_retries_unhandled"
             processing_summary_logs["final_message"] += " Max retries exceeded."
        except Exception as retry_exc_inner:
             logger.error(f"Task {task_id} (deactivate_product_task): Error during retry mechanism for {product_id_lower}: {retry_exc_inner}", exc_info=True)
             processing_summary_logs["status"] = "failed_retry_mechanism_error"
             processing_summary_logs["final_message"] += " Error in retry mechanism."
    logger.info(f"Task {task_id} Deactivation Summary: {processing_summary_logs}")
    return processing_summary_logs
