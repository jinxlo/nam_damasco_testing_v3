# /home/ec2-user/namwoo_app/namwoo_app/celery_tasks.py

import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ValidationError, validator
from decimal import Decimal, InvalidOperation
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from celery.exceptions import Ignore, MaxRetriesExceededError, OperationalError as CeleryOperationalError

from .celery_app import celery_app, FlaskTask
from .services import product_service, openai_service, llm_processing_service
from .utils import db_utils, product_utils
from .models.product import Product
from .config import Config

logger = logging.getLogger(__name__)

# --- Pydantic Model for Validating Incoming Snake_Case Product Data ---
# This model remains unchanged as it's good practice.
class DamascoProductDataSnake(BaseModel):
    item_code: str
    item_name: str
    description: Optional[str] = None
    specifitacion: Optional[str] = None
    stock: int
    price: Optional[Decimal] = None
    price_bolivar: Optional[Decimal] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand: Optional[str] = None
    line: Optional[str] = None
    item_group_name: Optional[str] = None
    warehouse_name: str
    branch_name: Optional[str] = None
    # This field is no longer needed here as we will pass the whole dict
    # original_input_data: Optional[Dict[str, Any]] = None

    @validator('price', 'price_bolivar', pre=True, allow_reuse=True)
    def validate_prices_to_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None: return None
        if isinstance(v, (int, float, str)):
            try: return Decimal(str(v))
            except InvalidOperation: return None
        if isinstance(v, Decimal): return v
        return None

    class Config:
        extra = 'allow'
        validate_assignment = True

# --- NEW, EFFICIENT, AND ROBUST BATCH PROCESSING TASK ---
@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.process_products_batch_task',
    max_retries=Config.CELERY_TASK_MAX_RETRIES if hasattr(Config, 'CELERY_TASK_MAX_RETRIES') else 3,
    default_retry_delay=Config.CELERY_TASK_RETRY_DELAY if hasattr(Config, 'CELERY_TASK_RETRY_DELAY') else 300,
    acks_late=True
)
def process_products_batch_task(self, products_batch_snake_case: List[Dict[str, Any]]):
    """
    Processes a batch of products: validates, determines necessity of new summaries/embeddings,
    generates them if needed, and saves the entire batch to the database using a single,
    atomic, and concurrent-safe upsert operation.
    """
    task_id = self.request.id
    batch_size = len(products_batch_snake_case)
    logger.info(f"Task {task_id}: Starting batch processing for {batch_size} products.")

    if not products_batch_snake_case:
        logger.info(f"Task {task_id}: Received an empty batch. Nothing to do.")
        return {"status": "success_empty_batch", "processed_count": 0}

    # --- STEP 1: PRE-PROCESSING (Validation and ID Generation in Memory) ---
    validated_items = []
    product_ids_to_check = []
    for raw_product_data in products_batch_snake_case:
        try:
            validated_product = DamascoProductDataSnake(**raw_product_data)
            product_id = product_utils.generate_product_location_id(
                validated_product.item_code,
                validated_product.warehouse_name
            )
            if product_id:
                validated_items.append((product_id, validated_product, raw_product_data))
                product_ids_to_check.append(product_id)
            else:
                logger.error(f"Task {task_id}: Failed to generate ID for item {raw_product_data.get('item_code')}. Skipping.")
        except ValidationError as e:
            logger.error(f"Task {task_id}: Pydantic validation failed for item {raw_product_data.get('item_code')}. Skipping. Error: {e.errors()}")
            
    if not validated_items:
        logger.warning(f"Task {task_id}: No items survived validation. Exiting.")
        return {"status": "failed_all_invalid", "processed_count": 0}

    # --- STEP 2: EFFICIENTLY FETCH EXISTING DATA FOR THE ENTIRE BATCH ---
    existing_products_map = {}
    try:
        with db_utils.get_db_session() as session:
            existing_db_entries = session.query(
                Product.id,
                Product.description,
                Product.llm_summarized_description,
                Product.searchable_text_content,
                Product.embedding
            ).filter(Product.id.in_(product_ids_to_check)).all()

            for entry in existing_db_entries:
                existing_products_map[entry.id] = {
                    "description": entry.description,
                    "llm_summarized_description": entry.llm_summarized_description,
                    "searchable_text_content": entry.searchable_text_content,
                    "embedding": entry.embedding
                }
        logger.info(f"Task {task_id}: Fetched existing data for {len(existing_products_map)} of {len(validated_items)} products.")
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e:
        logger.error(f"Task {task_id}: Retriable DB/Broker error during batch read: {e}", exc_info=True)
        raise self.retry(exc=e)

    # --- STEP 3: PROCESS EACH ITEM (Summaries, Embeddings) ---
    db_ready_products = []
    for product_id, validated_product, original_data in validated_items:
        try:
            # Determine if new summary or embedding is needed by comparing with pre-fetched data
            existing_details = existing_products_map.get(product_id)
            
            # Summarization Logic
            llm_summary_to_use = existing_details.get("llm_summarized_description") if existing_details else None
            needs_new_summary = (
                (not existing_details and validated_product.description) or
                (existing_details and validated_product.description != existing_details.get("description")) or
                (existing_details and not existing_details.get("llm_summarized_description") and validated_product.description)
            )

            if needs_new_summary:
                new_summary = llm_processing_service.generate_llm_product_summary(
                    html_description=validated_product.description,
                    item_name=validated_product.item_name
                )
                if new_summary:
                    llm_summary_to_use = new_summary
            
            # Embedding Logic
            text_to_embed = Product.prepare_text_for_embedding(
                damasco_product_data=validated_product.model_dump(),
                llm_generated_summary=llm_summary_to_use,
                raw_html_description_for_fallback=validated_product.description
            )
            
            if not text_to_embed:
                logger.warning(f"Task {task_id}: No text content for embedding for product {product_id}. Skipping item.")
                continue

            embedding_to_use = None
            if existing_details and text_to_embed == existing_details.get("searchable_text_content") and existing_details.get("embedding") is not None:
                embedding_to_use = existing_details["embedding"]
            else:
                embedding_to_use = openai_service.generate_product_embedding(text_to_embed)

            if embedding_to_use is None:
                logger.error(f"Task {task_id}: Failed to get embedding for {product_id}. Skipping item.")
                continue
            
            # Assemble the final dictionary for the database upsert
            db_ready_products.append({
                "id": product_id,
                "item_code": validated_product.item_code,
                "item_name": validated_product.item_name,
                "description": validated_product.description,
                "llm_summarized_description": llm_summary_to_use,
                "specifitacion": validated_product.specifitacion,
                "category": validated_product.category,
                "sub_category": validated_product.sub_category,
                "brand": validated_product.brand,
                "line": validated_product.line,
                "item_group_name": validated_product.item_group_name,
                "warehouse_name": validated_product.warehouse_name,
                "warehouse_name_canonical": product_utils.get_canonical_warehouse_name(validated_product.warehouse_name),
                "branch_name": validated_product.branch_name,
                "price": validated_product.price,
                "price_bolivar": validated_product.price_bolivar,
                "stock": validated_product.stock,
                "searchable_text_content": text_to_embed,
                "embedding": embedding_to_use,
                "source_data_json": original_data,
            })
        except Exception as item_proc_exc:
            # Catch errors during LLM/embedding calls for a single item
            logger.error(f"Task {task_id}: Failed to process item {product_id} due to: {item_proc_exc}. Skipping item.", exc_info=True)
            continue # Continue to the next item in the batch

    # --- STEP 4: PERFORM THE ATOMIC BATCH UPSERT ---
    if not db_ready_products:
        logger.warning(f"Task {task_id}: No products were ready for DB write after processing. Exiting.")
        return {"status": "success_nothing_to_write", "processed_count": 0}

    try:
        with db_utils.get_db_session() as session:
            product_service.upsert_products_batch(session, db_ready_products)
            session.commit()
            logger.info(f"Task {task_id}: Successfully committed batch upsert for {len(db_ready_products)} products.")
            return {"status": "success", "processed_count": len(db_ready_products)}
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e_db_op:
        logger.error(f"Task {task_id}: Retriable DB/Broker error during final batch write: {e_db_op}", exc_info=True)
        raise self.retry(exc=e_db_op)
    except Exception as e_final:
        logger.critical(f"Task {task_id}: Non-retriable error during final batch write: {e_final}", exc_info=True)
        # Depending on severity, you might want to raise Ignore or let it fail
        # For now, we let it retry as a generic exception.
        raise self.retry(exc=e_final)


# =================================================================================================
# == DEPRECATED TASK - DO NOT USE =================================================================
# =================================================================================================
# The task below, `process_product_item_task`, processes items one by one. This approach
# is inefficient (many DB round-trips) and was the source of the `UniqueViolation` errors
# because its underlying service function used a non-atomic "check-then-act" pattern.
#
# It has been replaced by `process_products_batch_task`, which uses an efficient, atomic
# `INSERT ... ON CONFLICT` operation for the entire batch, eliminating race conditions
# and drastically improving performance. This old task should be removed after confirming
# the new batch task works as expected.
# =================================================================================================
@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.process_product_item_task_DEPRECATED', # Renamed to avoid accidental use
    # ... (rest of the old task decorator) ...
)
def process_product_item_task(self, product_data_dict_snake: Dict[str, Any]):
    logger.warning("DEPRECATED TASK 'process_product_item_task' was called. Please switch to 'process_products_batch_task'.")
    # You can choose to either raise an error or just ignore the call.
    # For a smoother transition, you might initially delegate to the new task,
    # but the goal is to stop calling this altogether.
    # For now, we will simply ignore the call.
    raise Ignore("Called a deprecated single-item processing task.")


# The `deactivate_product_task` is fine as it is, since it operates on a single, specific product ID for a different purpose.
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
    # ... The rest of this function remains unchanged ...
    try:
        with db_utils.get_db_session() as session:
            entry = session.query(Product).filter_by(id=product_id_lower).first()
            if entry:
                if entry.stock != 0:
                    entry.stock = 0
                    logger.info(f"Task {task_id}: Product_id: {product_id_lower} stock set to 0 for deactivation. Committing.")
                    session.commit()
                else:
                    logger.info(f"Task {task_id}: Product_id: {product_id_lower} already has stock 0. No change needed.")
            else:
                logger.warning(f"Task {task_id}: Product_id {product_id_lower} not found for deactivation. No action taken.")
                raise Ignore("Product not found for deactivation")
    except Ignore:
        logger.warning(f"Task {task_id}: Deactivation task for {product_id_lower} ignored.")
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e_op_deactivate:
        logger.error(f"Task {task_id}: Retriable OperationalError during deactivation of {product_id_lower}: {e_op_deactivate}", exc_info=True)
        raise self.retry(exc=e_op_deactivate)
    except Exception as exc:
        logger.exception(f"Task {task_id}: Unexpected error during deactivation of product_id {product_id_lower}: {exc}")
        raise self.retry(exc=exc)