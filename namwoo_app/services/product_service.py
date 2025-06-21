# NAMWOO/services/product_service.py

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation as InvalidDecimalOperation
from datetime import datetime

import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
# NEW IMPORT for the atomic upsert operation
from sqlalchemy.dialects.postgresql import insert

from ..models.product import Product
from ..utils import db_utils, embedding_utils, text_utils
from ..config import Config

logger = logging.getLogger(__name__)

# --- Semantic Helpers ---
_KNOWN_COLORS = {
    'negro', 'blanco', 'azul', 'rojo', 'verde', 'gris', 'plata', 'dorado',
    'rosado', 'violeta', 'morado', 'amarillo', 'naranja', 'marrón', 'beige',
    'celeste', 'turquesa', 'lila', 'crema', 'grafito', 'titanio', 'cobre',
    'negra', 'blanca', 'claro', 'oscuro', 'marino'
}
_SKU_PAT = re.compile(r'\b(SM-[A-Z0-9]+[A-Z]*|[A-Z0-9]{8,})\b')

def _extract_base_name_and_color(item_name: str) -> Tuple[str, Optional[str]]:
    if not item_name:
        return "", None
    name_without_sku = _SKU_PAT.sub('', item_name).strip()
    words = name_without_sku.split()
    base_parts, color_parts = [], []
    found_color = False
    for w in reversed(words):
        if not found_color and w.lower() in _KNOWN_COLORS:
            color_parts.insert(0, w)
        else:
            found_color = True
            base_parts.insert(0, w)
    base = " ".join(base_parts).strip()
    color = " ".join(color_parts).strip()
    return (base or name_without_sku, color.capitalize() if color else None)

def get_available_brands_by_category(category: str = 'CELULAR') -> Optional[List[str]]:
    if not category:
        return []
    logger.info(f"Fetching distinct, in-stock brands for category: {category}")
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_available_brands_by_category.")
            return None
        try:
            qry = (session.query(Product.brand)
                         .filter(Product.category == category.upper(),
                                 Product.stock > 0)
                         .distinct().order_by(Product.brand))
            brands = [b[0] for b in qry.all() if b[0] is not None]
            logger.info(f"Found {len(brands)} brands: {brands}")
            return brands
        except Exception:
            logger.exception("Error fetching distinct brands")
            return None

def search_local_products(
    query_text: str,
    limit: int = 300,
    filter_stock: bool = True,
    min_score: float = 0.10,
    warehouse_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not query_text or not isinstance(query_text, str):
        logger.warning("Empty or invalid search query.")
        return {}
    logger.info("Starting vector search for '%s…'", query_text[:50])
    model = getattr(Config, 'OPENAI_EMBEDDING_MODEL', "text-embedding-3-small")
    q_emb = embedding_utils.get_embedding(query_text, model=model)
    if not q_emb:
        logger.error("Failed to generate query embedding.")
        return None

    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for search.")
            return None
        try:
            q = session.query(Product, (1 - Product.embedding.cosine_distance(q_emb)).label("sim"))
            if filter_stock:
                q = q.filter(Product.stock > 0)
            if warehouse_names:
                q = q.filter(Product.warehouse_name.in_(warehouse_names))
            q = (q.filter(Product.item_group_name == "DAMASCO TECNO")
                  .filter((1 - Product.embedding.cosine_distance(q_emb)) >= min_score)
                  .order_by(Product.embedding.cosine_distance(q_emb))
                  .limit(limit))
            rows = q.all()

            grouped: Dict[str, Dict[str, Any]] = {}
            for prod, sim in rows:
                base, color = _extract_base_name_and_color(prod.item_name)
                if base not in grouped:
                    desc = prod.llm_summarized_description or text_utils.strip_html_to_text(prod.description or "")
                    specs = (prod.specifitacion or "").strip()
                    grouped[base] = {
                        "base_name": base,
                        "brand": prod.brand,
                        "category": prod.category,
                        "sub_category": prod.sub_category,
                        "marketing_description": desc.strip(),
                        "technical_specs": specs,
                        "variants": [],
                        "locations": []
                    }
                variant = {
                    "color": color or "N/A",
                    "price": float(prod.price) if prod.price is not None else None,
                    "price_bolivar": float(prod.price_bolivar) if prod.price_bolivar is not None else None,
                    "full_item_name": prod.item_name,
                    "item_code": prod.item_code
                }
                if variant not in grouped[base]["variants"]:
                    grouped[base]["variants"].append(variant)
                grouped[base]["locations"].append({
                    "warehouse_name": prod.warehouse_name,
                    "branch_name": prod.branch_name,
                    "stock": prod.stock,
                    "color_specific_item_name": prod.item_name
                })

            # merge stocks by branch
            for p in grouped.values():
                merged = {}
                for loc in p["locations"]:
                    br = loc["branch_name"]
                    merged.setdefault(br, {"branch_name": br, "total_stock": 0})
                    merged[br]["total_stock"] += loc["stock"]
                p["locations"] = list(merged.values())

            return {"status": "success", "products_grouped": list(grouped.values())}

        except SQLAlchemyError as db_err:
            logger.exception("DB error during search.")
            return None
        except Exception:
            logger.exception("Unexpected error during search.")
            return None

# --- NEW ROBUST BATCH UPSERT FUNCTION ---
def upsert_products_batch(db_session: Session, products_data: List[Dict[str, Any]]):
    """
    Atomically inserts or updates a batch of products using a single, efficient
    'INSERT ... ON CONFLICT' statement. This is the definitive, concurrent-safe
    method for writing product data, replacing the old add_or_update_product_in_db.

    Args:
        db_session (Session): The SQLAlchemy database session.
        products_data (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary represents a product and its keys match the Product model columns.
    """
    if not products_data:
        logger.info("upsert_products_batch called with an empty list. No DB operation performed.")
        return

    # Create the 'INSERT' statement using the PostgreSQL dialect.
    stmt = insert(Product).values(products_data)

    # Define the 'ON CONFLICT DO UPDATE' clause.
    # If a row with a conflicting 'id' already exists, this tells the database
    # to update the existing row with the new values.
    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=['id'],  # The column with the primary key constraint.
        set_={
            # Update every column EXCEPT the primary key ('id') and 'created_at'.
            # 'stmt.excluded' refers to the new values that were attempted to be inserted.
            'item_code': stmt.excluded.item_code,
            'item_name': stmt.excluded.item_name,
            'description': stmt.excluded.description,
            'llm_summarized_description': stmt.excluded.llm_summarized_description,
            'specifitacion': stmt.excluded.specifitacion,
            'category': stmt.excluded.category,
            'sub_category': stmt.excluded.sub_category,
            'brand': stmt.excluded.brand,
            'line': stmt.excluded.line,
            'item_group_name': stmt.excluded.item_group_name,
            'warehouse_name': stmt.excluded.warehouse_name,
            'warehouse_name_canonical': stmt.excluded.warehouse_name_canonical,
            'branch_name': stmt.excluded.branch_name,
            'price': stmt.excluded.price,
            'price_bolivar': stmt.excluded.price_bolivar,
            'stock': stmt.excluded.stock,
            'searchable_text_content': stmt.excluded.searchable_text_content,
            'embedding': stmt.excluded.embedding,
            'source_data_json': stmt.excluded.source_data_json,
            # NOTE: We DO NOT update 'updated_at' here. The database trigger
            # 'set_products_timestamp' handles this automatically and more reliably.
        }
    )

    # Execute the entire atomic operation.
    db_session.execute(on_conflict_stmt)
    logger.info(f"Executed batch upsert for {len(products_data)} products.")


def _normalize_string(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None

def get_product_by_id_from_db(db_session: Session, product_id: str) -> Optional[Product]:
    if not product_id:
        return None
    return db_session.query(Product).filter(Product.id == product_id).first()


# ==============================================================================
# == DEPRECATED FUNCTION - DO NOT USE ==========================================
# ==============================================================================
# The function below, `add_or_update_product_in_db`, is the source of the
# `UniqueViolation` errors. It uses a "check-then-act" pattern:
#
# 1. `existing = session.query(...).first()` (THE CHECK)
# 2. `if existing: ... else: session.add(...)` (THE ACT)
#
# This pattern fails under concurrency. Two Celery workers can both check for the
# same new product, both find it doesn't exist, and then both try to INSERT it,
# causing a race condition and an error.
#
# It has been replaced by the `upsert_products_batch` function above, which
# uses an atomic `INSERT ... ON CONFLICT` operation, delegating the logic to
# the database and completely eliminating the race condition.
# ==============================================================================

# def add_or_update_product_in_db(
#     session: Session,
#     product_location_id: str,
#     damasco_product_data_camel: Dict[str, Any],
#     embedding_vector: Optional[Any],
#     text_used_for_embedding: Optional[str],
#     llm_summarized_description_to_store: Optional[str]
# ) -> Tuple[bool, str]:
#     if not product_location_id:
#         logger.error("product_location_id was not provided to add_or_update_product_in_db.")
#         return False, "Missing product_location_id."

#     item_code = _normalize_string(damasco_product_data_camel.get("itemCode"))
#     whs_name = _normalize_string(damasco_product_data_camel.get("whsName"))

#     if not item_code or not whs_name:
#         return False, "Missing itemCode or whsName in product data."

#     sanitized_whs = re.sub(r'[^a-zA-Z0-9_-]', '_', whs_name)
#     canonical_whs = sanitized_whs.lower()

#     if not isinstance(damasco_product_data_camel, dict):
#         return False, "Invalid Damasco product data."

#     # prepare embedding
#     emb_list: Optional[List[float]] = None
#     if embedding_vector is not None:
#         if isinstance(embedding_vector, np.ndarray):
#             emb_list = embedding_vector.tolist()
#         elif isinstance(embedding_vector, list):
#             emb_list = embedding_vector
#         else:
#             return False, f"Invalid embedding type for {product_location_id}."
#         exp_dim = getattr(Config, 'EMBEDDING_DIMENSION', None)
#         if exp_dim and len(emb_list) != exp_dim:
#             return False, f"Dim mismatch (expected {exp_dim}, got {len(emb_list)})."

#     logp = f"ProductService DB Upsert (ID='{product_location_id}'):"
#     norm_html = _normalize_string(damasco_product_data_camel.get("description"))
#     norm_llm = _normalize_string(llm_summarized_description_to_store)
#     norm_text = _normalize_string(text_used_for_embedding)

#     # normalize price/stock
#     def _parse_decimal(val):
#         try:
#             return Decimal(str(val)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
#         except:
#             return None

#     price_db = _parse_decimal(damasco_product_data_camel.get("price"))
#     price_bs_db = _parse_decimal(damasco_product_data_camel.get("priceBolivar"))
#     try:
#         stock_db = max(int(damasco_product_data_camel.get("stock") or 0), 0)
#     except:
#         stock_db = 0

#     new_map = {
#         "item_code": item_code,
#         "item_name": _normalize_string(damasco_product_data_camel.get("itemName")),
#         "description": norm_html,
#         "llm_summarized_description": norm_llm,
#         "specifitacion": _normalize_string(damasco_product_data_camel.get("specifitacion")),
#         "category": _normalize_string(damasco_product_data_camel.get("category")),
#         "sub_category": _normalize_string(damasco_product_data_camel.get("subCategory")),
#         "brand": _normalize_string(damasco_product_data_camel.get("brand")),
#         "line": _normalize_string(damasco_product_data_camel.get("line")),
#         "item_group_name": _normalize_string(damasco_product_data_camel.get("itemGroupName")),
#         "warehouse_name": whs_name,
#         "warehouse_name_canonical": canonical_whs,
#         "branch_name": _normalize_string(damasco_product_data_camel.get("branchName")),
#         "price": price_db,
#         "price_bolivar": price_bs_db,
#         "stock": stock_db,
#         "searchable_text_content": norm_text,
#         "embedding": emb_list,
#         "source_data_json": damasco_product_data_camel
#     }

#     try:
#         existing = session.query(Product).filter_by(id=product_location_id).first()
#         if existing:
#             changed = False
#             for field, value in new_map.items():
#                 if field == "embedding":
#                     existing_val = list(existing.embedding) if existing.embedding is not None else None
#                     if existing_val != value:
#                         setattr(existing, field, value)
#                         changed = True
#                 else:
#                     if getattr(existing, field) != value:
#                         setattr(existing, field, value)
#                         changed = True
#             if not changed:
#                 return True, "unchanged"
#             existing.updated_at = datetime.utcnow()
#             return True, "updated"
#         else:
#             new_product = Product(id=product_location_id, **new_map)
#             session.add(new_product)
#             return True, "inserted"

#     except SQLAlchemyError as db_exc:
#         # The rollback will be handled by the context manager.
#         logger.exception(f"{logp} upsert failed: {db_exc}")
#         return False, f"db_error: {str(db_exc)[:200]}"
#     except Exception as exc:
#         # The rollback will be handled by the context manager.
#         logger.exception(f"{logp} unexpected error: {exc}")
#         return False, f"unexpected_error: {str(exc)[:200]}"


def get_live_product_details_by_sku(item_code_query: str) -> Optional[List[Dict[str, Any]]]:
    if not (code := _normalize_string(item_code_query)):
        logger.warning(f"Invalid SKU query: {item_code_query}")
        return []
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_live_product_details_by_sku.")
            return None
        try:
            rows = session.query(Product).filter_by(item_code=code).all()
            return [r.to_dict() for r in rows] if rows else []
        except SQLAlchemyError:
            logger.exception("DB error fetching by sku")
            return None
        except Exception:
            logger.exception("Unexpected error fetching by sku")
            return None

def get_live_product_details_by_id(composite_id: str) -> Optional[Dict[str, Any]]:
    if not composite_id:
        logger.error("Missing composite_id.")
        return None
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_live_product_details_by_id.")
            return None
        try:
            prod = session.query(Product).filter_by(id=composite_id).first()
            return prod.to_dict() if prod else None
        except SQLAlchemyError:
            logger.exception("DB error fetching by id")
            return None
        except Exception:
            logger.exception("Unexpected error fetching by id")
            return None