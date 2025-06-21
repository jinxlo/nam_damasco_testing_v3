# namwoo_app/utils/product_utils.py
import re
from typing import Optional, Any

def _normalize_string_for_id_part(value: Any) -> Optional[str]:
    if value is None: return None
    s = str(value).strip()
    return s if s else None

def generate_product_location_id(item_code_raw: Any, whs_name_raw: Any) -> Optional[str]:
    """
    Generates the composite product ID consistently.
    Returns None if essential parts (item_code, whs_name) are missing after normalization.
    """
    item_code = _normalize_string_for_id_part(item_code_raw)
    whs_name = _normalize_string_for_id_part(whs_name_raw)

    if not item_code or not whs_name:
        # logger.warning("Cannot generate product_location_id: item_code or whs_name is missing/empty after normalization.")
        return None

    sanitized_whs_name = re.sub(r'[^a-zA-Z0-9_-]', '_', whs_name)
    product_id = f"{item_code}_{sanitized_whs_name}"
    if len(product_id) > 512:
        product_id = product_id[:512]
    return product_id

# --- NEW FUNCTION TO FIX THE ATTRIBUTE ERROR ---
def get_canonical_warehouse_name(warehouse_name: str) -> str:
    """
    Creates a sanitized, lowercase version of a warehouse name suitable for
    use in unique constraints. This is a required helper for the batch processing task.

    Example: "Almacén Principal SAN MARTÍN" -> "almacen_principal_san_martin"
    """
    if not warehouse_name:
        return ""
    # 1. Sanitize: Replace any character that is not a letter, number, underscore,
    #    or hyphen with an underscore.
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(warehouse_name))
    
    # 2. Canonicalize: Convert to lowercase for case-insensitive matching in constraints.
    return sanitized.lower()