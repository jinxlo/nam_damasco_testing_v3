# NAMWOO/services/damasco_service.py
import logging
from decimal import Decimal, InvalidOperation # <<< ADDED for precise price handling

# Logger for this service
logger = logging.getLogger('sync')  # Matches sync_service.py logger (or a dedicated one like 'damasco_service')

def process_damasco_data(raw_data_list: list) -> list:
    """
    Cleans and structures raw Damasco product data into a list of dictionaries
    with snake_case keys, ready for further processing (e.g., Celery tasks, DB sync).

    Args:
        raw_data_list: The raw JSON data received from the fetcher 
                       (list of product dictionaries, assumed to have camelCase keys
                        including 'description' with HTML content and 'priceBolivar').

    Returns:
        A list of cleaned product dictionaries with snake_case keys.
    """
    if not isinstance(raw_data_list, list):
        logger.error("Invalid data format: Expected a list of product dictionaries.")
        return []

    cleaned_products = []

    for item_index, item in enumerate(raw_data_list): # Added index for better logging
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dictionary item in raw_data_list at index {item_index}: {item}")
            continue
        
        item_code_log = item.get('itemCode', 'N/A') # For logging before full processing

        try:
            # --- Price Handling with Decimal for precision ---
            price_usd_raw = item.get('price')
            price_usd_decimal = None
            if price_usd_raw is not None:
                try:
                    price_usd_decimal = Decimal(str(price_usd_raw)) # Convert via string for precision
                except InvalidOperation:
                    logger.warning(f"Invalid 'price' value '{price_usd_raw}' for itemCode {item_code_log}. Setting to None.")
            
            # <<< ADD HANDLING FOR priceBolivar >>>
            price_bolivar_raw = item.get('priceBolivar')
            price_bolivar_decimal = None
            if price_bolivar_raw is not None:
                try:
                    price_bolivar_decimal = Decimal(str(price_bolivar_raw)) # Convert via string
                except InvalidOperation:
                    logger.warning(f"Invalid 'priceBolivar' value '{price_bolivar_raw}' for itemCode {item_code_log}. Setting to None.")
            # <<< END priceBolivar HANDLING >>>

            # Extract and normalize other fields, converting to snake_case
            product = {
                'item_code': str(item.get('itemCode', '')).strip(),
                'item_name': str(item.get('itemName', '')).strip(),
                'description': item.get('description'), # Keep as is (raw HTML)
                'specifitacion': str(item.get('specifitacion', '')).strip(), # <<< NEW FIELD ADDED
                'stock': int(item.get('stock', 0)), # Default to 0 if missing/invalid
                'price': price_usd_decimal, # Store as Decimal (or float if you prefer: float(price_usd_raw or 0.0))
                'price_bolivar': price_bolivar_decimal, # <<< ADDED new field, store as Decimal
                'category': str(item.get('category', '')).strip(),
                'sub_category': str(item.get('subCategory', '')).strip(),
                'brand': str(item.get('brand', '')).strip(),
                'line': str(item.get('line', '')).strip(),
                'item_group_name': str(item.get('itemGroupName', '')).strip(),
                'warehouse_name': str(item.get('whsName', '')).strip(), # Crucial for ID
                'branch_name': str(item.get('branchName', '')).strip()
            }

            # Ensure essential keys are not empty after stripping
            if not product['item_code']:
                logger.warning(f"Skipping item at index {item_index} with missing or empty itemCode: {item}")
                continue
            if not product['warehouse_name']:
                logger.warning(f"Skipping item '{product.get('item_code')}' at index {item_index} due to missing or empty warehouse_name (whsName): {item}")
                continue
            
            # Optional: Validate stock is non-negative
            if product['stock'] < 0:
                logger.warning(f"Stock for item '{product.get('item_code')}' is negative ({product['stock']}). Setting to 0.")
                product['stock'] = 0

            cleaned_products.append(product)

        except (ValueError, TypeError) as e: # Catch if int() or other type conversions fail
            logger.error(f"Error processing item (likely type conversion issue for stock): {item}. Error: {e}")
        except Exception as e: # Catch any other unexpected error for this item
            logger.error(f"Failed to process item: {item}. Error: {e}", exc_info=True)

    logger.info(f"Damasco Service: Processed {len(cleaned_products)} valid products out of {len(raw_data_list)} received.")
    return cleaned_products