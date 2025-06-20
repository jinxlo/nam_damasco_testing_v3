# namwoo_app/utils/conversation_location.py

import json
from pathlib import Path
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

# In-memory cache for conversation locations and warehouse data
_conversation_cities: Dict[str, str] = {}
_warehouse_city_map: Optional[Dict[str, List[str]]] = None

# Synonyms for better user input matching. The key is the user input, 
# the value is the canonical city name used in tiendas_data.json.
_city_synonyms: Dict[str, str] = {
    "caracas": "caracas",
    "ccs": "caracas",
    "maracaibo": "maracaibo",
    "mcbo": "maracaibo",
    "valencia": "valencia",
    "barquisimeto": "barquisimeto",
    "bqto": "barquisimeto",
    "maracay": "aragua", 
    "aragua": "aragua",
    "lecheria": "lecherias",
    "lecherías": "lecherias",
    "puerto la cruz": "puerto la cruz",
    "plc": "puerto la cruz",
    "san cristobal": "tachira",
    "tachira": "tachira",
    "maturin": "maturin",
    "puerto ordaz": "puerto ordaz",
    "pzo": "puerto ordaz",
    "la guaira": "terminal la guaira",
    "vargas": "terminal la guaira",
    "cagua": "cagua",
    "los teques": "los teques",
    "san felipe": "san felipe",
    "yaracuy": "san felipe",
    "trujillo": "trujillo",
    "valera": "trujillo"
}

def _load_and_process_tiendas_data() -> Dict[str, List[str]]:
    """
    Loads tienda data from JSON and creates a direct mapping from a canonical 
    city name to a list of exact `whsName` values for the database query.
    This version reads the 'city' and 'whsName' keys directly, no parsing needed.
    """
    global _warehouse_city_map
    if _warehouse_city_map is not None:
        return _warehouse_city_map

    logger.info("Initializing city-to-warehouse map from tiendas_data.json...")
    
    file_path = Path(__file__).parent.parent / 'data' / 'tiendas_data.json'
    
    if not file_path.exists():
        logger.error(f"FATAL: tiendas_data.json not found at {file_path}. Location features will fail.")
        _warehouse_city_map = {}
        return _warehouse_city_map

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tiendas = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading or parsing tiendas_data.json: {e}")
        _warehouse_city_map = {}
        return _warehouse_city_map

    processed_map: Dict[str, List[str]] = {}

    for tienda in tiendas:
        city = tienda.get("city")
        whs_name = tienda.get("whsName")
        
        if not city or not whs_name:
            logger.warning(f"Skipping store entry due to missing 'city' or 'whsName': {tienda}")
            continue
            
        canonical_city = city.lower()

        if canonical_city not in processed_map:
            processed_map[canonical_city] = []
        
        if whs_name not in processed_map[canonical_city]:
            processed_map[canonical_city].append(whs_name)

    _warehouse_city_map = processed_map
    logger.info(f"City-to-warehouse map initialized. Found mappings for {len(_warehouse_city_map)} cities.")
    logger.debug(f"Generated map: {_warehouse_city_map}")
    return _warehouse_city_map


def detect_city_from_text(text: str) -> Optional[str]:
    """
    Detects a known city from a text string using synonyms.
    Returns the canonical city name if found.
    """
    text_lower = text.lower().strip()
    # Direct lookup first
    if text_lower in _city_synonyms:
        return _city_synonyms[text_lower]
        
    # Check for substring matches for robustness
    for synonym, canonical_name in _city_synonyms.items():
        if synonym in text_lower:
            logger.info(f"Detected city '{canonical_name}' from text via synonym '{synonym}'.")
            return canonical_name
            
    return None

def set_conversation_city(conversation_id: str, city: str) -> None:
    """Stores the detected canonical city for a given conversation ID."""
    canonical_city = city.lower() # Assumes city is already canonical from detect_city_from_text
    _conversation_cities[conversation_id] = canonical_city
    logger.info(f"Set city for conversation {conversation_id} to '{canonical_city}'.")


def get_conversation_city(conversation_id: str) -> Optional[str]:
    """Retrieves the stored city for a given conversation ID."""
    return _conversation_cities.get(conversation_id)


def get_city_warehouses(conversation_id: str) -> Optional[List[str]]:
    """
    Gets the list of exact warehouse names (`whsName`) associated with the city
    stored for the given conversation.
    """
    city = get_conversation_city(conversation_id)
    if not city:
        return None
    
    warehouse_map = _load_and_process_tiendas_data()
    warehouses = warehouse_map.get(city)
    
    if warehouses:
        logger.info(f"Found warehouses for city '{city}': {warehouses}")
    else:
        logger.warning(f"No warehouses found for city '{city}' in the map.")
        
    return warehouses

# Initialize the map on module load
_load_and_process_tiendas_data()