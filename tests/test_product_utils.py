import os
import importlib.util
import pytest

UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "utils", "product_utils.py"))
spec = importlib.util.spec_from_file_location("product_utils", UTILS_PATH)
product_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(product_utils)
generate_product_location_id = product_utils.generate_product_location_id


def test_generate_product_location_id_basic():
    assert generate_product_location_id("ABC123", "Main Warehouse") == "ABC123_Main_Warehouse"


def test_generate_product_location_id_none_or_blank():
    assert generate_product_location_id(None, "Main") is None
    assert generate_product_location_id("Item", "") is None
    assert generate_product_location_id("Item", "   ") is None


def test_generate_product_location_id_sanitizes():
    assert generate_product_location_id("A1", "Warehouse/Location-1") == "A1_Warehouse_Location-1"


def test_generate_product_location_id_truncates():
    item_code = "A" * 500
    whs_name = "B" * 20
    result = generate_product_location_id(item_code, whs_name)
    expected = (f"{item_code}_{whs_name}")[:512]
    assert result == expected
