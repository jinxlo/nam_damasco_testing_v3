import json
import logging
import os
import importlib.util

UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "utils", "logging_utils.py"))
spec = importlib.util.spec_from_file_location("logging_utils", UTILS_PATH)
logging_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_utils)
JsonFormatter = logging_utils.JsonFormatter


def test_json_formatter_basic_fields():
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="/tmp/test.py",
        lineno=42,
        msg="hello",
        args=(),
        exc_info=None,
        func="test_func",
    )
    formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    output = formatter.format(record)
    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["message"] == "hello"
    assert data["funcName"] == "test_func"
    assert data["line"] == 42
    assert "timestamp" in data


def test_setup_json_file_logger_writes(tmp_path):
    log_file = tmp_path / "app.json"
    handler = logging_utils.setup_json_file_logger(str(log_file))
    logger = logging.getLogger("test_setup_json_file_logger")
    logger.info("json hello")
    handler.flush()
    logging.getLogger().removeHandler(handler)

    with open(log_file, "r", encoding="utf8") as f:
        line = json.loads(f.readline())

    assert line["message"] == "json hello"
    assert line["name"] == "test_setup_json_file_logger"

