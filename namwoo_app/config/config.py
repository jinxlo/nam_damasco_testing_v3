# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(basedir, '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"DEBUG [config.py]: Loaded .env from: {dotenv_path}") # Keep your original print
else:
    print(f"WARNING: .env file not found at {dotenv_path}") # Keep your original print

class Config:
    # --- Flask App ---
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-insecure-secret-key')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = FLASK_ENV == 'development'

    # --- Logging ---
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_DIR = os.path.join(basedir, 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, 'app.log')
    SYNC_LOG_FILE = os.path.join(LOG_DIR, 'sync.log')
    LOG_JSON_FILE = os.path.join(LOG_DIR, 'app.json')

    # --- LLM Configuration ---
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai').lower()
    if LLM_PROVIDER not in ['openai', 'google']:
        print(f"WARNING [Config]: Invalid LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to 'openai'.") # Keep your original print
        LLM_PROVIDER = 'openai'

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', 1024))
    EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION', 1536))

    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    GOOGLE_GEMINI_MODEL = os.environ.get('GOOGLE_GEMINI_MODEL', 'gemini-1.5-flash-latest') # Corrected model name based on .env
    GOOGLE_MAX_TOKENS = int(os.environ.get('GOOGLE_MAX_TOKENS', 2048))

    # --- PostgreSQL Database ---
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = DEBUG

    # ======================================================
    # ============ Celery 5+ Configuration (NEW) ===========
    # ======================================================
    broker_url = os.environ.get('broker_url', 'redis://localhost:6379/0') # Match .env key
    result_backend = os.environ.get('result_backend', 'redis://localhost:6379/0') # Match .env key
    task_serializer = os.environ.get('task_serializer', 'json') # Match .env key
    result_serializer = os.environ.get('result_serializer', 'json') # Match .env key
    accept_content = [os.environ.get('accept_content', 'json')] # Match .env key
    timezone = os.environ.get('timezone', 'America/Caracas') # Match .env key
    enable_utc = os.environ.get('enable_utc', 'true').lower() == 'true' # Match .env key

    # --- Scheduler ---
    SYNC_INTERVAL_MINUTES = int(os.environ.get('SYNC_INTERVAL_MINUTES', 60))

    # --- Support Board Configuration ---
    SUPPORT_BOARD_API_URL = os.environ.get('SUPPORT_BOARD_API_URL')
    SUPPORT_BOARD_API_TOKEN = os.environ.get('SUPPORT_BOARD_API_TOKEN')
    SUPPORT_BOARD_WEBHOOK_SECRET = os.environ.get('SUPPORT_BOARD_WEBHOOK_SECRET')

    # --- MODIFICATION START: Bot and Agent User IDs in Support Board ---
    # User ID of THIS Namwoo DM Bot in Support Board
    SUPPORT_BOARD_DM_BOT_USER_ID = os.environ.get('SUPPORT_BOARD_DM_BOT_USER_ID') # Was SUPPORT_BOARD_BOT_USER_ID

    # User ID in Support Board that Instagram Page DMs (and thus Comment Bot DMs) are attributed to
    COMMENT_BOT_PROXY_USER_ID = os.environ.get('COMMENT_BOT_PROXY_USER_ID')

    # (Optional) A unique string embedded by the Comment Bot in its initial DMs.
    # If empty in .env, this will be None or an empty string, which is fine.
    COMMENT_BOT_INITIATION_TAG = os.environ.get('COMMENT_BOT_INITIATION_TAG')

    # Comma-separated list of User IDs for ACTUAL HUMAN agents in Support Board.
    _agent_ids_str = os.environ.get('SUPPORT_BOARD_AGENT_IDS', '')
    # Convert to a set of strings for efficient lookup and to match string IDs from webhook
    SUPPORT_BOARD_AGENT_IDS = {id.strip() for id in _agent_ids_str.split(',') if id.strip()} if _agent_ids_str else set()
    
    HUMAN_TAKEOVER_PAUSE_MINUTES = int(os.environ.get('HUMAN_TAKEOVER_PAUSE_MINUTES', 43200)) # Default to 30 days
    # --- Department IDs (New) ---
    SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID = int(os.getenv("SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID", 1))
    SUPPORT_BOARD_SALES_DEPARTMENT_ID = int(os.getenv("SUPPORT_BOARD_SALES_DEPARTMENT_ID", 2))
    # --- MODIFICATION END ---

    # Note: SUPPORT_BOARD_SENDER_USER_ID was present in your original config.py but not in .env.example.
    # If it's used elsewhere, ensure it's correctly defined or removed if redundant.
    # For the current DM bot logic, SUPPORT_BOARD_DM_BOT_USER_ID is what's important for sending.

    # --- WhatsApp Cloud API ---
    WHATSAPP_CLOUD_API_TOKEN = os.environ.get('WHATSAPP_CLOUD_API_TOKEN')
    WHATSAPP_PHONE_NUMBER_ID = os.environ.get('WHATSAPP_PHONE_NUMBER_ID')
    WHATSAPP_DEFAULT_COUNTRY_CODE = os.environ.get('WHATSAPP_DEFAULT_COUNTRY_CODE', '')
    WHATSAPP_API_VERSION = os.environ.get('WHATSAPP_API_VERSION', 'v19.0')

    # --- Application Specific ---
    MAX_HISTORY_MESSAGES = int(os.environ.get('MAX_HISTORY_MESSAGES', 16))
    PRODUCT_SEARCH_LIMIT = max(5, int(os.environ.get('PRODUCT_SEARCH_LIMIT', 10)))

    # --- Damasco Specific ---
    DAMASCO_RECEIVER_API_URL = os.environ.get('DAMASCO_RECEIVER_API_URL')
    DAMASCO_API_SECRET = os.environ.get('DAMASCO_API_SECRET')

    if not DAMASCO_API_SECRET:
        print("WARNING [Config]: DAMASCO_API_SECRET is not set. Receiver endpoint will reject requests.") # Keep your original print

    # --- System Prompt for AI Assistant (from file) ---
    SYSTEM_PROMPT_FILE = os.path.join(basedir, 'data', 'system_prompt.txt')
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            SYSTEM_PROMPT = f.read().strip()
        print(f"INFO [Config]: Loaded system prompt from {SYSTEM_PROMPT_FILE}") # Keep your original print
    except FileNotFoundError:
        print(f"ERROR [Config]: system_prompt.txt not found at {SYSTEM_PROMPT_FILE}. Using fallback.") # Keep your original print
        SYSTEM_PROMPT = "Default fallback system prompt content here." # Or load from os.environ.get('SYSTEM_PROMPT')


# --- Config Sanity Check ---
if __name__ != "__main__": # This runs when the module is imported
    print(f"--- Config Initialized ---") # Keep your original print
    print(f"ENV: {Config.FLASK_ENV}, DEBUG={Config.DEBUG}, LLM Provider: {Config.LLM_PROVIDER}")
    print(f"DB URI: {'SET' if Config.SQLALCHEMY_DATABASE_URI else 'MISSING'}") # Keep your original print for this project's DB
    print(f"Celery broker_url: {Config.broker_url}") 
    print(f"Celery result_backend: {Config.result_backend}") 
    print(f"Support Board URL: {Config.SUPPORT_BOARD_API_URL}")
    print(f"DM Bot User ID: {Config.SUPPORT_BOARD_DM_BOT_USER_ID}")
    print(f"Comment Bot Proxy User ID: {Config.COMMENT_BOT_PROXY_USER_ID}")
    print(f"Human Agent IDs: {Config.SUPPORT_BOARD_AGENT_IDS}")
    print(f"Comment Bot Initiation Tag: '{Config.COMMENT_BOT_INITIATION_TAG}' (None or empty means not used)")
    print(f"WhatsApp Config Loaded: {'Yes' if Config.WHATSAPP_CLOUD_API_TOKEN else 'No'}")
    print(f"Damasco Receiver API URL: {Config.DAMASCO_RECEIVER_API_URL}")
    print(f"Damasco API Secret Loaded: {'Yes' if Config.DAMASCO_API_SECRET else 'No'}")
    # Added print statements for the new config variables
    print(f"Support Dept ID: {Config.SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID}")
    print(f"Sales Dept ID: {Config.SUPPORT_BOARD_SALES_DEPARTMENT_ID}")
    print(f"--------------------")