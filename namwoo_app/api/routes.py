# --- START OF FILE routes.py ---

import logging
import datetime
import hmac
import hashlib
import json
from datetime import timedelta, timezone

from flask import request, jsonify, current_app, abort
from sqlalchemy import text
# Import necessary SQLAlchemy components for session management
from ..utils import db_utils # Make sure this has: is_conversation_paused, pause_conversation_for_duration, (optional) get_pause_record
# --- Import BOTH LLM Services ---
from ..services import openai_service
from ..services.openai_service import extract_customer_info_via_llm
from ..services import google_service
# --- Import Support Board service if needed for error replies ---
from ..services import support_board_service
from ..services.support_board_service import (
    send_order_confirmation_template,
    send_template_by_phone_number,
    route_conversation_to_sales,
)
# Text utilities
from ..utils.text_utils import split_full_name
# ------------------------------
from ..config import Config
from ..models.conversation_pause import ConversationPause # Assuming you have this for explicit pauses

from . import api_bp

logger = logging.getLogger(__name__)

# --- Optional: Helper for Webhook Secret Validation ---
def _validate_sb_webhook_secret(request):
    secret = current_app.config.get('SUPPORT_BOARD_WEBHOOK_SECRET')
    if not secret:
        logger.debug("No SB Webhook Secret configured, skipping validation.")
        return True
    signature_header = request.headers.get('X-Sb-Signature')
    if not signature_header:
        logger.warning("Webhook secret configured, but 'X-Sb-Signature' header missing.")
        return False
    try:
        method, signature_hash = signature_header.split('=', 1)
        if method != 'sha1':
            logger.warning(f"Unsupported webhook signature method: {method}")
            return False
        request_data_bytes = request.get_data()
        mac = hmac.new(secret.encode('utf-8'), msg=request_data_bytes, digestmod=hashlib.sha1)
        expected_signature = mac.hexdigest()
        if hmac.compare_digest(expected_signature, signature_hash):
            logger.debug("Support Board webhook signature validated successfully.")
            return True
        else:
            logger.warning("Invalid Support Board webhook signature.")
            return False
    except ValueError:
        logger.warning(f"Malformed signature header received: {signature_header}")
        return False
    except Exception as e:
        logger.exception(f"Error during webhook signature validation: {e}")
        return False


# --- Support Board Webhook Receiver ---
@api_bp.route('/sb-webhook', methods=['POST'])
def handle_support_board_webhook():
    """
    Receives 'message-sent' webhooks.
    - Differentiates between DM Bot, Comment Bot (proxy), Customer, and Human Agents.
    - Pauses DM Bot on Human Agent intervention.
    - Allows DM Bot to respond after Comment Bot initiates.
    """
    # 1. (Optional) Validate Signature
    # if not _validate_sb_webhook_secret(request):
    #     abort(401, description="Unauthorized: Invalid webhook signature.")

    # 2. Parse Request Body
    try:
        payload = request.get_json(force=True)
        if not payload:
            logger.warning("Received empty payload on /sb-webhook endpoint.")
            abort(400, description="Invalid payload: Empty body.")
        logger.info(f"Received SB Webhook Payload: {json.dumps(payload, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to parse request JSON for SB Webhook: {e}", exc_info=True)
        try: raw_data = request.get_data(as_text=True); logger.error(f"Raw request data (first 500 chars): {raw_data[:500]}")
        except Exception: logger.error("Could not get raw request data either.")
        abort(400, description="Invalid JSON payload received.")

    webhook_function = payload.get('function')
    if webhook_function != 'message-sent':
        logger.debug(f"Ignoring webhook function type: {webhook_function}")
        return jsonify({"status": "ok", "message": "Webhook type ignored"}), 200

    data = payload.get('data', {})
    sb_conversation_id = data.get('conversation_id')
    sender_user_id_str_from_payload = data.get('user_id')
    customer_user_id_str = data.get('conversation_user_id')
    triggering_message_id = data.get('message_id')
    new_user_message_text = data.get('message') # Will be used for tag checking
    conversation_source = data.get('conversation_source')
    order_vars = data.get('order_confirmation_variables')  # Expect list of 8 vars

    if not all([sb_conversation_id, sender_user_id_str_from_payload, customer_user_id_str]):
        missing_keys = [k for k, v in {'conversation_id': sb_conversation_id, 'user_id': sender_user_id_str_from_payload, 'conversation_user_id': customer_user_id_str}.items() if v is None]
        logger.error(f"Missing critical ID data in SB webhook payload's 'data' section. Missing keys: {missing_keys}. Data received: {data}")
        return jsonify({"status": "error", "message": "Webhook payload missing required ID fields"}), 200

    sb_conversation_id_str = str(sb_conversation_id)
    sender_user_id_str = str(sender_user_id_str_from_payload)
    customer_user_id_str = str(customer_user_id_str)

    if order_vars and isinstance(order_vars, list) and len(order_vars) == 8:
        send_order_confirmation_template(
            user_id=customer_user_id_str,
            conversation_id=sb_conversation_id_str,
            variables=order_vars,
        )
        route_conversation_to_sales(sb_conversation_id_str)
        return (
            jsonify({"status": "ok", "message": "Order confirmation sent"}),
            200,
        )

    # --- Get Configured IDs ---
    DM_BOT_ID_STR = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    COMMENT_BOT_PROXY_USER_ID_STR = str(Config.COMMENT_BOT_PROXY_USER_ID) if Config.COMMENT_BOT_PROXY_USER_ID else None
    HUMAN_AGENT_IDS_SET = Config.SUPPORT_BOARD_AGENT_IDS # This is already a set of strings from config.py
    COMMENT_BOT_INITIATION_TAG = Config.COMMENT_BOT_INITIATION_TAG # Can be None or empty string

    pause_minutes = Config.HUMAN_TAKEOVER_PAUSE_MINUTES

    if not DM_BOT_ID_STR:
         logger.critical("FATAL: SUPPORT_BOARD_DM_BOT_USER_ID not configured correctly.")
         return jsonify({"status": "error", "message": "Internal configuration error: DM Bot User ID missing."}), 200
    # Warning for COMMENT_BOT_PROXY_USER_ID is handled implicitly by logic below if tag isn't used.

    logger.info(f"Processing webhook for SB Conv ID: {sb_conversation_id_str} from Sender: {sender_user_id_str}, Customer: {customer_user_id_str}, Source: {conversation_source}, Trigger Msg ID: {triggering_message_id}")

    # --- ORDER OF CHECKS IS IMPORTANT ---

    # 1. Message from DM Bot (Namwoo) itself (echo)
    if sender_user_id_str == DM_BOT_ID_STR:
        logger.info(f"Ignoring own message echo from DM bot (ID: {sender_user_id_str}) in conversation {sb_conversation_id_str}.")
        return jsonify({"status": "ok", "message": "Bot message echo ignored"}), 200

    # 2. Message from the Comment Bot's Proxy User ID (e.g., user "1")
    if COMMENT_BOT_PROXY_USER_ID_STR and sender_user_id_str == COMMENT_BOT_PROXY_USER_ID_STR:
        # If a tag is configured, a message from proxy ID must have the tag to be considered comment bot.
        # Otherwise (tag configured but not present), it's treated as human admin using proxy ID.
        # If no tag is configured, any message from proxy ID is considered comment bot.
        is_comment_bot_message = False
        if COMMENT_BOT_INITIATION_TAG:
            if COMMENT_BOT_INITIATION_TAG in (new_user_message_text or ""):
                is_comment_bot_message = True
                logger.info(f"Message from Comment Bot (proxy ID: {sender_user_id_str}, WITH configured tag) in conv {sb_conversation_id_str}. DM Bot will not reply to this message.")
            else:
                # Message from proxy ID, tag configured, but tag NOT found in message. Assume human admin.
                logger.info(f"Message from human admin using proxy ID {sender_user_id_str} (tag configured but NOT found) in conv {sb_conversation_id_str}. Pausing DM Bot.")
                db_utils.pause_conversation_for_duration(sb_conversation_id_str, duration_seconds=pause_minutes * 60)
                return jsonify({"status": "ok", "message": "Human admin (using proxy ID, tag mismatch) message, bot paused"}), 200
        elif COMMENT_BOT_PROXY_USER_ID_STR: # No tag configured, so any message from proxy ID is comment bot
            is_comment_bot_message = True
            logger.info(f"Message from Comment Bot's proxy (ID: {sender_user_id_str}, no tag configured) in conv {sb_conversation_id_str}. DM Bot will not reply to this message.")
        
        if is_comment_bot_message:
            return jsonify({"status": "ok", "message": "Comment bot proxy message processed"}), 200
        # If it wasn't identified as comment bot here (e.g. proxy ID not configured but sender was "1"), it will be caught by rule 5.


    # 3. Message from a configured DEDICATED HUMAN AGENT (not the proxy ID)
    if sender_user_id_str in HUMAN_AGENT_IDS_SET:
        logger.info(f"Detected message from dedicated human agent (ID: {sender_user_id_str}) in conversation {sb_conversation_id_str}. Pausing DM bot for {pause_minutes} minutes.")
        try:
            db_utils.pause_conversation_for_duration(sb_conversation_id_str, duration_seconds=pause_minutes * 60)
            logger.info(f"Successfully set/updated pause for conversation {sb_conversation_id_str}.")
        except Exception as db_err:
            logger.exception(f"Database error while setting pause for conversation {sb_conversation_id_str}: {db_err}")
        return jsonify({"status": "ok", "message": "Human agent message received, bot paused"}), 200

    # 4. Message from the CUSTOMER
    if sender_user_id_str == customer_user_id_str:
        logger.debug(f"Message from customer {customer_user_id_str}. Checking pause/intervention status for conv {sb_conversation_id_str}.")

        # Check for explicit DB pause
        if db_utils.is_conversation_paused(sb_conversation_id_str):
            # pause_record = db_utils.get_pause_record(sb_conversation_id_str) # Implement if you want to log expiry
            # pause_until_iso = pause_record.paused_until.isoformat() if pause_record else "unknown"
            logger.info(f"Conversation {sb_conversation_id_str} is explicitly paused in DB. DM Bot will not reply.")
            return jsonify({"status": "ok", "message": "Conversation explicitly paused"}), 200

        # Implicit Human Takeover Check
        conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id_str)
        is_implicitly_human_handled = False
        if conversation_data and conversation_data.get('messages'):
            for msg in reversed(conversation_data['messages']): # Check recent messages
                msg_sender_id_history = str(msg.get('user_id'))
                msg_text_history = msg.get('message', '')

                if msg_sender_id_history == customer_user_id_str:
                    continue 

                if msg_sender_id_history == DM_BOT_ID_STR:
                    is_implicitly_human_handled = False # Last was DM bot
                    break 
                
                is_hist_comment_bot = False
                if COMMENT_BOT_PROXY_USER_ID_STR and msg_sender_id_history == COMMENT_BOT_PROXY_USER_ID_STR:
                    if COMMENT_BOT_INITIATION_TAG and COMMENT_BOT_INITIATION_TAG in msg_text_history:
                        is_hist_comment_bot = True
                    elif not COMMENT_BOT_INITIATION_TAG: # No tag configured, assume proxy ID is comment bot
                        is_hist_comment_bot = True
                
                if is_hist_comment_bot:
                    is_implicitly_human_handled = False # Last was Comment Bot
                    break

                # If it's not customer, DM bot, or identified Comment Bot, then it's human/other agent
                is_implicitly_human_handled = True
                logger.info(f"Implicit human takeover detected in conv {sb_conversation_id_str}. Last non-bot/non-customer message from: {msg_sender_id_history}. DM Bot will not reply.")
                # Optionally set an explicit pause here to formalize this
                # db_utils.pause_conversation_for_duration(sb_conversation_id_str, duration_seconds=pause_minutes * 60)
                break
        
        if is_implicitly_human_handled:
            return jsonify({"status": "ok", "message": "Implicit human takeover, bot will not reply"}), 200

        # If not paused and no human intervention, check if message includes full customer data
        if new_user_message_text:
            try:
                customer_data = extract_customer_info_via_llm(new_user_message_text)
            except Exception as info_err:
                logger.exception(
                    f"LLM extraction error for conv {sb_conversation_id_str}: {info_err}"
                )
                customer_data = None
            required_keys = [
                "full_name",
                "cedula",
                "telefono",
                "correo",
                "direccion",
                "productos",
                "total",
            ]
            if customer_data and all(customer_data.get(k) for k in required_keys):
                nombre, apellido = split_full_name(str(customer_data["full_name"]))
                params = [
                    str(nombre),
                    str(apellido),
                    str(customer_data["cedula"]).strip(),
                    str(customer_data["telefono"]).strip(),
                    str(customer_data["correo"]).strip(),
                    str(customer_data["direccion"]).strip(),
                    str(customer_data["productos"]).strip(),
                    str(customer_data["total"]).strip(),
                ]
                phone = str(customer_data.get("telefono", "")).strip()
                if phone:
                    send_template_by_phone_number(
                        phone_number=phone,
                        template_params=params,
                    )
                    route_conversation_to_sales(sb_conversation_id_str)
                    return (
                        jsonify({"status": "ok", "message": "Template sent via phone"}),
                        200,
                    )

        # If not paused and no human intervention, proceed with LLM
        provider = current_app.config.get('LLM_PROVIDER', 'openai').lower()
        logger.info(f"Conversation {sb_conversation_id_str} is not paused and no overriding human intervention. Triggering processing using LLM Provider: {provider}.")

        process_args = {
            "sb_conversation_id": sb_conversation_id_str,
            "new_user_message": new_user_message_text,
            "conversation_source": conversation_source,
            "sender_user_id": sender_user_id_str, 
            "customer_user_id": customer_user_id_str,
            "triggering_message_id": str(triggering_message_id) if triggering_message_id is not None else None
        }

        try:
            if provider == 'google':
                logger.debug(f"Calling google_service.process_new_message_gemini with args: {process_args}")
                google_service.process_new_message_gemini(**process_args)
            elif provider == 'openai':
                logger.debug(f"Calling openai_service.process_new_message with args: {process_args}")
                openai_service.process_new_message(**process_args)
            else:
                logger.error(f"Invalid LLM_PROVIDER configured: '{provider}' for conv {sb_conversation_id_str}.")
                support_board_service.send_reply_to_channel(
                    conversation_id=sb_conversation_id_str,
                    message_text=f"Disculpa, hay un problema de configuración interna (Proveedor IA: {provider}).",
                    source=conversation_source,
                    target_user_id=customer_user_id_str,
                    conversation_details=None, 
                    triggering_message_id=process_args["triggering_message_id"]
                )
                return jsonify({"status": "error", "message": f"Invalid LLM provider: {provider}"}), 200

            return jsonify({"status": "ok", "message": f"Customer message processing initiated via {provider}"}), 200

        except Exception as e:
            logger.exception(f"Error triggering {provider}_service processing for SB conv {sb_conversation_id_str}: {e}")
            support_board_service.send_reply_to_channel(
                 conversation_id=sb_conversation_id_str,
                 message_text="Lo siento, ocurrió un error inesperado al intentar procesar tu mensaje.",
                 source=conversation_source,
                 target_user_id=customer_user_id_str,
                 conversation_details=None,
                 triggering_message_id=process_args["triggering_message_id"]
             )
            return jsonify({"status": "error", "message": "Error occurred during message processing trigger"}), 200
    
    # 5. Message from an unrecognized sender (not DM bot, not Comment Bot proxy, not configured Human Agent, not Customer)
    # This often means an admin user (like User "1" if COMMENT_BOT_PROXY_USER_ID is different or tag logic didn't catch it as comment bot)
    # or an agent whose ID is not in HUMAN_AGENT_IDS_SET. Treat as human intervention.
    logger.warning(f"Received message in conv {sb_conversation_id_str} from unhandled/unidentified agent {sender_user_id_str}. Assuming human intervention and pausing.")
    try:
        db_utils.pause_conversation_for_duration(sb_conversation_id_str, duration_seconds=pause_minutes * 60)
    except Exception as db_err:
        logger.exception(f"Database error while setting pause for unhandled sender in conversation {sb_conversation_id_str}: {db_err}")
    return jsonify({"status": "ok", "message": "Message from unhandled sender type (assumed agent), bot paused"}), 200


# --- Health Check Endpoint (Keep as is) ---
@api_bp.route('/health', methods=['GET'])
def health_check():
    logger.debug("Health check endpoint hit.")
    db_ok = False
    try:
        with db_utils.get_db_session() as session: # Corrected to use db_utils
            session.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        db_ok = False
    return jsonify({"status": "ok", "database_connected": db_ok}), 200

# --- Test Endpoint (Keep or remove as needed) ---
@api_bp.route('/supportboard/test', methods=['GET'])
def handle_support_board_test():
    endpoint_name = "/api/supportboard/test"
    logger.info(f"--- TEST HIT --- Endpoint {endpoint_name} was successfully reached via GET request.")
    response_data = {
        "status": "success",
        "message": f"Namwoo endpoint {endpoint_name} reached successfully!",
        "timestamp": datetime.datetime.now(timezone.utc).isoformat()
    }
    return jsonify([response_data]), 200