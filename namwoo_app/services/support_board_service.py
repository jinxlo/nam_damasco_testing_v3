# -*- coding: utf-8 -*-
import requests
import logging
import json
import re # Import regex module for cleaning phone numbers
from flask import current_app
from typing import Optional, List, Dict, Any

# Assuming Config is correctly imported and loads .env variables
from ..config import Config

logger = logging.getLogger(__name__)

# --- PRIVATE HELPER: Make Support Board API Call ---
# (Kept unchanged)
def _call_sb_api(payload: Dict) -> Optional[Any]:
    """Internal helper to make POST requests to the Support Board API."""
    api_url = current_app.config.get('SUPPORT_BOARD_API_URL')
    api_token = current_app.config.get('SUPPORT_BOARD_API_TOKEN')

    if not api_url or not api_token:
        logger.error("Support Board API URL or Token is not configured.")
        return None

    payload['token'] = api_token
    function_name = payload.get('function', 'N/A')
    logger.debug(f"Calling SB API URL: {api_url} with function: {function_name}")
    try:
        log_payload = payload.copy()
        if 'token' in log_payload:
            log_payload['token'] = '***' + log_payload['token'][-4:] if len(log_payload.get('token','')) > 4 else '***'
        log_payload_str = json.dumps(log_payload)
    except Exception:
        log_payload_str = str(payload)
    logger.debug(f"Payload for {function_name} (requests data param): {log_payload_str}")

    try:
        response = requests.post(api_url, data=payload, timeout=20)
        response.raise_for_status()
        response_json = response.json()
        try:
            log_response_str = json.dumps(response_json)
        except Exception:
            log_response_str = str(response_json)
        logger.debug(f"Raw SB API response for {function_name}: {log_response_str}")

        if response_json.get("success") is True:
             return response_json.get("response")
        else:
            error_detail = response_json.get("response", f"API call failed for function '{function_name}' with success=false or missing")
            logger.error(f"Support Board API reported failure for {function_name}: {error_detail}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error calling Support Board API ({function_name}): {e}", exc_info=True)
        if e.response is not None:
            logger.error(f"Response body from failed request: {e.response.text[:500]}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        raw_text = getattr(response, 'text', 'N/A')
        logger.error(f"Failed to decode JSON response from Support Board API ({function_name}): {e}. Response text: {raw_text[:500]}", exc_info=True)
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling SB API ({function_name}): {e}")
        return None

# --- Public Function: Get Conversation Data ---
# (Kept unchanged)
def get_sb_conversation_data(conversation_id: str) -> Optional[Dict]:
    """Fetches the full conversation details from Support Board."""
    payload = {
        'function': 'get-conversation',
        'conversation_id': conversation_id
    }
    logger.info(f"Attempting to fetch full conversation data from Support Board API for ID: {conversation_id}")
    response_data = _call_sb_api(payload)
    if isinstance(response_data, dict):
        if "messages" in response_data and "details" in response_data:
            logger.info(f"Successfully fetched conversation data for SB conversation {conversation_id}")
            return response_data
        else:
            logger.warning(f"SB API get-conversation success reported, but response for {conversation_id} might be incomplete or malformed. Response: {response_data}")
            return response_data # Return potentially malformed but successful response
    else:
        logger.error(f"Failed to fetch or parse valid conversation data dictionary for SB conversation {conversation_id}. Raw response from _call_sb_api call was not a valid dictionary: {response_data}")
        return None

# --- PRIVATE HELPER: Get User PSID (for FB/IG) ---
# (Kept unchanged - uses get-user + extra=true)
def _get_user_psid(user_id: str) -> Optional[str]:
    """Fetches user details and extracts the PSID (Facebook/Instagram ID)."""
    logger.info(f"Attempting to fetch user details for User ID: {user_id} to get PSID.")
    payload = {'function': 'get-user', 'user_id': user_id, 'extra': 'true'}
    user_data = _call_sb_api(payload)
    if user_data and isinstance(user_data, dict):
        details_list = user_data.get('details', [])
        expected_slug = 'facebook-id' # As per common SB setup, adjust if different
        if isinstance(details_list, list):
            for detail in details_list:
                if isinstance(detail, dict) and detail.get('slug') == expected_slug:
                    psid = detail.get('value')
                    if psid and isinstance(psid, str) and psid.strip():
                        logger.info(f"Found PSID (using slug '{expected_slug}') for User ID {user_id}")
                        return psid.strip()
                    else:
                        logger.warning(f"Found '{expected_slug}' detail slug for user {user_id} but its value is empty or invalid: '{psid}'")
            logger.warning(f"Could not find PSID using slug '{expected_slug}' in the details list for User ID: {user_id}. Details received: {details_list}")
            return None
        else:
             logger.warning(f"User details for {user_id} received, but 'details' key is not a list: {details_list}")
             return None
    else:
        logger.error(f"Failed to fetch or parse valid user details dictionary for User ID: {user_id} needed for PSID lookup.")
        return None

# --- CORRECTED PRIVATE HELPER: Get User WAID (for WhatsApp) ---
# (Kept unchanged from your provided file)
def _get_user_waid(user_id: str) -> Optional[str]:
    """
    Fetches user details from SB API using 'get-user' + 'extra=true'
    and extracts/formats the WAID. Ignores any pre-fetched data.
    Prioritizes 'phone' detail slug, falls back to 'first_name' if it looks like a number.
    Requires WHATSAPP_DEFAULT_COUNTRY_CODE in config for numbers missing the prefix.
    """
    logger.info(f"Attempting to get WAID for User ID: {user_id}. ALWAYS fetching user details via 'get-user' + 'extra=true'.")
    phone_number = None
    user_first_name = None
    user_details_data = None # Ensure we fetch fresh data

    # --- ALWAYS Fetch user details using get-user + extra=true ---
    logger.debug(f"Fetching user details for {user_id} using 'get-user' + 'extra=true'.")
    payload = {'function': 'get-user', 'user_id': str(user_id), 'extra': 'true'}
    user_details_data = _call_sb_api(payload)
    # --- End Fetch ---

    if user_details_data and isinstance(user_details_data, dict):
        # Extract first name for potential fallback
        user_first_name = user_details_data.get('first_name')
        details_list = user_details_data.get('details', []) # Expecting a list here from get-user

        if isinstance(details_list, list):
            # Try to find the phone number in the details list
            for detail in details_list:
                if isinstance(detail, dict) and detail.get('slug') == 'phone':
                    phone_value = detail.get('value')
                    if phone_value and isinstance(phone_value, str) and phone_value.strip():
                        phone_number = phone_value.strip()
                        logger.info(f"Found phone number for User ID {user_id} in 'phone' detail from get-user response.")
                        break # Found it, stop looking
                    else:
                        logger.warning(f"Found 'phone' detail slug for user {user_id} but its value is empty or invalid: '{phone_value}'")

            if not phone_number:
                logger.warning(f"Could not find valid 'phone' detail in the details list returned by get-user for User ID {user_id}. Details received: {details_list}. Will check first_name as fallback.")
        else:
            logger.warning(f"User details fetched via get-user for {user_id} received, but 'details' key is not a list or is missing. Response: {user_details_data}")
            # Continue to check first_name fallback
    else:
        # This error covers cases where the _call_sb_api call for get-user failed
        logger.error(f"Failed to fetch or parse valid user details dictionary via get-user for User ID: {user_id} needed for WAID lookup.")
        return None # Cannot proceed without user details

    # --- Fallback: Use first_name if phone number wasn't found in details ---
    if not phone_number:
        if user_first_name and isinstance(user_first_name, str):
            cleaned_first_name = user_first_name.strip()
            if re.fullmatch(r'[\d\s\-\(\)\+]+', cleaned_first_name) and len(re.sub(r'\D', '', cleaned_first_name)) >= 7:
                logger.info(f"Using 'first_name' field '{cleaned_first_name}' as fallback phone number for User ID {user_id}.")
                phone_number = cleaned_first_name
            else:
                 logger.warning(f"'first_name' for user {user_id} ('{cleaned_first_name}') does not appear to be a valid phone number. Cannot use as fallback.")
        else:
            logger.warning(f"Cannot fallback to first_name for user {user_id}: field is missing, not a string, or empty.")


    if not phone_number:
        logger.error(f"Could not determine phone number for User ID {user_id} from get-user details or first_name fallback.")
        return None

    # --- Format the phone number into WAID (Kept Unchanged) ---
    waid = re.sub(r'\D', '', phone_number) # Remove non-digits

    if not phone_number.lstrip().startswith('+') and not waid.startswith(Config.WHATSAPP_DEFAULT_COUNTRY_CODE or ''):
        default_cc = Config.WHATSAPP_DEFAULT_COUNTRY_CODE
        if default_cc:
            default_cc_digits = re.sub(r'\D', '', default_cc) # Clean the default CC itself
            if default_cc_digits: # Check if the cleaned default CC has digits
                logger.warning(f"Phone number '{phone_number}' for user {user_id} appears to be missing country code prefix. Prepending default: '{default_cc_digits}'.")
                waid = default_cc_digits + waid
            else:
                logger.error(f"Configured WHATSAPP_DEFAULT_COUNTRY_CODE ('{default_cc}') contains no digits. Cannot prepend.")
                return None # Cannot form valid WAID
        else:
            logger.error(f"Phone number '{phone_number}' for user {user_id} is missing country code prefix, and WHATSAPP_DEFAULT_COUNTRY_CODE is not set or is invalid. Cannot form valid WAID.")
            return None
    elif not phone_number.lstrip().startswith('+') and waid.startswith(Config.WHATSAPP_DEFAULT_COUNTRY_CODE or ''):
        logger.debug(f"Phone number '{phone_number}' for user {user_id} seemed to be missing '+' but already started with the default country code. Assuming it's correct.")


    logger.info(f"Successfully derived WAID '{waid}' for User ID {user_id}.")
    return waid


# --- PRIVATE HELPER: Send Messenger/Instagram Message (External Delivery via SB API) ---
def _send_messenger_message( # This function was already corrected in previous iterations and had detailed logging.
    psid: str,
    page_id: str,
    message_text: str,
    conversation_id: str,
    triggering_message_id: Optional[str]
) -> bool:
    """Sends a message via the SB messenger-send-message API."""
    logger.debug(f"[_send_messenger_message CALLED] Conv ID: {conversation_id}")
    logger.debug(f"[_send_messenger_message] Received triggering_message_id: {repr(triggering_message_id)} (Type: {type(triggering_message_id)})")

    # --- MODIFICATION: Use SUPPORT_BOARD_DM_BOT_USER_ID and update warning ---
    bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id:
        logger.warning(f"SUPPORT_BOARD_DM_BOT_USER_ID not configured. This might affect how messages are attributed or if internal logging for the DM bot itself works as expected for this specific _send_messenger_message call, though external send will proceed if other params are fine.")
    # --- END MODIFICATION ---

    logger.info(f"Attempting to send Messenger/IG message via specific SB API for Conv ID {conversation_id} to PSID: ...{psid[-6:]} on Page ID: {page_id}")

    payload = {
        'function': 'messenger-send-message',
        'psid': psid,
        'facebook_page_id': page_id,
        'message': message_text
    }

    if triggering_message_id is not None and str(triggering_message_id).strip() != '':
        logger.info(f"Including metadata (triggering message ID: {triggering_message_id}) in messenger-send-message call for conv {conversation_id}.")
        payload['metadata'] = str(triggering_message_id)
    else:
        logger.warning(f"No triggering message ID available/valid for conv {conversation_id}. Sending messenger message without metadata. Dashboard linking might fail.")

    try:
        log_payload_msg = json.dumps(payload)
    except Exception:
        log_payload_msg = str(payload)
    logger.debug(f"[_send_messenger_message] Final payload before API call: {log_payload_msg}")

    response_data = _call_sb_api(payload)
    logger.debug(f"[_send_messenger_message] response_data from _call_sb_api: {response_data} (Type: {type(response_data)})") # Existing debug log

    is_list = isinstance(response_data, list)
    logger.debug(f"[_send_messenger_message] Check: isinstance(response_data, list) = {is_list}") # Existing debug log

    if is_list:
        list_len = len(response_data)
        logger.debug(f"[_send_messenger_message] Check: len(response_data) = {list_len}") # Existing debug log
        if list_len > 0:
            first_element_is_dict = isinstance(response_data[0], dict)
            logger.debug(f"[_send_messenger_message] Check: isinstance(response_data[0], dict) = {first_element_is_dict}") # Existing debug log
            if first_element_is_dict:
                has_recipient_id = 'recipient_id' in response_data[0]
                has_message_id = 'message_id' in response_data[0]
                logger.debug(f"[_send_messenger_message] Check: 'recipient_id' in response_data[0] = {has_recipient_id}") # Existing debug log
                logger.debug(f"[_send_messenger_message] Check: 'message_id' in response_data[0] = {has_message_id}") # Existing debug log

                if has_recipient_id and has_message_id:
                    fb_message_id = response_data[0].get('message_id', 'N/A')
                    logger.info(f"Messenger/IG message acknowledged as successful by SB API (FB Msg ID: {fb_message_id}) for Conv ID {conversation_id} to PSID ...{psid[-6:]} (Parsed from list response)")
                    return True

    is_dict = isinstance(response_data, dict)
    logger.debug(f"[_send_messenger_message] Fallback Check: isinstance(response_data, dict) = {is_dict}") # Existing debug log
    if is_dict:
        has_recipient_id_dict = 'recipient_id' in response_data
        has_message_id_dict = 'message_id' in response_data
        logger.debug(f"[_send_messenger_message] Fallback Check: 'recipient_id' in response_data = {has_recipient_id_dict}") # Existing debug log
        logger.debug(f"[_send_messenger_message] Fallback Check: 'message_id' in response_data = {has_message_id_dict}") # Existing debug log
        if has_recipient_id_dict and has_message_id_dict:
            fb_message_id = response_data.get('message_id', 'N/A')
            logger.info(f"Messenger/IG message acknowledged as successful by SB API (FB Msg ID: {fb_message_id}) for Conv ID {conversation_id} to PSID ...{psid[-6:]} (Parsed from dict response)")
            return True

    is_true_bool = response_data is True
    logger.debug(f"[_send_messenger_message] Fallback Check: response_data is True = {is_true_bool}") # Existing debug log
    if is_true_bool:
        logger.warning(f"Messenger/IG message API call for Conv ID {conversation_id} returned 'True', which differs from documented structure, but treating as success.")
        return True

    logger.error(f"Failed to send Messenger/IG message via SB API for Conv ID {conversation_id} to PSID ...{psid[-6:]}. Unexpected response structure after all checks: {response_data}")
    return False


# --- PRIVATE HELPER: Add Message Internally to SB (Dashboard Visibility) ---
# (Kept unchanged)
def _add_internal_sb_message(conversation_id: str, message_text: str, bot_user_id: str) -> bool:
    """Adds a message internally to the SB conversation using send-message."""
    if not bot_user_id:
        logger.error("Cannot add internal SB message: Bot User ID not provided or configured.") # This log is fine as bot_user_id is a param here
        return False

    logger.info(f"Adding bot reply internally to SB conversation ID: {conversation_id} as User ID: {bot_user_id}")
    payload = {
        'function': 'send-message',
        'user_id': bot_user_id,
        'conversation_id': conversation_id,
        'message': message_text,
        'attachments': json.dumps([]) # Send as empty JSON array string
    }
    response_data = _call_sb_api(payload) # _call_sb_api returns the "response" part of the JSON

    if isinstance(response_data, dict) and ('id' in response_data or 'message-id' in response_data):
        internal_msg_id = response_data.get('id', response_data.get('message-id', 'N/A'))
        logger.info(f"Internal SB message added successfully (Internal Msg ID: {internal_msg_id}) to conversation {conversation_id}")
        return True
    elif response_data is True: # Handle cases where SB API simply returns True for success
         logger.info(f"Internal SB message add attempt reported 'response': True for conversation {conversation_id}, treating as success.")
         return True
    else:
        logger.error(f"Failed to add internal SB message to conversation {conversation_id}. API response: {response_data}")
        return False


# --- NEW PRIVATE HELPER: Send WhatsApp Message DIRECTLY via Meta Cloud API ---
# (Kept unchanged)
def _send_whatsapp_cloud_api(recipient_waid: str, message_text: str) -> bool:
    token = Config.WHATSAPP_CLOUD_API_TOKEN
    phone_number_id = Config.WHATSAPP_PHONE_NUMBER_ID
    api_version = Config.WHATSAPP_API_VERSION

    if not token or not phone_number_id:
        logger.error("WhatsApp Cloud API Token or Phone Number ID not configured. Cannot send direct message.")
        return False
    api_url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload_dict = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_waid,
        "type": "text",
        "text": { "preview_url": False, "body": message_text }
    }
    logger.info(f"Attempting to send direct WhatsApp message via Meta Cloud API to WAID: ...{recipient_waid[-6:]}")
    logger.debug(f"Direct WhatsApp API URL: {api_url}")
    try: log_payload = payload_dict.copy(); logger.debug(f"Direct WhatsApp API Payload: {json.dumps(log_payload)}")
    except Exception: logger.debug(f"Direct WhatsApp API Payload (fallback log): {str(payload_dict)}")

    try:
        response = requests.post(api_url, headers=headers, json=payload_dict, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        try: log_response_str = json.dumps(response_json)
        except Exception: log_response_str = str(response_json)
        logger.debug(f"Direct WhatsApp API Raw Response: {log_response_str}")
        if isinstance(response_json, dict) and \
           response_json.get("messaging_product") == "whatsapp" and \
           isinstance(response_json.get("messages"), list) and \
           len(response_json["messages"]) > 0 and \
           isinstance(response_json["messages"][0], dict) and \
           "id" in response_json["messages"][0]:
            message_wamid = response_json["messages"][0]["id"]
            logger.info(f"Direct WhatsApp API call successful. Message WAMID: {message_wamid}")
            return True
        else:
            logger.error(f"Direct WhatsApp API call returned unexpected success structure: {response_json}")
            return False
    except requests.exceptions.HTTPError as http_err:
        response_text = http_err.response.text if http_err.response else "N/A"
        logger.error(f"Direct WhatsApp API HTTP error: {http_err.response.status_code} - {response_text}", exc_info=False)
        return False
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Direct WhatsApp API request error: {req_err}", exc_info=True)
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during direct WhatsApp API call: {e}")
        return False

# --- NEW PRIVATE HELPER: Send Telegram Message (External Delivery via SB API) ---
# (Kept unchanged)
def _send_telegram_message(
    chat_id: str,
    message_text: str,
    conversation_id: Optional[str]
) -> bool:
    logger.info(f"Attempting to send Telegram message via SB API to Chat ID: {chat_id} for SB Conv ID: {conversation_id or 'N/A'}")
    payload = {
        'function': 'telegram-send-message',
        'chat_id': chat_id,
        'message': message_text,
        'attachments': json.dumps([])
    }
    if conversation_id: payload['conversation_id'] = conversation_id
    response_data = _call_sb_api(payload)
    if isinstance(response_data, dict) and \
       response_data.get("ok") is True and \
       isinstance(response_data.get("result"), dict) and \
       "message_id" in response_data["result"]:
        tg_message_id = response_data["result"].get("message_id", "N/A")
        logger.info(f"Telegram message acknowledged as successful by SB API (TG Msg ID: {tg_message_id}) for Chat ID {chat_id}, SB Conv ID {conversation_id or 'N/A'}")
        return True
    elif response_data is True:
         logger.warning(f"Telegram message API call for Chat ID {chat_id} returned 'True', which differs from documented structure, but treating as success.")
         return True
    else:
        logger.error(f"Failed to send Telegram message via SB API for Chat ID {chat_id}, SB Conv ID {conversation_id or 'N/A'}. Unexpected response: {response_data}")
        return False


# --- REVISED PUBLIC FUNCTION: send_reply_to_channel (Added Telegram Handler) ---
def send_reply_to_channel(
    conversation_id: str,
    message_text: str,
    source: Optional[str],
    target_user_id: str,
    conversation_details: Optional[Dict],
    triggering_message_id: Optional[str]
) -> bool:
    if not message_text or not message_text.strip():
        logger.warning(f"Attempted to send empty reply to conversation {conversation_id}. Skipping.")
        return False

    effective_source = source.strip().lower() if isinstance(source, str) and source.strip() else 'web'
    logger.info(f"Routing reply for conversation {conversation_id} to target customer User ID {target_user_id} via effective source channel '{effective_source}'")

    external_success = False
    # --- MODIFICATION: Use SUPPORT_BOARD_DM_BOT_USER_ID ---
    dm_bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    # --- END MODIFICATION ---

    if effective_source == 'wa':
        logger.info(f"Processing WA reply for conversation {conversation_id} using Direct Cloud API.")
        recipient_waid = _get_user_waid(target_user_id)
        if not recipient_waid:
             logger.error(f"Cannot send WA reply to conv {conversation_id}: Failed to get recipient WAID for user {target_user_id}.")
             return False
        logger.info(f"Step 1 (WA - Direct): Sending externally via Meta Cloud API for conv {conversation_id}")
        external_success = _send_whatsapp_cloud_api(recipient_waid, message_text)
        if external_success:
            logger.info(f"Step 2 (WA - Direct): External send successful for conv {conversation_id}. Adding message internally via SB send-message.")
            if dm_bot_user_id: # Check the specifically fetched DM Bot ID
                internal_add_success = _add_internal_sb_message(
                    conversation_id=conversation_id,
                    message_text=message_text,
                    bot_user_id=dm_bot_user_id # Use the DM Bot ID for internal logging
                )
                if not internal_add_success:
                    logger.error(f"Failed to add WA message internally to SB dashboard for conv {conversation_id} after successful direct external send.")
            else:
                # --- MODIFICATION: Updated error log ---
                logger.error("Cannot add WA message internally to SB dashboard: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
                # --- END MODIFICATION ---
        else:
             logger.error(f"Direct external WA send via Meta Cloud API failed for conv {conversation_id}.")
        return external_success

    elif effective_source in ['fb', 'ig']:
        logger.info(f"Processing FB/IG reply for conversation {conversation_id} using SB API.")
        conv_details = conversation_details
        if not conv_details:
            logger.info(f"Conversation details not provided for FB/IG {conversation_id}, fetching...")
            conv_details = get_sb_conversation_data(conversation_id)
            if not conv_details:
                logger.error(f"Cannot send FB/IG reply to conv {conversation_id}: Failed to fetch conversation details.")
                return False
        psid = _get_user_psid(target_user_id)
        page_id = conv_details.get('details', {}).get('extra')
        page_id_str = str(page_id).strip() if page_id else None
        if psid and page_id_str:
            logger.info(f"Step 1 (FB/IG - SB): Sending externally via messenger-send-message for conv {conversation_id}")
            external_success = _send_messenger_message(
                psid=psid,
                page_id=page_id_str,
                message_text=message_text,
                conversation_id=conversation_id,
                triggering_message_id=triggering_message_id
            )
            if external_success:
                logger.info(f"Step 2 (FB/IG - SB): External send successful for conv {conversation_id}. Adding message internally via SB send-message.")
                if dm_bot_user_id: # Check the specifically fetched DM Bot ID
                    internal_add_success = _add_internal_sb_message(
                        conversation_id=conversation_id,
                        message_text=message_text,
                        bot_user_id=dm_bot_user_id # Use the DM Bot ID for internal logging
                    )
                    if not internal_add_success:
                        logger.error(f"Failed to add FB/IG message internally to SB dashboard for conv {conversation_id} after successful external send.")
                else:
                    # --- MODIFICATION: Updated error log ---
                    logger.error("Cannot add FB/IG message internally to SB dashboard: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
                    # --- END MODIFICATION ---
            else:
                logger.error(f"External FB/IG send via messenger-send-message failed for conv {conversation_id}.")
            return external_success
        else:
            error_details_list = []
            if not psid: error_details_list.append(f"PSID not found for user {target_user_id}")
            if not page_id_str: error_details_list.append("Page ID not found in conversation details 'extra' field")
            reason = ", ".join(error_details_list)
            logger.error(f"Cannot send FB/IG reply to conv {conversation_id}: Required IDs missing ({reason}).")
            return False

    elif effective_source == 'tg':
        logger.info(f"Processing Telegram reply for conversation {conversation_id} using SB API.")
        conv_details = conversation_details
        if not conv_details:
            logger.info(f"Conversation details not provided for Telegram conv {conversation_id}, fetching...")
            conv_details = get_sb_conversation_data(conversation_id)
            if not conv_details:
                logger.error(f"Cannot send Telegram reply to conv {conversation_id}: Failed to fetch conversation details.")
                return False
        chat_id_from_extra = conv_details.get('details', {}).get('extra')
        if not chat_id_from_extra:
            logger.error(f"Cannot send Telegram reply to conv {conversation_id}: chat_id (from details.extra) not found in conversation details. Details: {conv_details.get('details')}")
            return False
        chat_id = str(chat_id_from_extra).strip()
        if not chat_id:
            logger.error(f"Cannot send Telegram reply to conv {conversation_id}: chat_id (from details.extra) is empty after stripping. Original: '{chat_id_from_extra}'")
            return False

        logger.info(f"Step 1 (TG - SB): Sending externally via telegram-send-message for conv {conversation_id} to Chat ID {chat_id}")
        external_success = _send_telegram_message(
            chat_id=chat_id,
            message_text=message_text,
            conversation_id=conversation_id
        )
        if external_success:
            logger.info(f"Step 2 (TG - SB): External send successful for conv {conversation_id}. Adding message internally via SB send-message.")
            if dm_bot_user_id: # Check the specifically fetched DM Bot ID
                internal_add_success = _add_internal_sb_message(
                    conversation_id=conversation_id,
                    message_text=message_text,
                    bot_user_id=dm_bot_user_id # Use the DM Bot ID for internal logging
                )
                if not internal_add_success:
                    logger.error(f"Failed to add Telegram message internally to SB dashboard for conv {conversation_id} after successful external send.")
            else:
                # --- MODIFICATION: Updated error log ---
                logger.error("Cannot add Telegram message internally to SB dashboard: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
                # --- END MODIFICATION ---
        else:
            logger.error(f"External Telegram send via SB API failed for conv {conversation_id}.")
        return external_success

    else:
        logger.warning(f"Unhandled conversation source '{effective_source}' for conv {conversation_id}. Message not sent.")
        return False


# --- NEW PUBLIC FUNCTIONS: Order Confirmation Template & Sales Routing ---
def send_order_confirmation_template(user_id: str, conversation_id: str, variables: list):
    """Send WhatsApp order confirmation template with provided variables."""
    payload = {
        "function": "messaging-platforms-send-template",
        "user": {"id": user_id},
        "conversation_id": conversation_id,
        "source": {"source": "wa"},
        "template": {
            "name": "confirmacion_datos_cliente",
            "language": "es",
            "parameters": variables,
        },
    }
    return _call_sb_api(payload)


def send_template_by_phone_number(phone_number: str, template_params: list):
    """Send WhatsApp template using a raw phone number in international format."""
    payload = {
        "function": "messaging-platforms-send-template",
        "source": {"source": "wa"},
        "phone": phone_number,
        "template": {
            "name": "confirmacion_datos_cliente",
            "language": "es",
            "parameters": template_params,
        },
    }
    return _call_sb_api(payload)


def route_conversation_to_sales(conversation_id: str) -> None:
    """Assign conversation to the Sales department and disable the bot."""
    sales_department_id = Config.SUPPORT_BOARD_SALES_DEPARTMENT_ID

    _call_sb_api(
        {
            "function": "update-conversation-department",
            "conversation_id": conversation_id,
            "department": sales_department_id,
        }
    )

    _call_sb_api(
        {
            "function": "sb-human-takeover",
            "conversation_id": conversation_id,
        }
    )
