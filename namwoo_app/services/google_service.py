# NAMWOO/services/google_service.py
# -*- coding: utf-8 -*-
import logging
import json
import time # Kept as it was in your original file
from typing import List, Dict, Optional, Tuple, Union, Any

from openai import ( # This lib is used for the OpenAI-compatible Gemini endpoint
    OpenAI,
    APIError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
)
from flask import current_app # Kept as it was in your original file

# ── Local services ──────────────────────────────────────────────────────────────
from . import product_service, support_board_service
from ..config import Config
from ..utils import conversation_location
# from ..utils.text_utils import strip_html_to_text # Not strictly needed here if llm_processing_service pre-strips

logger = logging.getLogger(__name__)

# ── Initialise OpenAI‑compatible client that talks to Google Gemini ────────────
google_gemini_client_via_openai_lib: Optional[OpenAI] = None
GOOGLE_SDK_AVAILABLE = False

try:
    google_api_key = Config.GOOGLE_API_KEY
    # Your original base URL, assuming it's correct for your setup
    google_base_url_for_openai_lib = "https://generativelanguage.googleapis.com/v1beta/openai/"

    if google_api_key:
        timeout_seconds = getattr(Config, 'GOOGLE_REQUEST_TIMEOUT', 60.0) # Make timeout configurable
        google_gemini_client_via_openai_lib = OpenAI(
            api_key=google_api_key,
            base_url=google_base_url_for_openai_lib,
            timeout=timeout_seconds, # Use configured timeout
        )
        logger.info(
            "OpenAI-lib client initialised for Google Gemini at %s with timeout %s.",
            google_base_url_for_openai_lib, timeout_seconds
        )
        GOOGLE_SDK_AVAILABLE = True
    else:
        logger.error("GOOGLE_API_KEY not configured. Gemini service (via OpenAI lib) disabled.")
        GOOGLE_SDK_AVAILABLE = False # Ensure it's set if no key
except Exception as e: # Your original general Exception catch, added 'e' for logging
    logger.exception(f"Failed to initialise Google Gemini client (via OpenAI lib): {e}")
    google_gemini_client_via_openai_lib = None
    GOOGLE_SDK_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = Config.MAX_HISTORY_MESSAGES if hasattr(Config, 'MAX_HISTORY_MESSAGES') else 10
TOOL_CALL_RETRY_LIMIT = 2 
DEFAULT_GOOGLE_MODEL = Config.GOOGLE_GEMINI_MODEL if hasattr(Config, 'GOOGLE_GEMINI_MODEL') else "gemini-1.5-flash-latest" # Ensure it's just the model ID
DEFAULT_MAX_TOKENS = Config.GOOGLE_MAX_TOKENS if hasattr(Config, 'GOOGLE_MAX_TOKENS') else 1024
DEFAULT_GOOGLE_TEMPERATURE = getattr(Config, "GOOGLE_TEMPERATURE", 0.7) # Add temperature from Config

# --- PRODUCT DESCRIPTION SUMMARIZATION (using Google Gemini via OpenAI lib) ---
def get_google_product_summary(
    plain_text_description: str, 
    item_name: Optional[str] = None
) -> Optional[str]:
    global google_gemini_client_via_openai_lib
    if not GOOGLE_SDK_AVAILABLE or not google_gemini_client_via_openai_lib:
        logger.error("Google Gemini client not available. Cannot summarize description with Google.")
        return None
    if not plain_text_description or not plain_text_description.strip():
        logger.debug("Google summarizer: No plain text description provided to summarize.")
        return None

    prompt_context_parts = []
    if item_name:
        prompt_context_parts.append(f"Nombre del Producto: {item_name}")
    prompt_context_parts.append(f"Descripción Original (texto plano):\n{plain_text_description}")
    prompt_context = "\n".join(prompt_context_parts)
    
    system_prompt_for_gemini = ( # Kept your exact system prompt
        "Eres un redactor experto en comercio electrónico. Resume la siguiente descripción de producto. "
        "El resumen debe ser conciso (objetivo: 50-75 palabras, 2-3 frases clave), resaltar los principales beneficios y características, y ser factual. "
        "Evita la jerga de marketing, la repetición y frases como 'este producto es'. "
        "La salida debe ser texto plano adecuado para una base de datos de productos y un asistente de IA. "
        "No incluyas etiquetas HTML."
    )

    max_input_chars_for_summary = 3000 # Your original value
    if len(prompt_context) > max_input_chars_for_summary:
        cutoff_point = prompt_context.rfind('.', 0, max_input_chars_for_summary)
        if cutoff_point == -1: cutoff_point = max_input_chars_for_summary
        prompt_context = prompt_context[:cutoff_point] + " [DESCRIPCIÓN TRUNCADA]"
        logger.warning(f"Google summarizer: Description for '{item_name or 'Unknown'}' was truncated for prompt construction.")

    messages_for_api = [
        {"role": "system", "content": system_prompt_for_gemini},
        {"role": "user", "content": f"Por favor, resume la siguiente información del producto:\n\n{prompt_context}"}
    ]
    
    # Use specific summary model if configured, else default chat model
    gemini_model_name = getattr(Config, "GOOGLE_SUMMARY_MODEL", DEFAULT_GOOGLE_MODEL)

    try:
        logger.debug(f"Requesting summary from Google model '{gemini_model_name}' for item '{item_name or 'Unknown'}'")
        response = google_gemini_client_via_openai_lib.chat.completions.create(
            model=gemini_model_name,
            messages=messages_for_api,
            temperature=0.2, # Low temperature for factual summaries
            max_tokens=150, 
        )
        summary = response.choices[0].message.content.strip() if response.choices and response.choices[0].message.content else None

        if summary:
            logger.info(f"Google summary generated for '{item_name or 'Unknown'}'. Preview: '{summary[:100]}...'")
        else:
            logger.warning(f"Google returned an empty or null summary for '{item_name or 'Unknown'}'. Original text length: {len(plain_text_description)}")
        return summary

    except APIError as e: 
        logger.error(f"Google APIError (via OpenAI lib) during description summarization for '{item_name or 'Unknown'}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling Google for description summarization for '{item_name or 'Unknown'}': {e}", exc_info=True)
        return None

# ── Tool schema (mirrors openai_service, descriptions updated) ───────────────────────────────────
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_local_products",
            "description": ( # MODIFIED
                "Busca en el catálogo de productos de la tienda utilizando una consulta en lenguaje natural. "
                "Esta herramienta es ideal cuando el usuario pregunta por tipos de productos, características específicas, o si hay 'algo como...'. "
                "Devuelve una lista de productos que coinciden semánticamente con la consulta. "
                "Cada producto devuelto incluye nombre, marca, precio, stock, y una descripción lista para el usuario (llm_formatted_description) que ya contiene la información más relevante."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": (
                            "La consulta del usuario que describe el producto o característica deseada. "
                            "Por ejemplo: 'televisor inteligente de 55 pulgadas con buen sonido', 'neveras marca Samsung', 'aires acondicionados de ventana'."
                        ),
                    },
                    "filter_stock": { # Your original parameter
                        "type": "boolean",
                        "description": (
                            "Si es True (predeterminado), solo devuelve productos con stock disponible (>0) " # Clarified based on your original intent
                            "en alguna de sus ubicaciones."
                        ),
                        "default": True,
                    },
                    "warehouse_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Opcional. Lista de almacenes para limitar la búsqueda según la ciudad.",
                    },
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_product_details",
            "description": ( # MODIFIED
                "Obtiene información detallada y actualizada de un producto específico, incluyendo precio exacto y disponibilidad de stock por almacén/sucursal. "
                "Utiliza esta herramienta cuando el usuario pregunta por un producto muy específico (por su código/SKU o ID único si se conoce) o después de que `search_local_products` haya devuelto un producto y el usuario quiera más detalles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_identifier": {
                        "type": "string",
                        "description": "El código de item (SKU) del producto o el ID compuesto (itemCode_warehouseName) del producto en una ubicación específica.",
                    },
                    "identifier_type": {
                        "type": "string",
                        "enum": ["sku", "composite_id"], # MODIFIED from wc_product_id
                        "description": "Especifica si 'product_identifier' es un 'sku' (para buscar en todas las ubicaciones) o un 'composite_id' (para una ubicación específica).",
                    },
                },
                "required": ["product_identifier", "identifier_type"],
            },
        },
    },
]

# ── Helper functions ────────────────────────────────────────────────────────────
def _format_sb_history_for_openai_compatible_api( # Renamed from your _format_sb_history_for_openai
    sb_messages: Optional[List[Dict[str, Any]]], # Using List[Dict[str, Any]] for better type safety
) -> List[Dict[str, Any]]:
    if not sb_messages: return []
    
    openai_messages: List[Dict[str, Any]] = []
    bot_user_id_str = Config.SUPPORT_BOARD_BOT_USER_ID
    if not bot_user_id_str: 
        logger.error("SUPPORT_BOARD_BOT_USER_ID not set for formatting SB history.")
        return []

    for msg in sb_messages:
        sender_id = msg.get("user_id")
        text_content = msg.get("message","").strip()
        attachments = msg.get("attachments")
        image_urls: List[str] = []

        if attachments and isinstance(attachments, list):
            for att in attachments:
                url=att.get("url","")
                if url and (att.get("type","").startswith("image") or any(url.lower().endswith(ext) for ext in [".jpg",".jpeg",".png",".gif",".webp"])):
                    if url.startswith(("http://","https://")): 
                        image_urls.append(url)
                    else: 
                        logger.warning("Skipping possible non-public image URL: %s",url)
        
        if not text_content and not image_urls: continue
        if sender_id is None: 
            logger.warning("Skipping msg with no sender_id: %s", msg.get("id"))
            continue
            
        role="assistant" if str(sender_id)==bot_user_id_str else "user"
        
        # MODIFIED: Content construction for Gemini (which might also support multimodal via OpenAI lib)
        content_list_for_api: List[Dict[str, Any]] = []
        if text_content:
            content_list_for_api.append({"type":"text","text":text_content}) # Gemini expects "text" key for text parts in a list
        
        # Check if the current Gemini model (via OpenAI lib) supports vision
        # This might require checking Google's documentation for specific model capabilities
        # For now, assume if image_urls are present, we attempt to send them.
        # The Gemini API (via OpenAI lib) might ignore them if not supported by the model.
        if image_urls:
            for img_url in image_urls:
                # Gemini's direct API uses "inline_data" or "file_data".
                # When using OpenAI library, it might map "image_url" to the appropriate Gemini format,
                # or it might require a different structure. This needs testing with the specific Gemini model.
                # For now, using OpenAI's image_url structure.
                content_list_for_api.append({"type":"image_url","image_url":{"url":img_url}})
            if not text_content: # If only images, ensure the LLM knows an image was sent.
                 logger.debug("Message for Gemini contains only images.")
                 # It's generally better to let the LLM infer from the image parts if it can.
                 # Adding placeholder text can sometimes confuse it if it *can* see the image.

        if content_list_for_api:
            if len(content_list_for_api) == 1 and content_list_for_api[0]["type"] == "text":
                openai_messages.append({"role":role, "content": content_list_for_api[0]["text"]})
            else:
                openai_messages.append({"role":role, "content": content_list_for_api})
    return openai_messages


def _format_search_results_for_llm(results: Optional[List[Dict[str, Any]]]) -> str:
    # MODIFIED: Return structured JSON
    if results is None:
        return json.dumps({"status": "error", "message": "Ocurrió un error interno al buscar en el catálogo. Intenta más tarde."}, ensure_ascii=False)
    if not results:
        return json.dumps({"status": "not_found", "message": "No se encontraron productos que coincidan con esa descripción."}, ensure_ascii=False)
    try:
        return json.dumps({"status": "success", "products": results}, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Error serialising search results to JSON for LLM: {e}")
        return json.dumps({"status": "error", "message": f"Error al formatear los resultados de búsqueda: {str(e)}"}, ensure_ascii=False)


def _format_live_details_for_llm(details_input: Union[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]], identifier_type: str = "ID") -> str: # Added identifier_type
    # MODIFIED: Return structured JSON and handle list (for SKU multi-location) or dict
    if details_input is None:
        return json.dumps({"status": "error", "message": f"No se pudieron recuperar los detalles en tiempo real para ese producto ({identifier_type})."}, ensure_ascii=False)

    if isinstance(details_input, dict):
        if not details_input: # Empty dict means not found for composite_id
            return json.dumps({"status": "not_found", "message": f"No se encontró ningún producto con el {identifier_type} especificado."}, ensure_ascii=False)
        # Wrap single product in a list for consistent "products" key, or define separate structure
        product_list = [details_input]
        message_if_empty = f"No se encontró ningún producto con el {identifier_type} especificado."
    elif isinstance(details_input, list):
        if not details_input: # Empty list means not found for SKU
            return json.dumps({"status": "not_found", "message": f"No se encontró ningún producto con el {identifier_type} especificado."}, ensure_ascii=False)
        product_list = details_input
        message_if_empty = f"No se encontró ningún producto con el {identifier_type} especificado."
    else:
        logger.error(f"Unexpected type for details_input in _format_live_details_for_llm: {type(details_input)}")
        return json.dumps({"status": "error", "message": "Formato de detalles inesperado."}, ensure_ascii=False)

    # Reconstruct product details for consistent output, prioritizing llm_formatted_description
    formatted_products = []
    for item_details in product_list:
        formatted_products.append({
            "name": item_details.get("item_name", "Producto Desconocido"),
            "item_code": item_details.get("item_code", "N/A"),
            "id": item_details.get("id"), 
            "description": item_details.get("llm_formatted_description") or item_details.get("llm_summarized_description") or item_details.get("plain_text_description_derived", "Descripción no disponible."),
            "brand": item_details.get("brand", "N/A"),
            "category": item_details.get("category", "N/A"),
            "price": item_details.get("price"),
            "stock": item_details.get("stock"),
            "warehouse_name": item_details.get("warehouse_name"),
            "branch_name": item_details.get("branch_name")
        })
    
    if not formatted_products: # Should have been caught by earlier checks, but as a safeguard
         return json.dumps({"status": "not_found", "message": message_if_empty}, ensure_ascii=False)

    if len(formatted_products) == 1:
        return json.dumps({"status": "success", "product": formatted_products[0]}, indent=2, ensure_ascii=False)
    else: # Multiple locations for an SKU
        return json.dumps({"status": "success_multiple_locations", "message": "El producto está disponible en varias ubicaciones.", "locations": formatted_products}, indent=2, ensure_ascii=False)


# ── Main entry point ────────────────────────────────────────────────────────────
def process_new_message_gemini_via_openai_lib( # Your original function name
    sb_conversation_id: str,
    new_user_message: Optional[str], # This is the text of the current user message
    conversation_source: Optional[str],
    sender_user_id: str,
    customer_user_id: str,
    triggering_message_id: Optional[str],
):
    logger.info(
        f"[Namwoo-Gemini] Handling SB Conv {sb_conversation_id} (User: {customer_user_id}, Sender: {sender_user_id}, TriggerMsg: {triggering_message_id})"
    )

    if new_user_message:
        detected_city = conversation_location.detect_city_from_text(new_user_message)
        if detected_city:
            conversation_location.set_conversation_city(sb_conversation_id, detected_city)

    if not GOOGLE_SDK_AVAILABLE or not google_gemini_client_via_openai_lib:
        logger.error("[Namwoo-Gemini] Google Gemini client (via OpenAI lib) not available.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id,
            message_text="Disculpa, el servicio de IA (Google) no está disponible en este momento.",
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=None, triggering_message_id=triggering_message_id,
        )
        return

    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)
    if conversation_data is None or not conversation_data.get("messages"):
        logger.error(f"[Namwoo-Gemini] Failed to fetch conversation data or no messages for Conv {sb_conversation_id}.")
        support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text="Lo siento, no pude acceder a los detalles de esta conversación.", source=conversation_source, target_user_id=customer_user_id, conversation_details=None, triggering_message_id=triggering_message_id); return
    
    try:
        api_history = _format_sb_history_for_openai_compatible_api(conversation_data.get("messages", []))
        if not api_history :
            logger.warning(f"[Namwoo-Gemini] Formatted history is empty for Conv {sb_conversation_id}, though SB history exists. This might indicate an issue with history formatting or only bot messages.")
            # If new_user_message is available, use it as the sole user message
            if new_user_message:
                logger.info(f"[Namwoo-Gemini] Proceeding with only the new user message for Conv {sb_conversation_id}.")
                api_history = [{"role": "user", "content": new_user_message}]
            else:
                logger.error(f"[Namwoo-Gemini] Formatted history is empty and no new_user_message for Conv {sb_conversation_id}. Aborting.")
                support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text="Lo siento, no pude procesar los mensajes anteriores.", source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id); return
    except Exception as e:
        logger.exception(f"[Namwoo-Gemini] Error formatting SB history for Conv {sb_conversation_id}: {e}")
        support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas al procesar el historial.", source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id); return

    gemini_model_name = DEFAULT_GOOGLE_MODEL # Using the constant defined above

    messages_for_api = [{"role": "system", "content": Config.SYSTEM_PROMPT}] + api_history
    
    # Your original history trimming logic
    current_max_history = getattr(Config, "MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES) # Use the constant or Config
    if len(messages_for_api) > (current_max_history + 1): # +1 for system prompt
        messages_for_api = [messages_for_api[0]] + messages_for_api[-(current_max_history):]


    final_assistant_response: Optional[str] = None
    try:
        tool_call_count = 0
        current_call_params: Dict[str, Any] = {
            "model": gemini_model_name,
            "messages": messages_for_api,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_GOOGLE_TEMPERATURE, # Use configured temperature
            # Tools are added on the first attempt only usually, or if LLM is expected to chain them
        }

        while tool_call_count <= TOOL_CALL_RETRY_LIMIT: 
            # Add tools only if LLM might need them (typically first call, or if previous was tool call)
            # The OpenAI library handles this well if `tool_choice` is "auto"
            if tool_call_count == 0 or (current_call_params["messages"][-1].get("role") == "assistant" and current_call_params["messages"][-1].get("tool_calls")):
                api_call_params["tools"] = tools_schema
                api_call_params["tool_choice"] = "auto" 
            else: # If last message was from 'tool', don't send tools again, let LLM generate response
                api_call_params.pop("tools", None)
                api_call_params.pop("tool_choice", None)

            logger.debug(f"[Namwoo-Gemini] Sending request to Gemini (Attempt {tool_call_count + 1}): Model={gemini_model_name}, Tools Sent={ 'tools' in api_call_params }")

            response = google_gemini_client_via_openai_lib.chat.completions.create(**api_call_params)
            response_msg_obj = response.choices[0].message # This is an ChatCompletionMessage object
            
            # Append the assistant's response (which might contain tool calls or content) to message history
            # model_dump() converts Pydantic model to dict, exclude_none to keep it clean
            messages_for_api.append(response_msg_obj.model_dump(exclude_none=True))
            
            tool_calls = response_msg_obj.tool_calls

            if response.usage:
                logger.info(
                    "[Namwoo-Gemini] Tokens usage (Conv %s, Attempt %d) - Prompt: %d, Completion: %d, Total: %d",
                    sb_conversation_id, tool_call_count + 1,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            if not tool_calls: # LLM decided not to call a tool OR has finished tool sequence
                final_assistant_response = response_msg_obj.content
                logger.debug(f"[Namwoo-Gemini] Received final/intermediate response from Gemini: '{str(final_assistant_response)[:200]}...'")
                break # Exit tool call loop

            # --- Execute tool requests ---
            tool_outputs_for_api: List[Dict[str, str]] = []
            for tc in tool_calls:
                fname = tc.function.name
                tool_call_id = tc.id
                logger.info("[Namwoo-Gemini] Gemini requested tool: %s with ID %s, Args: %s", fname, tool_call_id, tc.function.arguments)
                
                tool_content_str: str
                try:
                    args = json.loads(tc.function.arguments) # Arguments are a JSON string

                    if fname == "search_local_products":
                        query = args.get("query_text")
                        filter_stock_flag = args.get("filter_stock", True) # Default as per schema
                        warehouse_names_arg = args.get("warehouse_names")
                        if not warehouse_names_arg:
                            warehouse_names_arg = conversation_location.get_city_warehouses(sb_conversation_id)
                        if query:
                            search_results_list = product_service.search_local_products(
                                query_text=query,
                                limit=getattr(Config, "PRODUCT_SEARCH_LIMIT", 10), # Use configured limit
                                filter_stock=filter_stock_flag,
                                warehouse_names=warehouse_names_arg,
                            )
                            tool_content_str = _format_search_results_for_llm(search_results_list)
                        else:
                            tool_content_str = json.dumps({"status": "error", "message": "Argumento 'query_text' es requerido para search_local_products."}, ensure_ascii=False)
                    
                    elif fname == "get_live_product_details":
                        identifier = args.get("product_identifier")
                        id_type = args.get("identifier_type") # Changed 'identifier_type' to match parameter name
                        if not identifier or not id_type:
                            tool_content_str = json.dumps({"status": "error", "message": "Argumentos 'product_identifier' e 'identifier_type' son requeridos."}, ensure_ascii=False)
                        else:
                            details_result: Union[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]] = None
                            if id_type == "sku":
                                details_result = product_service.get_live_product_details_by_sku(item_code_query=identifier)
                            elif id_type == "composite_id": 
                                details_result = product_service.get_live_product_details_by_id(composite_id=identifier)
                            else: # Unknown id_type
                                tool_content_str = json.dumps({"status": "error", "message": f"Tipo de identificador '{id_type}' no soportado. Use 'sku' o 'composite_id'."}, ensure_ascii=False)
                                # Skip _format_live_details_for_llm if id_type is invalid
                                tool_outputs_for_api.append({"tool_call_id": tool_call_id, "role": "tool", "name": fname, "content": tool_content_str})
                                continue # Next tool call

                            tool_content_str = _format_live_details_for_llm(details_result, identifier_type=id_type)
                    else:
                        logger.warning(f"[Namwoo-Gemini] Unknown tool requested: {fname}")
                        tool_content_str = json.dumps({"status": "error", "message": f"Herramienta desconocida '{fname}'."}, ensure_ascii=False)

                except json.JSONDecodeError:
                    logger.error("[Namwoo-Gemini] Invalid JSON arguments for tool %s: %s", fname, tc.function.arguments)
                    tool_content_str = json.dumps({"status": "error", "message": f"Argumentos JSON inválidos para la herramienta {fname}."}, ensure_ascii=False)
                except Exception as e_tool:
                    logger.exception(f"[Namwoo-Gemini] Error executing tool {fname}: {e_tool}")
                    tool_content_str = json.dumps({"status": "error", "message": f"Error interno ejecutando la herramienta {fname}: {str(e_tool)}"}, ensure_ascii=False)
                
                tool_outputs_for_api.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": fname,
                    "content": tool_content_str, # This is now a JSON string
                })

            messages_for_api.extend(tool_outputs_for_api)
            tool_call_count += 1
            
            if tool_call_count > TOOL_CALL_RETRY_LIMIT: # Check if limit exceeded
                logger.warning(f"[Namwoo-Gemini] Tool call retry limit ({TOOL_CALL_RETRY_LIMIT}) reached for conv {sb_conversation_id}. Will attempt to generate final response without tools.")
                # Prepare for a final response generation call by removing tool parameters
                api_call_params.pop("tools", None)
                api_call_params.pop("tool_choice", None)
                # The loop will make one more call due to <= which should be for final response generation
                # If that call also results in tool_calls (unlikely if tools aren't offered), it will exit.
                # If it generates content, final_assistant_response will be set.

        # After loop, if final_assistant_response is still None, it means the last attempt (even after tool retries) didn't yield content.
        if final_assistant_response is None:
            if tool_calls: # LLM ended on a tool_call even after retries.
                logger.warning(f"[Namwoo-Gemini] LLM still attempting tool calls after retry limit for Conv {sb_conversation_id}. Sending fallback.")
                final_assistant_response = ("Parece que estoy teniendo dificultades para completar tu solicitud con mis herramientas. "
                                            "¿Podrías intentar preguntarme de otra manera?")
            else: # Should have content if no tool_calls on last iteration. If not, it's an issue.
                 logger.error(f"[Namwoo-Gemini] No final response generated and no tool calls on last attempt for Conv {sb_conversation_id}. This is unexpected. Sending fallback.")
                 final_assistant_response = "Lo siento, no pude procesar tu última solicitud completamente. ¿Podrías intentarlo de nuevo?"


    except RateLimitError:
        logger.warning("[Namwoo-Gemini] RateLimitError for conv %s", sb_conversation_id)
        final_assistant_response = "El servicio de IA (Google) está experimentando un alto volumen de solicitudes. Por favor, intenta más tarde."
    except APITimeoutError:
        logger.warning("[Namwoo-Gemini] APITimeoutError for conv %s", sb_conversation_id)
        final_assistant_response = "No pude obtener respuesta del servicio de IA (Google) a tiempo. Por favor, intenta más tarde."
    except BadRequestError as br_err: 
        err_body_str = str(getattr(br_err, "body", br_err)).lower()
        if "user location" in err_body_str and "not supported" in err_body_str:
            logger.warning("[Namwoo-Gemini] User location not supported for conv %s.", sb_conversation_id)
            final_assistant_response = "Lo siento, tu ubicación actual no es compatible con el servicio de IA de Google en este momento."
        elif "image" in err_body_str or "url" in err_body_str:
            logger.warning("[Namwoo-Gemini] Invalid image error for conv %s: %s", sb_conversation_id, err_body_str)
            final_assistant_response = "Lo siento, parece que una imagen proporcionada no es válida o accesible para Google."
        else:
            logger.warning("[Namwoo-Gemini] BadRequestError for conv %s: %s", sb_conversation_id, br_err, exc_info=True)
            final_assistant_response = "Lo siento, hubo un problema con el formato de nuestra conversación o con una imagen proporcionada."
    except APIError as api_err: 
        status_code = getattr(api_err, "status_code", "N/A")
        logger.error(f"[Namwoo-Gemini] APIError (Status: {status_code}) for conv {sb_conversation_id}: {api_err}", exc_info=True)
        final_assistant_response = f"Hubo un error general ({status_code}) con el servicio de IA de Google. Por favor, inténtalo más tarde."
    except Exception as e: 
        logger.exception(f"[Namwoo-Gemini] Unexpected error during Gemini interaction for conv {sb_conversation_id}: {e}")
        final_assistant_response = "Ocurrió un error inesperado al procesar tu solicitud con Google AI. Por favor, intenta de nuevo."

    if final_assistant_response:
        logger.info(f"[Namwoo-Gemini] Sending final response to SB for conv {sb_conversation_id}: '{str(final_assistant_response)[:100]}...'")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text=str(final_assistant_response),
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=conversation_data, triggering_message_id=triggering_message_id,
        )
    else: 
        logger.error("[Namwoo-Gemini] No final assistant response was generated for conv %s after all attempts. Sending generic fallback.", sb_conversation_id)
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, no pude generar una respuesta en este momento.",
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=conversation_data, triggering_message_id=triggering_message_id,
        )

process_new_message_gemini = process_new_message_gemini_via_openai_lib