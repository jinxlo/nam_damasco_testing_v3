# NAMWOO/services/openai_service.py
# -*- coding: utf-8 -*-
import logging
import json
import time # Keep time if used by retry logic within embedding_utils
from typing import List, Dict, Optional, Tuple, Union, Any
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError
from flask import current_app # For accessing app config like OPENAI_EMBEDDING_MODEL

# Import local services and utils
from . import product_service
from . import support_board_service
from ..config import Config # For SYSTEM_PROMPT, MAX_HISTORY_MESSAGES etc.
from ..utils import embedding_utils
from ..utils import conversation_location


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialise OpenAI client for Chat Completions
# This client instance is primarily for chat. Embeddings can use a fresh call
# or this client if preferred, but embedding_utils handles its own client init.
_chat_client: Optional[OpenAI] = None
try:
    openai_api_key = Config.OPENAI_API_KEY
    if openai_api_key:
        # Use configured timeout if available, otherwise default to 60.0
        timeout_seconds = getattr(Config, 'OPENAI_REQUEST_TIMEOUT', 60.0)
        _chat_client = OpenAI(api_key=openai_api_key, timeout=timeout_seconds)
        logger.info(f"OpenAI client initialized for Chat Completions service with timeout: {timeout_seconds}s.")
    else:
        _chat_client = None
        logger.error(
            "OpenAI API key not configured during initial load. "
            "Chat functionality will fail."
        )
except Exception as e: 
    logger.exception(f"Failed to initialize OpenAI client for chat during initial load: {e}")
    _chat_client = None

# ---------------------------------------------------------------------------
# Constants
MAX_HISTORY_MESSAGES = Config.MAX_HISTORY_MESSAGES 
TOOL_CALL_RETRY_LIMIT = 2 
DEFAULT_OPENAI_MODEL = getattr(Config, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = getattr(Config, "OPENAI_MAX_TOKENS", 1024)
DEFAULT_OPENAI_TEMPERATURE = getattr(Config, "OPENAI_TEMPERATURE", 0.7)

# ---------------------------------------------------------------------------
# Embedding Generation Function
# ---------------------------------------------------------------------------
def generate_product_embedding(text_to_embed: str) -> Optional[List[float]]:
    """
    Generates an embedding for the given product text using the configured
    OpenAI embedding model via embedding_utils.
    """
    if not text_to_embed or not isinstance(text_to_embed, str):
        logger.warning("openai_service.generate_product_embedding: No valid text provided.")
        return None
    embedding_model_name = Config.OPENAI_EMBEDDING_MODEL
    if not embedding_model_name:
        logger.error("openai_service.generate_product_embedding: OPENAI_EMBEDDING_MODEL not configured in Config.")
        return None
    logger.debug(f"Requesting embedding for text: '{text_to_embed[:100]}...' using model: {embedding_model_name}")
    embedding_vector = embedding_utils.get_embedding(
        text=text_to_embed,
        model=embedding_model_name
    )
    if embedding_vector is None:
        logger.error(f"openai_service.generate_product_embedding: Failed to get embedding from embedding_utils for text: '{text_to_embed[:100]}...'")
        return None
    logger.info(f"Successfully generated embedding for text (first 100 chars): '{text_to_embed[:100]}...'")
    return embedding_vector

# ---------------------------------------------------------------------------
# FUNCTION FOR PRODUCT DESCRIPTION SUMMARIZATION (using OpenAI)
# ---------------------------------------------------------------------------
def get_openai_product_summary(
    plain_text_description: str,
    item_name: Optional[str] = None
) -> Optional[str]:
    """
    Generates a concise, factual summary for a product's plain text description.
    """
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized. Cannot summarize description with OpenAI.")
        return None
    if not plain_text_description or not plain_text_description.strip():
        logger.debug("OpenAI summarizer: No plain text description provided to summarize.")
        return None
    prompt_context_parts = []
    if item_name:
        prompt_context_parts.append(f"Nombre del Producto: {item_name}")
    prompt_context_parts.append(f"Descripción Original (texto plano):\n{plain_text_description}")
    prompt_context = "\n".join(prompt_context_parts)
    system_prompt = (
        "Eres un redactor experto en comercio electrónico. Resume la siguiente descripción de producto. "
        "El resumen debe ser conciso (objetivo: 50-75 palabras, 2-3 frases clave), resaltar los principales beneficios y características, y ser factual. "
        "Evita la jerga de marketing, la repetición y frases como 'este producto es'. "
        "La salida debe ser texto plano adecuado para una base de datos de productos y un asistente de IA. "
        "No incluyas etiquetas HTML."
    )
    max_input_chars_for_summary = 3000
    if len(prompt_context) > max_input_chars_for_summary:
        cutoff_point = prompt_context.rfind('.', 0, max_input_chars_for_summary)
        if cutoff_point == -1: cutoff_point = max_input_chars_for_summary
        prompt_context = prompt_context[:cutoff_point] + " [DESCRIPCIÓN TRUNCADA]"
        logger.warning(f"OpenAI summarizer: Description for '{item_name or 'Unknown'}' was truncated for prompt construction.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Por favor, resume la siguiente información del producto:\n\n{prompt_context}"}
    ]
    try:
        summarization_model = getattr(Config, "OPENAI_SUMMARY_MODEL", DEFAULT_OPENAI_MODEL)
        logger.debug(f"Requesting summary from OpenAI model '{summarization_model}' for item '{item_name or 'Unknown'}'")
        completion = _chat_client.chat.completions.create(
            model=summarization_model,
            messages=messages,
            temperature=0.2,
            max_tokens=150,
            n=1,
            stop=None,
        )
        summary = completion.choices[0].message.content.strip() if completion.choices and completion.choices[0].message.content else None
        if summary:
            logger.info(f"OpenAI summary generated for '{item_name or 'Unknown'}'. Preview: '{summary[:100]}...'")
        else:
            logger.warning(f"OpenAI returned an empty or null summary for '{item_name or 'Unknown'}'. Original text length: {len(plain_text_description)}")
        return summary
    except APIError as e:
        logger.error(f"OpenAI APIError during description summarization for '{item_name or 'Unknown'}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI for description summarization for '{item_name or 'Unknown'}': {e}", exc_info=True)
        return None


def extract_customer_info_via_llm(message_text: str) -> Optional[Dict[str, Any]]:
    """Extract structured customer info from a plain text message using OpenAI."""
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized. Cannot extract customer info.")
        return None

    system_prompt = (
        "Extrae la siguiente información del mensaje del cliente. "
        "Devuelve solo JSON válido con las claves: full_name, cedula, telefono, "
        "correo, direccion, productos y total. Si falta algún campo, usa null. "
        "No incluyas explicaciones ni comentarios."
    )
    user_prompt = f"Mensaje del cliente:\n\"\"\"{message_text}\"\"\""

    try:
        response = _chat_client.chat.completions.create(
            model=current_app.config.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("OpenAI returned empty content when extracting customer info.")
            return None
        return json.loads(content)
    except json.JSONDecodeError as jde:
        logger.error(f"JSON decoding error extracting customer info via OpenAI: {jde}")
        return None
    except Exception as e:
        logger.exception(f"Error extracting customer info via OpenAI: {e}")
        return None

# ===========================================================================
# LLM TOOL IMPLEMENTATION FUNCTIONS
# ===========================================================================
def _tool_send_whatsapp_order_summary_template(
    customer_platform_user_id: str,
    conversation_id: str,
    template_variables: List[str]
) -> str:
    """Send WhatsApp order summary template via Support Board."""
    logger.info(
        f"Executing _tool_send_whatsapp_order_summary_template for user {customer_platform_user_id} conv {conversation_id}"
    )
    if not customer_platform_user_id or not conversation_id or not template_variables:
        logger.error("send_whatsapp_order_summary_template called with missing data.")
        return "ERROR_MISSING_DATA_FOR_TEMPLATE: Faltan datos requeridos para enviar la plantilla."

    try:
        result = support_board_service.send_order_confirmation_template(
            user_id=customer_platform_user_id,
            conversation_id=conversation_id,
            variables=template_variables,
        )
        if result is not None:
            logger.info(
                f"Order confirmation template sent via SB for conv {conversation_id} to user {customer_platform_user_id}"
            )
            return "OK_TEMPLATE_SENT"
        else:
            logger.error(
                f"Support Board failed to send template for conv {conversation_id} to user {customer_platform_user_id}"
            )
            return "ERROR_SENDING_TEMPLATE: No se pudo enviar la plantilla de resumen de pedido."
    except Exception as exc:
        logger.exception(f"Exception sending WhatsApp template for conv {conversation_id}: {exc}")
        return "ERROR_SENDING_TEMPLATE: Hubo un problema interno al enviar la plantilla."

# ---------------------------------------------------------------------------
# Tool definitions for OpenAI
# ---------------------------------------------------------------------------
tools_schema = [
    { 
        "type": "function",
        "function": {
            "name": "search_local_products",
            "description": ( 
                "Busca en el catálogo de productos de la tienda Damasco usando una consulta en lenguaje natural. "
                "Ideal cuando el usuario pregunta por tipos de productos o características. "
                "Devuelve una lista de productos coincidentes con nombre, marca, precio (USD), precio en Bolívares (`priceBolivar`), especificaciones (`specifitacion`), y una descripción lista para el usuario (`llm_formatted_description`)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": (
                            "Consulta del usuario describiendo el producto. "
                            "Ej: 'televisor inteligente de 55 pulgadas', 'neveras Samsung'."
                        ),
                    },
                    "filter_stock": {
                        "type": "boolean",
                        "description": (
                            "Opcional. Si es true (defecto), filtra solo productos con stock. "
                            "False si se quiere verificar si un producto existe en catálogo sin importar stock."
                        ),
                        "default": True,
                    },
                    "warehouse_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Opcional y generalmente no se debe usar. El sistema backend lo infiere de la ciudad del usuario."
                        ),
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
            "description": ( 
                "Obtiene información detallada y actualizada de un producto específico de Damasco, incluyendo precio (USD), precio en Bolívares (`priceBolivar`), especificaciones (`specifitacion`), y stock por sucursal. "
                "Usar cuando el usuario pregunta por un producto específico (por SKU/código) o después de `search_local_products` si quiere más detalles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_identifier": {
                        "type": "string",
                        "description": "El código de item (SKU) del producto o el ID compuesto (itemCode_warehouseName).",
                    },
                    "identifier_type": {
                        "type": "string",
                        "enum": ["sku", "composite_id"], 
                        "description": "Especifica si 'product_identifier' es 'sku' (para todas las ubicaciones) o 'composite_id' (para una ubicación específica).",
                    },
                },
                "required": ["product_identifier", "identifier_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_order_summary_template",
            "description": "Envía la plantilla de resumen de pedido por WhatsApp al cliente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_platform_user_id": {
                        "type": "string",
                        "description": "ID del usuario en la plataforma de mensajería (número de teléfono o ID interno)."
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "ID de la conversación donde se enviará la plantilla."
                    },
                    "template_variables": {
                        "type": "array",
                        "description": "Lista de 8 cadenas con nombre, apellido, cédula, teléfono, correo, dirección, descripción del producto y total.",
                        "items": {"type": "string"},
                        "minItems": 8,
                        "maxItems": 8
                    }
                },
                "required": ["customer_platform_user_id", "conversation_id", "template_variables"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Helper: format Support‑Board history for OpenAI
# ---------------------------------------------------------------------------
def _format_sb_history_for_openai(
    sb_messages: Optional[List[Dict[str, Any]]], 
) -> List[Dict[str, Any]]: 
    if not sb_messages:
        return []
    openai_messages: List[Dict[str, Any]] = []
    bot_user_id_str = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id_str:
        logger.error("Cannot format SB history: SUPPORT_BOARD_DM_BOT_USER_ID is not configured.")
        return []
    for msg in sb_messages:
        sender_id = msg.get("user_id")
        text_content = msg.get("message", "").strip()
        attachments = msg.get("attachments")
        image_urls: List[str] = []
        if attachments and isinstance(attachments, list):
            for att in attachments:
                if (isinstance(att, dict) and att.get("url") and 
                    (att.get("type", "").startswith("image") or 
                     any(att["url"].lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]))):
                    url = att["url"]
                    if url.startswith(("http://", "https://")):
                        image_urls.append(url)
                    else:
                        logger.warning("Skipping possible non‑public URL for attachment %s", url)
        if not text_content and not image_urls:
            continue
        if sender_id is None:
            continue
        role = "assistant" if str(sender_id) == bot_user_id_str else "user"
        content_list_for_openai: List[Dict[str, Any]] = []
        if text_content:
            content_list_for_openai.append({"type": "text", "text": text_content})
        current_openai_model = getattr(Config, "OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL)
        vision_capable_models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"] 
        if image_urls and current_openai_model in vision_capable_models:
            for img_url in image_urls:
                content_list_for_openai.append({"type": "image_url", "image_url": {"url": img_url}})
        elif image_urls: 
            logger.warning(f"Image URLs found but current model {current_openai_model} may not support vision. Images not explicitly sent.")
            if not text_content:
                 content_list_for_openai.append({"type": "text", "text": "[Usuario envió una imagen]"})
        if content_list_for_openai:
            if len(content_list_for_openai) == 1 and content_list_for_openai[0]["type"] == "text":
                openai_messages.append({"role": role, "content": content_list_for_openai[0]["text"]})
            else:
                openai_messages.append({"role": role, "content": content_list_for_openai})
    return openai_messages

# ==============================================================================
# <<< START OF MODIFIED SECTION >>>
# ==============================================================================
# Helper: format search results
def _format_search_results_for_llm(results: Optional[Dict[str, Any]]) -> str:
    """
    Formats the structured product search results (from product_service) into a
    JSON string to be sent back to the LLM. It now expects a dictionary
    containing the grouped products.
    """
    # Handle cases where the service might return None (e.g., database error)
    if results is None:
        return json.dumps({
            "status": "error",
            "message": "Lo siento, ocurrió un error interno al buscar en el catálogo. Por favor, intenta de nuevo más tarde."
        }, ensure_ascii=False)

    # Handle cases where the search was successful but found no items.
    # The 'products_grouped' key will be present but the list will be empty.
    if not results.get("products_grouped"):
        return json.dumps({
            "status": "not_found",
            "message": "Lo siento, no pude encontrar productos que coincidan con esa descripción en nuestro catálogo actual."
        }, ensure_ascii=False)
    
    # If we have results, serialize the entire dictionary structure as-is.
    # The LLM is now trained (via the system prompt) to understand this new structure.
    try:
        # Use indent=2 for readability during debugging, but for production,
        # you can remove it to save a few tokens.
        return json.dumps(results, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as err:
        logger.error(f"JSON serialisation error for new grouped search results: {err}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Lo siento, hubo un problema al formatear los resultados de la búsqueda."
        }, ensure_ascii=False)
# ==============================================================================
# <<< END OF MODIFIED SECTION >>>
# ==============================================================================

# ---------------------------------------------------------------------------
# Helper: live‑detail formatter
# ---------------------------------------------------------------------------
def _format_live_details_for_llm(details: Optional[Dict[str, Any]], identifier_type: str = "ID") -> str: 
    """
    Formats detailed product information into a JSON string for the LLM.
    """
    if details is None: 
        return json.dumps({"status": "error", "message": f"Lo siento, no pude recuperar los detalles en tiempo real para ese producto ({identifier_type})."}, ensure_ascii=False)
    if not details: 
         return json.dumps({"status": "not_found", "message": f"No se encontraron detalles para el producto con el {identifier_type} proporcionado."}, ensure_ascii=False)
    
    product_info = {
        "name": details.get("item_name", "Producto Desconocido"),
        "item_code": details.get("item_code", "N/A"),
        "id": details.get("id"), # This is the composite ID
        "description": details.get("llm_formatted_description") or \
                       details.get("llm_summarized_description") or \
                       details.get("plain_text_description_derived", "Descripción no disponible."),
        "specifitacion": details.get("specifitacion"),
        "brand": details.get("brand", "N/A"),
        "category": details.get("category", "N/A"),
        "price": details.get("price"), # Assumed to be USD price
        "priceBolivar": details.get("priceBolivar"),
        "stock": details.get("stock"),
        "warehouse_name": details.get("warehouse_name"),
        "branch_name": details.get("branch_name")
    }

    return json.dumps({"status": "success", "product": product_info}, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main processing entry‑point
# ---------------------------------------------------------------------------
def process_new_message(
    sb_conversation_id: str,
    new_user_message: Optional[str], 
    conversation_source: Optional[str],
    sender_user_id: str,
    customer_user_id: str,
    triggering_message_id: Optional[str],
) -> None:
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized. Cannot process message.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Disculpa, el servicio de IA no está disponible en este momento.",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=None, triggering_message_id=triggering_message_id,
        )
        return

    logger.info(
        "Processing message for SB Conv %s (trigger_user=%s, customer=%s, source=%s, trig_msg_id=%s)",
        sb_conversation_id, sender_user_id, customer_user_id, conversation_source, triggering_message_id,
    )

    if new_user_message:
        detected_city = conversation_location.detect_city_from_text(new_user_message)
        if detected_city:
            conversation_location.set_conversation_city(sb_conversation_id, detected_city)

    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)
    if conversation_data is None or not conversation_data.get("messages"): 
        logger.error(f"Failed to fetch conversation data or no messages found for SB Conv {sb_conversation_id}. Aborting.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas para acceder al historial de esta conversación. ¿Podrías intentarlo de nuevo?",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=None, triggering_message_id=triggering_message_id
        )
        return

    sb_history_list = conversation_data.get("messages", []) 
    try:
        openai_history = _format_sb_history_for_openai(sb_history_list)
    except Exception as err:
        logger.exception(f"Error formatting SB history for Conv {sb_conversation_id}: {err}")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas al procesar el historial de la conversación.",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id
        )
        return

    if not openai_history: 
        if new_user_message and not sb_history_list: 
            logger.info(f"Formatted OpenAI history is empty for Conv {sb_conversation_id}, using new_user_message as initial prompt.")
            openai_history = [{"role": "user", "content": new_user_message}]
        else:
            logger.error(f"Formatted OpenAI history is empty for Conv {sb_conversation_id}, and no new message or history. Aborting.")
            support_board_service.send_reply_to_channel(
                conversation_id=sb_conversation_id, message_text="Lo siento, no pude procesar los mensajes anteriores adecuadamente.",
                source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id
            )
            return

    system_prompt_content = Config.SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt_content}] + openai_history

    max_hist_current = getattr(Config, "MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES)
    if len(messages) > (max_hist_current + 1 ): 
        messages = [messages[0]] + messages[-(max_hist_current):]

    final_assistant_response: Optional[str] = None
    try:
        tool_call_count = 0
        while tool_call_count <= TOOL_CALL_RETRY_LIMIT: 
            openai_model = current_app.config.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL)
            max_tokens = current_app.config.get("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS)
            temperature = current_app.config.get("OPENAI_TEMPERATURE", DEFAULT_OPENAI_TEMPERATURE)

            call_params: Dict[str, Any] = {
                "model": openai_model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            }
            
            if tool_call_count < TOOL_CALL_RETRY_LIMIT and not (messages[-1].get("role") == "tool"):
                 call_params["tools"] = tools_schema
                 call_params["tool_choice"] = "auto"
            else: 
                call_params.pop("tools", None)
                call_params.pop("tool_choice", None)

            logger.debug(f"OpenAI API call attempt {tool_call_count + 1} for Conv {sb_conversation_id}. Tools offered: {'tools' in call_params}")
            response = _chat_client.chat.completions.create(**call_params)
            response_message = response.choices[0].message 

            if response.usage:
                 logger.info(f"OpenAI Tokens (Conv {sb_conversation_id}, Attempt {tool_call_count+1}): Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")

            messages.append(response_message.model_dump(exclude_none=True))
            tool_calls = response_message.tool_calls

            if not tool_calls:
                final_assistant_response = response_message.content
                logger.info(f"OpenAI response (no tool call this turn) for Conv {sb_conversation_id}: '{str(final_assistant_response)[:200]}...'")
                break 

            tool_outputs_for_llm: List[Dict[str, str]] = [] 
            for tc in tool_calls:
                fn_name = tc.function.name
                tool_call_id = tc.id 
                args_str = ""
                try:
                    args_str = tc.function.arguments
                    args = json.loads(args_str)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSONDecodeError for tool {fn_name} args (Conv {sb_conversation_id}): {args_str}. Error: {json_err}")
                    args = {} 
                    output_txt = json.dumps({"status": "error", "message": f"Error: Argumentos para {fn_name} no son JSON válido: {args_str}"}, ensure_ascii=False)
                else:
                    # This line is not needed, default error message is handled in the `except` block below
                    # output_txt = json.dumps({"status":"error", "message":f"Error: Falló la ejecución de la herramienta {fn_name}."}, ensure_ascii=False)
                    pass
                
                logger.info(f"OpenAI requested tool call: {fn_name} with args: {args} for Conv {sb_conversation_id}")
                
                try:
                    if fn_name == "search_local_products":
                        query = args.get("query_text")
                        filter_stock_flag = args.get("filter_stock", True)
                        
                        warehouse_names_arg = conversation_location.get_city_warehouses(sb_conversation_id)
                        if args.get("warehouse_names"):
                            logger.warning(
                                f"LLM attempted to provide 'warehouse_names': {args.get('warehouse_names')}. "
                                f"Ignoring and using backend-derived warehouses for city: {conversation_location.get_conversation_city(sb_conversation_id)}"
                            )

                        if query:
                            search_res = product_service.search_local_products(
                                query_text=query,
                                filter_stock=filter_stock_flag,
                                warehouse_names=warehouse_names_arg,
                            )
                            output_txt = _format_search_results_for_llm(search_res)
                        else:
                            output_txt = json.dumps({"status": "error", "message": "Error: 'query_text' es requerido para search_local_products."}, ensure_ascii=False)
                    
                    elif fn_name == "get_live_product_details":
                        ident = args.get("product_identifier")
                        id_type = args.get("identifier_type")
                        if ident and id_type:
                            details_result = None 
                            if id_type == "sku":
                                details_result = product_service.get_live_product_details_by_sku(item_code_query=ident)
                                output_txt = _format_live_details_for_llm(details_result, identifier_type="SKU")
                            elif id_type == "composite_id":
                                details_result = product_service.get_live_product_details_by_id(composite_id=ident)
                                output_txt = _format_live_details_for_llm(details_result, identifier_type="ID Compuesto")
                            else:
                                output_txt = json.dumps({"status": "error", "message": f"Error: Tipo de identificador '{id_type}' no soportado. Use 'sku' o 'composite_id'."}, ensure_ascii=False)
                        else:
                            output_txt = json.dumps({"status": "error", "message": "Error: Faltan 'product_identifier' o 'identifier_type' para get_live_product_details."}, ensure_ascii=False)
                    
                    elif fn_name == "send_whatsapp_order_summary_template":
                        cust_id_arg = args.get("customer_platform_user_id") or customer_user_id
                        conv_id_arg = args.get("conversation_id") or sb_conversation_id
                        template_vars_arg = args.get("template_variables")
                        output_txt = _tool_send_whatsapp_order_summary_template(
                            customer_platform_user_id=cust_id_arg,
                            conversation_id=conv_id_arg,
                            template_variables=template_vars_arg,
                        )
                    else:
                        output_txt = json.dumps({"status": "error", "message": f"Error: Herramienta desconocida '{fn_name}'."}, ensure_ascii=False)
                        logger.warning(f"LLM called unknown tool: {fn_name} in Conv {sb_conversation_id}")

                except Exception as tool_exec_err: 
                    logger.exception(f"Tool execution error for {fn_name} (Conv {sb_conversation_id}): {tool_exec_err}")
                    output_txt = json.dumps({"status": "error", "message": f"Error interno al ejecutar la herramienta {fn_name}: {str(tool_exec_err)}"}, ensure_ascii=False)
                
                tool_outputs_for_llm.append({ 
                    "tool_call_id": tool_call_id, "role": "tool", "name": fn_name, "content": output_txt,
                })

            messages.extend(tool_outputs_for_llm) 
            tool_call_count += 1
            
            if tool_call_count > TOOL_CALL_RETRY_LIMIT and not final_assistant_response: 
                logger.warning(f"Tool call retry limit ({TOOL_CALL_RETRY_LIMIT}) strictly exceeded for Conv {sb_conversation_id}. Breaking loop.")
                break

    except RateLimitError:
        logger.warning(f"OpenAI RateLimitError for Conv {sb_conversation_id}")
        final_assistant_response = ("Estoy experimentando un alto volumen de solicitudes. "
                                    "Por favor, espera un momento y vuelve a intentarlo.")
    except APITimeoutError:
        logger.warning(f"OpenAI APITimeoutError for Conv {sb_conversation_id}")
        final_assistant_response = "No pude obtener respuesta del servicio de IA (OpenAI) a tiempo. Por favor, intenta más tarde."
    except BadRequestError as bre:
        logger.error(f"OpenAI BadRequestError for Conv {sb_conversation_id}: {bre}", exc_info=True)
        if "image_url" in str(bre).lower() and "invalid" in str(bre).lower():
            final_assistant_response = ("Parece que una de las imágenes en nuestra conversación no pudo ser procesada. "
                                        "¿Podrías intentarlo sin la imagen o con una diferente?")
        else:
            error_code = getattr(bre, 'code', None)
            if error_code == 'invalid_request_error' and 'tools' in str(bre).lower():
                 final_assistant_response = ("Lo siento, hubo un problema con la forma en que intenté usar mis herramientas internas. "
                                            "Intentaré de nuevo o puedes reformular tu solicitud.")
                 logger.error(f"BadRequestError possibly related to tool usage: {bre.message}")
            else:
                final_assistant_response = ("Lo siento, hubo un problema con el formato de nuestra conversación. "
                                            "Por favor, revisa si enviaste alguna imagen que no sea válida o reformula tu pregunta.")
    except APIError as apie:
        logger.error(f"OpenAI APIError for Conv {sb_conversation_id} (Status: {apie.status_code}): {apie}", exc_info=True)
        final_assistant_response = (f"Hubo un error ({apie.status_code}) con el servicio de IA. Por favor, inténtalo más tarde.")
    except Exception as e:
        logger.exception(f"Unexpected OpenAI interaction error for Conv {sb_conversation_id}: {e}")
        final_assistant_response = ("Ocurrió un error inesperado al procesar tu solicitud. Por favor, intenta de nuevo.")

    if final_assistant_response:
        logger.info(f"Final assistant response for Conv {sb_conversation_id}: '{str(final_assistant_response)[:200]}...'")
        support_board_service.send_reply_to_channel( 
            conversation_id=sb_conversation_id, message_text=str(final_assistant_response),
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=conversation_data, triggering_message_id=triggering_message_id,
        )
    else:
        logger.error("No final assistant response generated for Conv %s; sending generic fallback.", sb_conversation_id)
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text=("Lo siento, no pude generar una respuesta en este momento. Por favor, intenta de nuevo."),
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=conversation_data, triggering_message_id=triggering_message_id,
        )