"""
Voice-first billing: LLM parsing API (Flask) for Render's free tier.

Endpoints
    GET  /         -> health/info (Render health check can point here or at /health)
    GET  /health   -> {"status": "ok"}
    POST /parse    -> body: {"user_input_text": "...", "language": "hi-IN" (optional)}
                      returns the parsed JSON structure

Render setup (Web Service, Python)
    Build Command:  pip install -r requirements.txt
    Start Command:  gunicorn app:app --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:$PORT
    Environment variables:
        GEMINI_API_KEY   (required)  your Google AI Studio key
        GEMINI_MODEL     (optional)  default gemini-2.5-flash
        API_KEY          (recommended) shared secret; clients must send header X-API-Key
        CORS_ORIGIN      (optional)  default *
        PYTHON_VERSION   (optional)  e.g. 3.11.9

FlutterFlow API Call
    Method: POST   URL: https://<your-service>.onrender.com/parse
    Headers: Content-Type: application/json ; X-API-Key: [apiKey]
    Body (JSON): {"user_input_text": "<rawText>", "language": "<language>"}
    JSON paths: $.success, $.needs_confirmation, $.warnings,
                $.data.intent, $.data.customer_name, $.data.payment_method,
                $.data.items, $.data.items[:].name / quantity / unit / hsn_code
    Render free tier sleeps after ~15 min idle; the first call after a nap can take
    30-60 s, so set a long timeout in FlutterFlow and show a loading state.

Local run:  GEMINI_API_KEY=... python app.py   (binds to PORT, default 5000)
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Optional

from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("voice-orchestrator")
logging.basicConfig(level=logging.INFO)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TEXT_CHARS = 1000
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")  # lock down for FlutterFlow web
API_KEY = os.environ.get("API_KEY")  # optional shared secret, see /parse


# --------------------------------------------------------------------------- #
# 1. Output schema (also given to Gemini as the response schema)
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    GENERATE_BILL = "GENERATE_BILL"
    UPDATE_STOCK = "UPDATE_STOCK"
    CHECK_LEDGER = "CHECK_LEDGER"
    UNKNOWN = "UNKNOWN"  # model could not map the request to a supported action


class PaymentMethod(str, Enum):
    CASH = "CASH"
    UPI = "UPI"
    UDHAAR = "UDHAAR"
    CARD = "CARD"


class Item(BaseModel):
    name: str = Field(description="Product name in Latin script, e.g. 'Basmati Rice', '15W LED Bulb'")
    quantity: Optional[float] = Field(default=None, description="Numeric quantity, e.g. 2, 0.5, 10")
    unit: Optional[str] = Field(default=None, description="kg, g, litre, ml, pcs, packet, box, dozen or null")
    hsn_code: Optional[str] = Field(default=None, description="HSN digits only if the user spoke one")
    price: Optional[float] = Field(default=None, description="Per-unit price in rupees only if the user spoke one")


class ParsedCommand(BaseModel):
    intent: Intent
    customer_name: Optional[str] = Field(default=None, description="Customer name in Latin script, or null")
    payment_method: Optional[PaymentMethod] = Field(default=None, description="Null if not mentioned")
    items: list[Item] = Field(default_factory=list)
    confidence: float = Field(default=0.0, description="0.0 to 1.0 self-assessed parse confidence")


# --------------------------------------------------------------------------- #
# 2. Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are the command parser for a voice-first GST billing app used by small
Indian shopkeepers (kirana, electronics, hardware). The input is raw speech-to-text output
in Hindi, Tamil, Telugu, Marathi, Gujarati, Bengali, English or a code-switched mix
(Hinglish, Tamil-English, etc.), possibly in native script, possibly with transcription errors.

Return ONE JSON object matching the schema. Rules:

INTENT
- GENERATE_BILL: the user wants to sell something / make a bill / put something on udhaar
  ("bill banao", "do", "de do", "likh do", "udhaar pe do", "bill podu").
- UPDATE_STOCK: the user is adding or receiving stock ("stock mein add karo", "maal aaya",
  "inventory badhao"). Quantities are additions.
- CHECK_LEDGER: the user asks how much a customer owes / the udhaar balance / hisaab
  ("Ramesh ka kitna baaki hai", "Sharma ji ka hisaab").
- UNKNOWN: anything else, or if the request is too unclear to act on.

FIELDS
- customer_name: the person named as buyer, in Latin script (transliterate native script,
  keep honorifics like "ji" or "bhai": "Sharma ji"). Null if none, or for walk-in sales.
- payment_method: UDHAAR for udhaar/credit/khata/kadan/baaki; CASH for cash/nakad/rokda;
  UPI for UPI/GPay/PhonePe/Paytm/online; CARD for card. Null if not stated. Do NOT guess.
  If a customer is named with "udhaar pe do" it is UDHAAR.
- items: one entry per product. Translate product names to common English/Latin-script
  trade names ("chawal" -> "Rice", "Basmati Chawal" -> "Basmati Rice", "namak" -> "Salt",
  "cheeni" -> "Sugar"). Keep brand names and specs exactly ("15W LED Bulb", "Tata Salt").
- quantity: a number. Convert number words in any language ("do" = 2, "aadha" = 0.5,
  "dhai" = 2.5, "dedh" = 1.5, "paanch" = 5, "irandu" = 2). Null if truly missing.
- unit: one of kg, g, litre, ml, pcs, packet, box, dozen. "kilo"/"kilo" -> kg,
  "piece"/"nag"/"number" -> pcs. Null if not stated and not obvious.
- hsn_code: ONLY if the user spoke an HSN number; digits only. Never invent one.
- price: ONLY if the user spoke a price for that item.
- Never invent items, quantities, names or amounts that are not in the text.
- confidence: 1.0 for clear commands, lower when words were ambiguous, garbled or guessed.

Examples
Input: "Ramesh ko 2 kilo Basmati Chawal udhaar pe do aur bill banao"
Output: {"intent":"GENERATE_BILL","customer_name":"Ramesh","payment_method":"UDHAAR",
"items":[{"name":"Basmati Rice","quantity":2,"unit":"kg","hsn_code":null,"price":null}],"confidence":0.95}

Input: "Add 10 pieces of 15W LED bulbs to stock with HSN 8539"
Output: {"intent":"UPDATE_STOCK","customer_name":null,"payment_method":null,
"items":[{"name":"15W LED Bulb","quantity":10,"unit":"pcs","hsn_code":"8539","price":null}],"confidence":0.97}

Input: "Sharma ji ka kitna udhaar baaki hai"
Output: {"intent":"CHECK_LEDGER","customer_name":"Sharma ji","payment_method":null,
"items":[],"confidence":0.95}
"""


# --------------------------------------------------------------------------- #
# 3. LLM call (Gemini JSON mode). Swap this one function to use another LLM.
# --------------------------------------------------------------------------- #
_client = None


def _get_client():
    """Lazy init so cold starts that only hit OPTIONS stay fast."""
    global _client
    if _client is None:
        from google import genai

        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
            # Vertex AI path: uses the function's service account, no API key needed.
            _client = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1"),
            )
        else:
            _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _call_llm(text: str, language_hint: Optional[str]) -> dict:
    """Returns the raw parsed JSON dict from the model."""
    from google.genai import types

    user_msg = f"Language hint: {language_hint or 'unknown'}\nInput: {text}"
    resp = _get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",  # JSON mode
            response_schema=ParsedCommand,          # enforce structure
            temperature=0,                          # deterministic parsing
            max_output_tokens=800,
        ),
    )
    # resp.parsed is a ParsedCommand when the schema is honoured; fall back to text.
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed.model_dump(mode="json")
    return json.loads(resp.text)


# --------------------------------------------------------------------------- #
# 4. Normalisation + validation (never trust LLM output blindly)
# --------------------------------------------------------------------------- #
UNIT_MAP = {
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "किलो": "kg", "கிலோ": "kg",
    "g": "g", "gm": "g", "gram": "g", "grams": "g", "ग्राम": "g", "கிராம்": "g",
    "l": "litre", "ltr": "litre", "litre": "litre", "liter": "litre", "लीटर": "litre", "லிட்டர்": "litre",
    "ml": "ml", "millilitre": "ml",
    "pcs": "pcs", "pc": "pcs", "piece": "pcs", "pieces": "pcs", "nos": "pcs", "no": "pcs",
    "nag": "pcs", "number": "pcs", "numbers": "pcs", "unit": "pcs", "units": "pcs", "नग": "pcs",
    "packet": "packet", "packets": "packet", "pkt": "packet", "पैकेट": "packet",
    "box": "box", "boxes": "box", "डिब्बा": "box",
    "dozen": "dozen", "darjan": "dozen", "दर्जन": "dozen",
}
ALLOWED_UNITS = {"kg", "g", "litre", "ml", "pcs", "packet", "box", "dozen"}


def _normalise_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.strip().lower().rstrip(".")
    u = UNIT_MAP.get(u, u)
    return u if u in ALLOWED_UNITS else None


def _clean_hsn(hsn: Optional[str]) -> Optional[str]:
    """HSN codes are 2/4/6/8 digits. Keep digits only, drop anything else."""
    if not hsn:
        return None
    digits = re.sub(r"\D", "", str(hsn))
    return digits if len(digits) in (2, 4, 6, 8) else None


def _validate_and_flag(parsed: ParsedCommand) -> tuple[dict, list[str], bool]:
    """Clean fields, collect warnings, decide if the app must ask the user to confirm."""
    warnings: list[str] = []
    cleaned_items = []

    for it in parsed.items:
        name = it.name.strip()
        if not name:
            continue
        qty = it.quantity
        if qty is not None and (qty <= 0 or qty > 100000):
            warnings.append(f"Unusual quantity for '{name}': {qty}")
            qty = None
        unit = _normalise_unit(it.unit)
        hsn = _clean_hsn(it.hsn_code)
        if it.hsn_code and not hsn:
            warnings.append(f"Ignored invalid HSN '{it.hsn_code}' for '{name}'")
        price = it.price if (it.price is not None and it.price > 0) else None
        if qty is None:
            warnings.append(f"Quantity missing for '{name}'")
        cleaned_items.append(
            {"name": name, "quantity": qty, "unit": unit, "hsn_code": hsn, "price": price}
        )

    intent = parsed.intent.value
    customer = (parsed.customer_name or "").strip() or None
    payment = parsed.payment_method.value if parsed.payment_method else None

    # Business-rule checks per intent
    if intent in ("GENERATE_BILL", "UPDATE_STOCK") and not cleaned_items:
        warnings.append("No items detected")
    if intent == "GENERATE_BILL":
        if payment == "UDHAAR" and not customer:
            warnings.append("Udhaar needs a customer name")
        if payment is None:
            warnings.append("Payment method not mentioned")
    if intent == "CHECK_LEDGER" and not customer:
        warnings.append("Customer name missing for ledger check")
    if intent == "CHECK_LEDGER":
        cleaned_items = []  # a ledger query never has items

    needs_confirmation = (
        intent == "UNKNOWN" or bool(warnings) or parsed.confidence < 0.8
    )
    data = {
        "intent": intent,
        "customer_name": customer,
        "payment_method": payment,
        "items": cleaned_items,
        "confidence": round(max(0.0, min(1.0, parsed.confidence)), 2),
    }
    return data, warnings, needs_confirmation


# --------------------------------------------------------------------------- #
# 5. Core, framework-agnostic entry point
# --------------------------------------------------------------------------- #
def parse_text(text: str, language_hint: Optional[str] = None) -> dict:
    """
    Parse raw speech-to-text into a structured command.

    Returns a dict ready to be JSON-serialised:
      {"success": bool, "needs_confirmation": bool, "warnings": [...],
       "data": {...}, "raw_text": str}
    Raises ValueError for bad input and RuntimeError if the LLM keeps failing.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("'text' is required")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"'text' too long (max {MAX_TEXT_CHARS} chars)")

    last_err: Optional[Exception] = None
    for attempt in (1, 2):  # one retry covers transient API errors / malformed JSON
        try:
            raw = _call_llm(text, language_hint)
            parsed = ParsedCommand.model_validate(raw)
            data, warnings, needs_conf = _validate_and_flag(parsed)
            return {
                "success": data["intent"] != "UNKNOWN",
                "needs_confirmation": needs_conf,
                "warnings": warnings,
                "data": data,
                "raw_text": text,
            }
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            logger.warning("Attempt %d: bad model output: %s", attempt, e)
        except Exception as e:  # network, quota, auth
            last_err = e
            logger.warning("Attempt %d: LLM call failed: %s", attempt, e)
    raise RuntimeError(f"LLM parsing failed: {last_err}")


# --------------------------------------------------------------------------- #
# 6. Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
app.json.ensure_ascii = False  # keep Hindi/Tamil readable in responses


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    return resp


@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "voice-billing-parser", "endpoints": ["POST /parse", "GET /health"]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/parse", methods=["POST", "OPTIONS"])
def parse():
    if request.method == "OPTIONS":  # CORS preflight
        return ("", 204)

    # Optional shared-secret check (Render URLs are public, so set API_KEY in production)
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"success": False, "error": "Body must be a JSON object"}), 400

    text = body.get("user_input_text")
    if not isinstance(text, str):
        return jsonify({"success": False, "error": "'user_input_text' (string) is required"}), 400

    try:
        result = parse_text(text, body.get("language"))
        logger.info("intent=%s conf=%s", result["data"]["intent"], result["data"]["confidence"])
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except RuntimeError as e:
        logger.error("%s", e)
        return jsonify({"success": False, "error": "Could not understand the command, please try again"}), 502


if __name__ == "__main__":
    # Render injects PORT; fall back to 5000 locally. Production uses gunicorn (see Start Command).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
