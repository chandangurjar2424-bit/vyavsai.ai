from flask import Flask, request, jsonify
import os
import json
import google.generativeai as genai

app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse_voice():
    try:
        # --- 1. VALIDATE INCOMING DATA FROM FLUTTERFLOW ---
        data = request.json
        if not data or 'user_input_text' not in data:
            return jsonify({"error": "Missing user_input_text in request body"}), 400
        
        text = data.get('user_input_text', '')

        # --- 2. VERIFY GEMINI API KEY IS LOADED ---
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY environment variable is missing on Render!"}), 500

        # --- 3. CONFIGURE & CALL GEMINI ---
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        system_prompt = (
            "You are a strict database JSON parsing engine for an Indian retail shop billing application. "
            "Analyze the shopkeeper's text (which might be mixed English/Hindi/regional vernacular). "
            "Identify their task and respond ONLY with a clean JSON object matching one of these structures. "
            "Do not include any conversational words or markdown formatting like ```json blocks.\n\n"
            "Options:\n"
            "1. Billing: {\"intent\": \"GENERATE_BILL\", \"customer_name\": \"...\", \"payment_method\": \"CASH/UPI/UDHAAR\", \"items\": [{\"product_name\": \"...\", \"quantity\": 1, \"unit\": \"kg\"}]}\n"
            "2. Inventory: {\"intent\": \"UPDATE_STOCK\", \"items\": [{\"product_name\": \"...\", \"quantity\": 10, \"hsn_code\": \"...\"}]}\n"
            "3. Ledger: {\"intent\": \"CHECK_LEDGER\", \"customer_name\": \"...\"}"
        )

        response = model.generate_content(f"{system_prompt}\n\nUser input text to parse: {text}")
        
        # --- 4. CLEAN THE TEXT RESPONSE ---
        cleaned_output = response.text.strip()
        
        # Strip away accidental markdown code block ticks if Gemini includes them
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.replace("```json", "").replace("```", "").strip()

        # --- 5. SAFE JSON PARSING TO PREVENT FLUTTERFLOW NULL RESPONSES ---
        try:
            # Convert string format into a real Python dictionary
            json_tree = json.loads(cleaned_output)
            return jsonify(json_tree), 200
        except Exception:
            # If Gemini returned text that isn't clean JSON, return it inside a key instead of breaking
            return jsonify({
                "intent": "ERROR",
                "error": "AI failed to respond in strict JSON format",
                "raw_ai_text": cleaned_output
            }), 200

    except Exception as e:
        # Catch connection or runtime crashes and pass the string explicitly to FlutterFlow
        return jsonify({
            "intent": "CRASH",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

