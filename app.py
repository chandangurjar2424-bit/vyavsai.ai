from flask import Flask, request, jsonify
import os
import json
import google.generativeai as genai

app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse_voice():
    # --- 1. CAPTURE DATA FROM FLUTTERFLOW ---
    try:
        data = request.get_json(force=True, silent=True)
    except Exception:
        return jsonify({"intent": "ERROR", "error": "Invalid request body style. Use application/json format."}), 400

    if not data or 'user_input_text' not in data:
        return jsonify({"intent": "ERROR", "error": "Missing user_input_text parameter inside JSON body."}), 400
    
    text = data.get('user_input_text', '')

    # --- 2. VALIDATE THE ENVIRONMENT VARIABLE ---
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return jsonify({"intent": "ERROR", "error": "CRITICAL: The GEMINI_API_KEY environment variable is completely missing from your Render Dashboard configuration."}), 500

    # --- 3. EXECUTE SAFE GEMINI CALL ---
    try:
        genai.configure(api_key=api_key)
        
        # We enforce a concrete configuration pattern to ensure it cannot return a structural null response
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        }
        
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=generation_config
        )

        system_prompt = (
            "You are a backend structural API database compiler. You convert messy Indian retail shop voice transcripts "
            "into structured database trees. Analyze the query and return ONLY a strict valid JSON dictionary tree match. "
            "Do not include any conversational sentences or markdown code blocks like ```json text formatting.\n\n"
            "Valid Options:\n"
            "1. Billing intent: {\"intent\": \"GENERATE_BILL\", \"customer_name\": \"...\", \"payment_method\": \"CASH/UPI/UDHAAR\", \"items\": [{\"product_name\": \"...\", \"quantity\": 1, \"unit\": \"kg\"}]}\n"
            "2. Stock intent: {\"intent\": \"UPDATE_STOCK\", \"items\": [{\"product_name\": \"...\", \"quantity\": 10, \"hsn_code\": \"...\"}]}\n"
            "3. Ledger intent: {\"intent\": \"CHECK_LEDGER\", \"customer_name\": \"...\"}"
        )

        response = model.generate_content(f"{system_prompt}\n\nShopkeeper Text: {text}")
        
        # Verify the model returned something valid
        if not response or not response.text:
            return jsonify({"intent": "ERROR", "error": "Gemini API successfully connected but returned an empty response string."}), 500

        cleaned_output = response.text.strip()
        
        # Clean any remaining code ticks if generated despite the mime setting
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.replace("```json", "").replace("```", "").strip()

        # Parse string cleanly to return real JSON objects to FlutterFlow
        json_tree = json.loads(cleaned_output)
        return jsonify(json_tree), 200

    except json.JSONDecodeError as json_err:
        return jsonify({
            "intent": "ERROR", 
            "error": "The AI model responded with raw conversational text instead of clean data fields.",
            "raw_text_received": cleaned_output
        }), 200
    except Exception as e:
        return jsonify({"intent": "ERROR", "error": f"The model call failed during initialization: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
