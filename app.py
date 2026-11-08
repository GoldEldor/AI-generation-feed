import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
# Enable CORS so your browser/Bubble frontend can talk to Render
CORS(app)

# Initialize the Gemini client using the environment variable
# (Make sure GEMINI_API_KEY is set in your Render dashboard environment variables)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided" }, 400)

        # Call Gemini 2.5 Flash with custom system instructions to shape Travis's personality
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config={
                'system_instruction': 'You are Travis, a helpful, witty, and concise AI assistant built on a £0 custom tech stack. Speak naturally and keep answers conversational.'
            }
        )

        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Travis AI backend is live!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
