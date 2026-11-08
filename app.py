import os
import time
import threading
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

# Initialize the Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}, 400)

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
    return jsonify({"status": "Travis AI backend is live and awake!"})

# --- SELF-PUMPING HEARTBEAT (Keeps Render from sleeping) ---
def keep_alive():
    # Give the server 10 seconds to fully boot up initially
    time.sleep(10)
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    
    if not render_url:
        print("ℹ️ RENDER_EXTERNAL_URL not found, self-ping skipped.")
        return

    print(f"🚀 Self-ping heartbeat activated for: {render_url}")
    while True:
        try:
            # Ping itself every 9 minutes (540 seconds)
            urllib.request.urlopen(render_url)
            print("💓 Heartbeat ping sent successfully.")
        except Exception as e:
            print(f"⚠️ Heartbeat ping failed: {e}")
        
        time.sleep(540) 

# Start the background thread when the app launches
if os.environ.get("RENDER"):
    threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
