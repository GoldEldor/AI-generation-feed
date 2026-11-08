import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)
client = genai.Client()

@app.route("/chat", methods=["POST"])
def generate_talking_head():
    data = request.json
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Prompt describing Travis speaking and moving
    prompt = (
        f"A video of Travis, a sharp and helpful AI assistant, talking directly to the camera "
        f"and saying: {user_input}"
    )

    try:
        # Using Gemini Omni Flash for native video generation with speech
        result = client.models.generate_content(
            model='gemini-omni-1.2-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="video/mp4" # Requesting direct video output if supported by your pipeline
            )
        )
        
        # In production, you'd parse the video URI or stream returned by the API
        return jsonify({"status": "success", "video_output": str(result)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
  
