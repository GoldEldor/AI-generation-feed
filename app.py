import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from werkzeug.exceptions import RequestEntityTooLarge

from native import strip_metadata

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

TAVUS_BASE = "https://tavusapi.com/v2"
SIMLI_BASE = "https://api.simli.ai"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("livevision")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
if frontend_origin:
    CORS(app, resources={r"/api/*": {"origins": [frontend_origin]}})
else:
    CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY", "").strip()
TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID", "").strip()
TAVUS_PERSONA_ID = os.getenv("TAVUS_PERSONA_ID", "").strip()
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "").strip()
SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "").strip()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def safe_provider_error(provider, response):
    logger.warning("%s returned HTTP %s", provider, response.status_code)
    return f"{provider} request failed (HTTP {response.status_code})."


def read_upload_signature(file_path):
    with open(file_path, "rb") as image:
        header = image.read(16)

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


def get_owned_upload(filename):
    if not isinstance(filename, str) or not re.fullmatch(r"[a-f0-9]{32}\.(jpg|png)", filename):
        return None
    path = UPLOAD_DIR / filename
    return path if path.is_file() else None


def post_provider(url, *, headers, payload, provider, timeout=45):
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(5, timeout),
        )
    except requests.Timeout:
        return None, f"{provider} timed out. Please try again.", 504
    except requests.RequestException:
        logger.exception("Could not reach %s", provider)
        return None, f"Could not reach {provider}. Please try again.", 502

    if not response.ok:
        return None, safe_provider_error(provider, response), 502

    try:
        return response.json(), None, 200
    except requests.exceptions.JSONDecodeError:
        logger.warning("%s returned a non-JSON response", provider)
        return None, f"{provider} returned an unexpected response.", 502


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/status")
def status():
    configured = {
        "gemini_configured": bool(GEMINI_API_KEY),
        "tavus_configured": bool(TAVUS_API_KEY and TAVUS_REPLICA_ID),
        "tavus_conversation_configured": bool(
            TAVUS_API_KEY and TAVUS_REPLICA_ID and TAVUS_PERSONA_ID
        ),
        "simli_configured": bool(SIMLI_API_KEY),
        "simli_face_configured": bool(SIMLI_FACE_ID),
    }
    return jsonify(**configured, status="ready" if any(configured.values()) else "needs_setup")


@app.post("/api/upload")
def upload():
    image = request.files.get("image")
    if not image or not image.filename:
        return jsonify(error="Choose an image first."), 400

    extension_by_mime = {
        "image/jpeg": "jpg",
        "image/png": "png",
    }

    upload_id = uuid.uuid4().hex
    temporary_path = UPLOAD_DIR / f"{upload_id}.part"
    try:
        image.save(temporary_path)
        detected_mime = read_upload_signature(temporary_path)
        if detected_mime not in ALLOWED_MIME_TYPES:
            temporary_path.unlink(missing_ok=True)
            return jsonify(error="Upload a JPEG or PNG image."), 400

        try:
            cleaned = strip_metadata(temporary_path.read_bytes())
        except ValueError:
            temporary_path.unlink(missing_ok=True)
            return jsonify(error="Could not read or parse image metadata."), 400
        temporary_path.write_bytes(cleaned)

        final_name = f"{upload_id}.{extension_by_mime[detected_mime]}"
        final_path = UPLOAD_DIR / final_name
        temporary_path.replace(final_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        logger.exception("Image upload failed")
        return jsonify(error="Could not save the image."), 500

    return jsonify(
        filename=final_name,
        image_url=f"/uploads/{final_name}",
        mime_type=detected_mime,
    ), 201


@app.get("/uploads/<path:filename>")
def uploaded_image(filename):
    path = get_owned_upload(filename)
    if path is None:
        return jsonify(error="Image not found."), 404
    return send_from_directory(UPLOAD_DIR, path.name, max_age=3600)


@app.post("/api/generate-script")
def generate_script():
    if not gemini_client:
        return jsonify(error="GEMINI_API_KEY is not configured."), 503

    data = json_body()
    filename = data.get("filename")
    user_prompt = data.get("prompt", "Write a concise, natural spoken script inspired by the image.")
    if not isinstance(user_prompt, str) or len(user_prompt) > 1000:
        return jsonify(error="Prompt must be 1,000 characters or fewer."), 400

    image_path = get_owned_upload(filename)
    if image_path is None:
        return jsonify(error="Image not found. Upload it again and retry."), 404

    mime_type = read_upload_signature(image_path)
    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify(error="The uploaded image is not valid."), 400

    try:
        image_bytes = image_path.read_bytes()
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                (
                    "Write a short script for an AI avatar to speak, using the image "
                    "only as visual context. Do not identify a real person or invent "
                    "specific facts about them. Return only the spoken script, no "
                    "heading or explanation. Keep it under three sentences.\n\n"
                    f"User direction: {user_prompt}"
                ),
            ],
        )
        script = (response.text or "").strip()
        if not script:
            return jsonify(error="Gemini did not return a script. Try a different prompt."), 502
        return jsonify(script=script)
    except Exception:
        logger.exception("Gemini script generation failed")
        return jsonify(error="Script generation failed. Check the server logs and try again."), 502


@app.post("/api/tavus/video")
def tavus_video():
    if not TAVUS_API_KEY or not TAVUS_REPLICA_ID:
        return jsonify(error="Configure TAVUS_API_KEY and TAVUS_REPLICA_ID on the server."), 503

    data = json_body()
    script = data.get("script", "")
    if not isinstance(script, str) or not script.strip():
        return jsonify(error="A script is required."), 400
    if len(script) > 5000:
        return jsonify(error="Script must be 5,000 characters or fewer."), 400

    payload = {
        "replica_id": TAVUS_REPLICA_ID,
        "script": script.strip(),
        "video_name": "LiveVision generated video",
    }
    result, error, code = post_provider(
        f"{TAVUS_BASE}/videos",
        headers={"x-api-key": TAVUS_API_KEY, "Content-Type": "application/json"},
        payload=payload,
        provider="Tavus",
    )
    return (jsonify(result), code) if error is None else (jsonify(error=error), code)


@app.post("/api/tavus/conversation")
def tavus_conversation():
    if not TAVUS_API_KEY or not TAVUS_REPLICA_ID or not TAVUS_PERSONA_ID:
        return jsonify(
            error="Configure TAVUS_API_KEY, TAVUS_REPLICA_ID, and TAVUS_PERSONA_ID."
        ), 503

    payload = {
        "replica_id": TAVUS_REPLICA_ID,
        "persona_id": TAVUS_PERSONA_ID,
        "conversation_name": "LiveVision conversation",
        "conversational_context": (
            "You are a helpful, friendly avatar. Keep responses concise. "
            "Do not claim to know facts that have not been provided."
        ),
    }
    result, error, code = post_provider(
        f"{TAVUS_BASE}/conversations",
        headers={"x-api-key": TAVUS_API_KEY, "Content-Type": "application/json"},
        payload=payload,
        provider="Tavus",
    )
    return (jsonify(result), code) if error is None else (jsonify(error=error), code)


@app.post("/api/simli/session")
def simli_session():
    if not SIMLI_API_KEY:
        return jsonify(error="SIMLI_API_KEY is not configured."), 503

    data = json_body()
    face_id = data.get("face_id") or SIMLI_FACE_ID
    if not isinstance(face_id, str) or not face_id.strip():
        return jsonify(error="Enter a Simli face ID or configure SIMLI_FACE_ID."), 400

    result, error, code = post_provider(
        f"{SIMLI_BASE}/compose/token",
        headers={"x-simli-api-key": SIMLI_API_KEY, "Content-Type": "application/json"},
        payload={
            "faceId": face_id.strip(),
            "maxSessionLength": 600,
            "maxIdleTime": 180,
        },
        provider="Simli",
        timeout=30,
    )
    return (jsonify(result), code) if error is None else (jsonify(error=error), code)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="Image is too large. Maximum upload size is 12 MB."), 413


@app.errorhandler(500)
def internal_error(_error):
    logger.exception("Unhandled server error")
    return jsonify(error="Unexpected server error."), 500


# --- SELF-PUMPING HEARTBEAT (Keeps Render awake 24/7) ---
def keep_alive():
    time.sleep(15)
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not render_url:
        return
    while True:
        try:
            urllib.request.urlopen(render_url)
        except Exception:
            pass
        time.sleep(540)

if os.environ.get("RENDER"):
    threading.Thread(target=keep_alive, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
