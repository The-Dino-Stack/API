from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# API key for authentication
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set. The application cannot start.")

@app.route("/upload-embeddings", methods=["POST"])
def upload_embeddings():
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # Check if JSON data is included in the request
    if not request.is_json:
        return jsonify({"error": "Invalid content type, expected JSON"}), 400

    data = request.get_json()

    # Save the JSON data to a file
    save_path = os.path.join("uploads", "embeddings.json")
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "w") as f:
        import json
        json.dump(data, f)

    return jsonify({"message": "Embeddings uploaded successfully", "path": save_path}), 200

@app.route("/health", methods=["GET"])
def health_check():
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({"status": "OK"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051)