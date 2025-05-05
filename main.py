from flask import Flask, request, jsonify
import os
import json
import numpy as np
from dotenv import load_dotenv

from llm_providers import get_llm_provider

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# API key for authentication
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set. The application cannot start.")

# Helper: compute cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
        json.dump(data, f)

    return jsonify({"message": "Embeddings uploaded successfully", "path": save_path}), 200

@app.route("/health", methods=["GET"])
def health_check():
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({"status": "OK"}), 200

@app.route("/prompt", methods=["POST"])
def handle_prompt():
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # Check if JSON data is included in the request
    if not request.is_json:
        return jsonify({"error": "Invalid content type, expected JSON"}), 400

    data = request.get_json()

    # Validate required fields
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing required fields: prompt"}), 400

    # Load document embeddings
    try:
        with open("uploads/embeddings.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Embeddings file not found"}), 500

    # Get LLM provider
    try:
        llm_provider = get_llm_provider()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Generate embedding for the prompt
    try:
        prompt_embedding = llm_provider.embed_prompt(prompt)
    except Exception as e:
        return jsonify({"error": f"Failed to generate embedding: {e}"}), 500

    # Fetch top-k relevant context snippets
    top_k = 5
    ranked_docs = sorted(
        docs,
        key=lambda d: cosine_similarity(prompt_embedding, d["embedding"]),
        reverse=True
    )
    top_context = "\n\n---\n\n".join(
        f"From `{doc['filename']}`:\n{doc['content']}" for doc in ranked_docs[:top_k]
    )

    # Call AI API with the context and prompt
    try:
        ai_response = llm_provider.prompt(
            f"Refer to the following documentation to answer:\n{top_context}\n\nQuestion: {prompt}"
        )
    except Exception as e:
        return jsonify({"error": f"Failed to call AI API: {e}"}), 500

    # Return the AI API response
    return jsonify({"answer": ai_response}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051)