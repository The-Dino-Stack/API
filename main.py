from flask import Flask, request, jsonify
import os
import json
import numpy as np
from dotenv import load_dotenv
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from llm_providers import get_llm_provider

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# API key for authentication on the prompt endpoint
PROMPT_API_KEY = os.getenv("PROMPT_API_KEY")
if not PROMPT_API_KEY:
    raise RuntimeError("PROMPT_API_KEY environment variable is not set. The application cannot start.")

# API key for authentication on the upload-embeddings endpoint
UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY")
if not UPLOAD_API_KEY:
    raise RuntimeError("UPLOAD_API_KEY environment variable is not set. The application cannot start.")

# Helper: compute cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/upload-embeddings", methods=["POST"])
def upload_embeddings():
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != UPLOAD_API_KEY:
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
    if api_key != UPLOAD_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({"status": "OK"}), 200

@app.route("/prompt", methods=["POST"])
def handle_prompt():
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != PROMPT_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # Check if JSON data is included in the request
    if not request.is_json:
        return jsonify({"error": "Invalid content type, expected JSON"}), 400

    data = request.get_json()

    # Validate required fields
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing required fields: prompt"}), 400

    print(prompt)

    # Load document embeddings
    try:
        with open("uploads/embeddings.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Embeddings file not found"}), 500

    doc_embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)

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

    # Convert to 2D arrays for sklearn
    q_emb = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)

    # Compute cosine similarities in one go
    sims = sk_cosine_similarity(q_emb, doc_embeddings)[0]

    # Get top-k docs
    top_k = 30
    top_indices = np.argsort(sims)[::-1][:top_k]
    ranked_docs = [docs[i] for i in top_indices]

    top_context = "\n\n---\n\n".join(
        f"[{os.path.splitext(os.path.basename(doc['filename']))[0]}]({os.path.splitext(doc['filename'])[0]})\n\n{doc['content']}"
        for doc in ranked_docs[:top_k]
    )

    # Call AI API with the context and prompt
    try:
        ai_response = llm_provider.prompt(
                f"""
            You are a helpful documentation assistant. You must follow these rules:
            - Use Markdown links in the format: `[something relevant to the file being referenced](path/filename)`
            - Only use filenames and paths provided in the documentation context.
            - Do NOT guess file paths.
            
            Use only the information provided in the documentation below to answer the user's question. 
            If the answer to a question about the documentation cannot be found in the documentation, say you don't know instead of making something up.
            
            Documentation:
            {top_context}
            
            User question:
            {prompt}
            """
        )
    except Exception as e:
        return jsonify({"error": f"Failed to call AI API: {e}"}), 500

    print(ai_response)

    # Return the AI API response
    return jsonify({"answer": ai_response}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051)