from flask import Flask, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load tuple data (vectorizer, model, df)
with open("chatbot.pkl", "rb") as f:
    vectorizer, model, df = pickle.load(f)

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Transform query
    X_query = vectorizer.transform([user_query])

    # Compute cosine similarity
    sims = cosine_similarity(X_query, model).flatten()
    idx = sims.argmax()
    best_answer = df.iloc[idx]["answer"]

    return jsonify({
        "query": user_query,
        "answer": best_answer
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)