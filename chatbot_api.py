from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from googletrans import Translator

with open("chatbot.pkl", "rb") as f:
    vectorizer, model, df = pickle.load(f)

app = Flask(__name__)
CORS(app)

translator = Translator()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    lang = data.get("lang", "en")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    query_for_model = user_query
    if lang != "en":
        try:
            query_for_model = translator.translate(user_query, src=lang, dest="en").text
        except Exception as e:
            print("Translation error (query):", e)

    X_query = vectorizer.transform([query_for_model])
    sims = cosine_similarity(X_query, model).flatten()
    idx = sims.argmax()
    best_answer = df.iloc[idx]["answer"]

    final_answer = best_answer
    if lang != "en":
        try:
            final_answer = translator.translate(best_answer, src="en", dest=lang).text
        except Exception as e:
            print("Translation error (answer):", e)

    return jsonify({
        "query": user_query,
        "answer": final_answer,
        "lang": lang
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)