from flask import Flask, jsonify, request

from rag import answer

app = Flask(__name__)


@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Request body must include a 'question' field"}), 400

    question = body["question"].strip()
    if not question:
        return jsonify({"error": "'question' must not be empty"}), 400

    k = body.get("k", 5)
    result = answer(question, k=k)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
