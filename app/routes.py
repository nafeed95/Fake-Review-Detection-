from flask import Blueprint, request, jsonify
from .utils import clean_text, predict_review, load_model_and_tools

main = Blueprint('main', __name__)


model, tokenizer = load_model_and_tools()

@main.route('/')
def index():
    return "Fake Review Detection API is Running!"

@main.route('/predict', methods=['POST'])
def predict():
    if not model or not tokenizer:
        return jsonify({"error": "Model or tokenizer not loaded."}), 500

    data = request.get_json()
    reviews = data.get('reviews', [])
    category = data.get('category', 'unknown')
    rating = str(data.get('rating', '0'))

    if not isinstance(reviews, list):
        return jsonify({"error": "Expected 'reviews' to be a list."}), 400

    results = []
    for review in reviews:
        label, prob = predict_review(review, category, rating, model, tokenizer)
        results.append({
            "review": review,
            "prediction": label,
            "probability": round(prob, 4)
        })

    return jsonify({
        "count": len(results),
        "results": results
    })
