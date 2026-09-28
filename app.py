from flask import Flask, render_template, request, jsonify
import json
import random
import pickle


# Create Flask app
app = Flask(__name__)


# Load chatbot data
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# Load trained ML model
with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Load label encoder
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# Find response for predicted intent
def get_response(intent):
    for item in data["intents"]:
        if item["tag"] == intent:
            return random.choice(item["responses"])

    return "Sorry, I don't understand your question."


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chat API
@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    # Convert user message into numbers
    message_vector = vectorizer.transform([user_message])

    # Predict intent
    prediction = model.predict(message_vector)

    # Convert number back to intent name
    intent = label_encoder.inverse_transform(prediction)[0]

    # Get chatbot response
    response = get_response(intent)

    return jsonify({
        "response": response,
        "intent": intent
    })


# Start server
if __name__ == "__main__":
    app.run(debug=True)