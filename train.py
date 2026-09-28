import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# 1. Load training data
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# 2. Prepare sentences and labels
sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])


# 3. Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)


# 4. Convert intent names into numbers
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)


# 5. Create Machine Learning model
model = LogisticRegression(max_iter=1000)

model.fit(X, y)


# 6. Save the trained model
with open("chatbot_model.pkl", "wb") as file:
    pickle.dump(model, file)


# 7. Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# 8. Save the label encoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)


print("=================================")
print("AI Chatbot Model Trained!")
print("=================================")
print("Training examples:", len(sentences))
print("Intents:", len(set(labels)))
print("Model saved successfully.")