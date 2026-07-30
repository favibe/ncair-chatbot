# model.py

import json

import torch

import numpy as np

import nltk

from nltk.stem.porter import PorterStemmer

import datetime

nltk.download('punkt', quiet=True)


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


stemmer = PorterStemmer()

CONFIDENCE_THRESHOLD = 0.75


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag


# Load saved model and metadata
data = torch.load("intent_model.pth")

all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)


def get_response(user_input, context=None):
    """
    Classify user_input (optionally combined with recent context) against
    the trained intent model.

    Args:
        user_input (str): current user message string
        context (list): list of previous conversation strings (optional)

    Returns:
        dict: {
            "text": str,          # the intent-based response, or a
                                    # low-confidence fallback message
            "timestamp": str,
            "confidence": float,   # 0.0-1.0, softmax confidence of the
                                    # predicted intent
            "tag": str or None,    # predicted intent tag, None if below
                                    # threshold
        }

    NOTE: This function does NOT decide whether to use the scraping
    fallback -- it just reports its confidence. The caller
    (chatbot.get_chatbot_response) is responsible for deciding what to do
    when confidence is below CONFIDENCE_THRESHOLD.
    """
    combined_text = ""
    if context:
        combined_text = " ".join(context[-1:]) + " "
    combined_text += user_input

    sentence = tokenize(combined_text)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0).float()

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()].item()

    response = "Sorry, I didn't understand that. Could you please rephrase?"
    predicted_tag = None

    if confidence > CONFIDENCE_THRESHOLD:
        predicted_tag = tag
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                response = np.random.choice(intent["responses"])
                break

    return {
        "text": response,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "confidence": confidence,
        "tag": predicted_tag,
    }