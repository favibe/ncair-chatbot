import requests
from bs4 import BeautifulSoup
import json
import urllib3
from collections import deque
from rapidfuzz import fuzz
import random
from autocorrect import Speller

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Spell corrector instance
spell = Speller(lang='en')

def preprocess_message(msg):
    return spell(msg.lower().strip())

# URLs to scrape
SCRAPE_URLS = [
    "https://ncair.nitda.gov.ng/",
    "https://ncair.nitda.gov.ng/about-us/",
    "https://ncair.nitda.gov.ng/work-done-so-far/",
    "https://ncair.nitda.gov.ng/aifund/",
    "https://ncair.nitda.gov.ng/aicollective/",
    "https://ncair.nitda.gov.ng/contact/"
]

# Load intents
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    intents_data = data["intents"]

def find_intent(user_message, context):
    user_msg_lower = preprocess_message(user_message)
    best_score = 0
    best_response = None
    best_tag = None

    for intent in intents_data:
        for pattern in intent.get("patterns", []):
            score = fuzz.token_set_ratio(pattern.lower(), user_msg_lower)
            if score > best_score:
                best_score = score
                best_response = random.choice(intent["responses"])
                best_tag = intent["tag"]

    #print(f"[DEBUG] Best score: {best_score} for tag: {best_tag}")  # Debug log

    if best_score >= 85:
        return best_response, best_tag
    else:
        return None, None

def extract_keywords(messages):
    keywords = set()
    for msg in messages:
        if msg.startswith("INTENT:") or msg.startswith("SCRAPED_RESPONSE"):
            continue
        for word in msg.split():
            if len(word) > 3:
                keywords.add(word.lower())
    return keywords

def scrape_relevant_content(keywords):
    for url in SCRAPE_URLS:
        try:
            resp = requests.get(url, verify=False, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
                for para in paragraphs:
                    if any(k in para.lower() for k in keywords):
                        return f"{para}\n\n(Read more: {url})"
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return None

class Chatbot:
    def __init__(self):
        self.user_context = {}
        self.user_last_response = {}

    def update_context(self, user_id, text):
        if user_id not in self.user_context:
            self.user_context[user_id] = deque(maxlen=7)
        self.user_context[user_id].append(text)

    def get_context(self, user_id):
        return list(self.user_context.get(user_id, []))

    def generate_reply(self, user_id, user_message):
        self.update_context(user_id, user_message)
        context = self.get_context(user_id)
        response, intent_tag = find_intent(user_message, context)

        if response:
            self.update_context(user_id, f"INTENT:{intent_tag}")
            if self.user_last_response.get(user_id) == response:
                return "I've already shared that information. Anything else you'd like to ask?"
            self.user_last_response[user_id] = response
            return response

        keywords = extract_keywords(context[-5:])
        fallback_response = scrape_relevant_content(keywords)

        if fallback_response:
            if self.user_last_response.get(user_id) == fallback_response:
                return "I've already shared the relevant info I found. Please ask something else or be more specific."
            self.user_last_response[user_id] = fallback_response
            self.update_context(user_id, "SCRAPED_RESPONSE")
            return fallback_response

        return "Sorry, I couldn't find relevant information on that. Could you please rephrase or ask something else?"

# if __name__ == "__main__":
#     bot = Chatbot()
#     user_id = "user1"
#     print("Chatbot started (type 'exit' to quit).")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         reply = bot.generate_reply(user_id, user_input)
#         print(f"Bot: {reply}")