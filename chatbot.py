from model import get_response, CONFIDENCE_THRESHOLD
from scrapper import Chatbot as ScrapeBot
from memory import get_user_history, append_user_message
import datetime

scrape_bot = ScrapeBot()
USER_ID = "user1"  # could be dynamic if you add a login system


def get_chatbot_response(user_input, context=None):
    """
    Handles a single turn of conversation:
    1. Loads user history from memory.json (single source of truth for
       chat state -- app.py should never write directly to memory.json).
    2. Appends the new user message.
    3. Tries the trained intent classifier first.
    4. If the classifier's confidence is below threshold, falls back to
       live keyword scraping of the NCAIR site.
    5. Appends the bot reply and returns the updated history.

    Args:
        user_input (str): the message just typed by the user.
        context: unused, kept for backwards-compatible call signature.
                 History is always read fresh from memory.json instead.

    Returns:
        dict: {"text": str, "timestamp": str, "context": list[dict]}
              "context" is the full updated chat history (list of
              {"sender", "message", "timestamp"} dicts) -- app.py should
              use this directly as its session_state.chat_history.
    """
    history = get_user_history(USER_ID)
    context_strings = [f"{item['sender']}: {item['message']}" for item in history]

    append_user_message(USER_ID, "user", user_input)
    context_strings.append(f"user: {user_input}")

    recent_context = context_strings[-6:]
    result = get_response(user_input, recent_context)

    if result["confidence"] >= CONFIDENCE_THRESHOLD:
        bot_reply = result["text"]
    else:
        # Neural net wasn't confident -- try the live scraping fallback
        # before giving up.
        scraped_reply = scrape_bot.generate_reply(USER_ID, user_input)
        bot_reply = scraped_reply if scraped_reply else result["text"]

    append_user_message(USER_ID, "bot", bot_reply)

    updated_history = get_user_history(USER_ID)
    return {
        "text": bot_reply,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "context": updated_history,
    }