import json
import os
from datetime import datetime

MEMORY_FILE = "memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def get_user_history(user_id):
    memory = load_memory()
    return memory.get(user_id, [])


def append_user_message(user_id, sender, message):
    memory = load_memory()
    if user_id not in memory:
        memory[user_id] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory[user_id].append({"sender": sender, "message": message, "timestamp": timestamp})
    save_memory(memory)


def clear_user_history(user_id):
    memory = load_memory()
    memory[user_id] = []
    save_memory(memory)


def clear_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump({}, f)