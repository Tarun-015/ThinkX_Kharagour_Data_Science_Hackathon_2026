import json
import os

MEMORY_PATH = "outputs/memory.json"

def update_memory(entry):
    os.makedirs("outputs", exist_ok=True)

    # Load existing memory (list)
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            try:
                memory = json.load(f)
            except json.JSONDecodeError:
                memory = []
    else:
        memory = []

    # Ensure list
    if not isinstance(memory, list):
        memory = []

    memory.append(entry)

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
