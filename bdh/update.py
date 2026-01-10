# bdh/update.py
from bdh.state import MemoryState

def update_memory(entry, memory_path="outputs/memory.json"):
    """
    entry = {
        "story_name": str,
        "character": str,
        "snippet": str,
        "prediction": int,
        "confidence": float
    }
    """
    memory = MemoryState(memory_path)
    key = f"{entry['story_name']}_{entry['character']}"
    memory.add(key, entry)
