import os
import sys
import json
import random
import uuid
from pathlib import Path
from dotenv import load_dotenv


def load_data(data_path: str = "reference/character_data.json"):
    """Loads character attribute pools from a JSON file."""
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)


def _is_dict_of_lists(values_dict: dict) -> bool:
    """Check if a dict's values are all lists (nested config like conversation_style)."""
    sample_values = list(values_dict.values())
    return bool(sample_values) and isinstance(sample_values[0], list)


def sample_list_values(values: list, counter: int, is_strict: bool) -> str:
    """Sample from a flat list of string options.

    Strict mode: pick exactly `counter` items.
    Non-strict mode: pick 1..counter items randomly.
    """
    if is_strict:
        sample_size = min(counter, len(values))
        picked = random.sample(values, sample_size)
    else:
        sample_size = random.randint(1, min(counter, len(values)))
        picked = random.sample(values, sample_size)
    return ", ".join(picked)


def parse_dict_of_lists(values_dict: dict, counter: int) -> str:
    """Parse a dict-of-lists config (e.g. conversation_style).

    All keys preserved; one value sampled per key's list.
    Output: [key1: val1; key2: val2; ...]
    """
    pairs = [
        (k, random.sample(v_list, min(counter, len(v_list)))[0])
        for k, v_list in values_dict.items()
    ]
    return f"[{'; '.join(f'{k}: {v}' for k, v in pairs)}]"


def parse_dict_pool(values_dict: dict, counter: int) -> str:
    """Parse a flat dict pool config.

    All keys preserved; values sampled without replacement.
    Output: [key1: val1; key2: val2; ...]
    """
    keys = list(values_dict.keys())
    pool = list(values_dict.values())
    sample_size = min(counter, len(pool))
    picked_indices = random.sample(range(len(pool)), sample_size)
    picked_pairs = [(keys[i], pool[i]) for i in picked_indices]
    return f"[{'; '.join(f'{k}: {v}' for k, v in picked_pairs)}]"


def parse_values_dict(values_dict: dict, counter: int) -> str:
    """Dispatch on the type of values inside a dict config."""
    if _is_dict_of_lists(values_dict):
        return parse_dict_of_lists(values_dict, counter)
    return parse_dict_pool(values_dict, counter)


def resolve_attribute(key: str, config) -> str:
    """Resolve a single attribute config using match-like dispatch."""
    match config:
        case {"values": list() as values_list}:
            return sample_list_values(
                values_list,
                config.get("counter", 1),
                config.get("is_strict", True),
            )
        case {"values": dict() as values_dict}:
            return parse_values_dict(values_dict, config.get("counter", 1))
        case {"values": _}:
            return str(config["values"])
        case list() as items:
            return random.choice(items)
        case _:
            return str(config)


def build_character(data: dict) -> dict:
    """Build a character by resolving all attributes."""
    return {key: resolve_attribute(key, config) for key, config in data.items()}


def format_plist(attributes: dict) -> str:
    """Format attributes dict as a PList string."""
    elements = [f"{k}: {v}" for k, v in attributes.items()]
    return f"[{'; '.join(elements)}]"


def derive_filename(attributes: dict) -> str:
    """Derive a filename from the character's name attribute."""
    name_val = None
    for key, value in attributes.items():
        if "name" in key.lower():
            name_val = value
            break

    if name_val:
        clean_name = name_val.split(",")[0].strip()
        name_parts = clean_name.split()
        if len(name_parts) >= 2:
            return f"{name_parts[0].lower()}_{name_parts[-1].lower()}.txt"
        return f"{clean_name.lower().replace(' ', '_')}.txt"
    return f"character_{uuid.uuid4().hex[:8]}.txt"


def save_character(content: str, characters_dir: str, filename: str) -> Path:
    """Write the character file, avoiding name collisions."""
    output_dir = Path(characters_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    if file_path.exists():
        new_name = f"{file_path.stem}_{uuid.uuid4().hex[:8]}{file_path.suffix}"
        file_path = file_path.with_name(new_name)

    file_path.write_text(content, encoding="utf-8")
    return file_path


def generate_character(characters_dir: str):
    """Generate a character and save it to disk."""
    data = load_data()
    attributes = build_character(data)
    content = format_plist(attributes)
    filename = derive_filename(attributes)
    file_path = save_character(content, characters_dir, filename)
    name_val = attributes.get("name", "Unknown")
    print(f"Successfully generated character: {name_val}")
    print(f"Saved to: {file_path}")


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = os.getenv("CHARACTERS_DIR", "data/characters/ssot")

    generate_character(target_dir)
