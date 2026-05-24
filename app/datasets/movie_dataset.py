from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

def _parse_conversations(conversation_path: Path, max_conversations: Optional[int] = None) -> Dict[int, str]:

    conversations: Dict[int, str] = {}
    current_id: Optional[int] = None
    current_lines: List[str] = []
    count = 0
    with open(conversation_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # A line consisting solely of digits starts a new conversation
            if stripped.isdigit():
                # Finalise the previous conversation, if any
                if current_id is not None:
                    conversations[current_id] = "\n".join(current_lines).strip()
                    current_lines = []
                    count += 1
                    if max_conversations is not None and count >= max_conversations:
                        break
                # Begin a new conversation
                try:
                    current_id = int(stripped)
                except ValueError:
                    current_id = None
                continue
            # Otherwise accumulate non‑empty lines as part of the current conversation
            if current_id is not None and stripped:
                current_lines.append(stripped)
        # Finalise the last conversation if it exists
        if current_id is not None and (max_conversations is None or count < max_conversations):
            conversations[current_id] = "\n".join(current_lines).strip()
    return conversations


def _parse_final_data(final_data_path: Path) -> Dict[int, List[str]]:

    mapping: Dict[int, List[str]] = {}
    with open(final_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            for user_data in data.values():
                conversations = user_data.get("Conversation", [])
                for conv in conversations:
                    conv_data = next(iter(conv.values()))
                    conv_id = conv_data.get("conversation_id")
                    rec_items = conv_data.get("rec_item", [])
                    if conv_id is not None:
                        mapping[conv_id] = rec_items
    return mapping


def load_movie_dataset(data_dir: str, max_conversations: Optional[int] = None) -> List[Dict[str, object]]:

    data_path = Path(data_dir)
    conversation_path = data_path / "Conversation.txt"
    final_data_path = data_path / "final_data.jsonl"
    conv_map = _parse_conversations(conversation_path, max_conversations)
    rec_map = _parse_final_data(final_data_path)
    examples = []
    for conv_id, text in conv_map.items():
        if conv_id in rec_map and text:
            examples.append({
                "conversation_id": conv_id,
                "text": text,
                "rec_items": rec_map[conv_id],
            })
    return examples
