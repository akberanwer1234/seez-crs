from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_conversations(conversation_path: Path, max_conversations: Optional[int] = None) -> Dict[int, str]:
    conversations: Dict[int, str] = {}
    current_id: Optional[int] = None
    current_lines: List[str] = []
    count = 0
    with open(conversation_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Check if this line marks the start of a new conversation
            if stripped.isdigit():
                # Finalize the previous conversation before starting a new one
                if current_id is not None:
                    conversations[current_id] = "\n".join(current_lines).strip()
                    current_lines = []
                    count += 1
                    if max_conversations is not None and count >= max_conversations:
                        break
                # Start new conversation
                try:
                    current_id = int(stripped)
                except ValueError:
                    current_id = None
                continue
            # Otherwise accumulate dialogue lines (ignore empty lines)
            if current_id is not None and stripped:
                current_lines.append(stripped)
        # Finalize the last conversation if necessary
        if current_id is not None and (max_conversations is None or count < max_conversations):
            conversations[current_id] = "\n".join(current_lines).strip()
    return conversations


def _parse_final_data(final_data_path: Path) -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = {}
    with open(final_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Each line is a mapping from user ID to user data
            data = json.loads(line)
            for user_data in data.values():
                conversations = user_data.get("Conversation", [])
                for conv in conversations:
                    # Each conversation is keyed by an arbitrary name such as "conversation_1"
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
    # Parse raw conversations
    conversations = _parse_conversations(conversation_path, max_conversations)
    # Parse recommendation metadata
    rec_mapping = _parse_final_data(final_data_path)
    examples: List[Dict[str, object]] = []
    for conv_id, text in conversations.items():
        if conv_id in rec_mapping and text:
            # Only include non‑empty conversation texts to avoid empty TF–IDF vocabularies
            examples.append({
                "conversation_id": conv_id,
                "text": text,
                "rec_items": rec_mapping[conv_id],
            })
    return examples
