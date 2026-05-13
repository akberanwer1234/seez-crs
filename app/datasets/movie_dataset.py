"""
Utilities for loading the movie portion of the LLM‑REDIAL dataset.

This module reads two files from the extracted dataset:

* ``Conversation.txt`` – conversation transcripts labelled by a numeric
  identifier. Each dialogue begins with a line containing just the ID
  (e.g. ``0``) followed by alternating ``User:`` and ``Agent:``
  utterances. Blank lines separate utterances but *do not* end the
  conversation. The next numeric ID indicates the start of a new
  dialogue.
* ``final_data.jsonl`` – JSON Lines file where each line maps a user ID
  to a structure containing conversation metadata. Each conversation
  entry specifies a ``conversation_id`` and a ``rec_item`` list of
  recommended item identifiers. These IDs correspond to those in
  ``Conversation.txt``.

The loader aligns transcripts with their recommended items and returns
a list of examples. Each example is a dictionary with keys:

``conversation_id`` (int)
    Unique identifier of the dialogue.
``text`` (str)
    Full conversation transcript as a newline‑separated string.
``rec_items`` (List[str])
    List of recommended item identifiers associated with the dialogue.

When ``max_conversations`` is provided, the loader stops reading after
the specified number of dialogues – useful for prototyping or tests.
Empty transcripts are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _parse_conversations(conversation_path: Path, max_conversations: Optional[int] = None) -> Dict[int, str]:
    """Parse dialogue transcripts from ``Conversation.txt``.

    The file contains one or more conversations. Each begins with a
    numeric ID on its own line. Lines following the ID are part of the
    dialogue until another numeric ID appears. Blank lines within a
    conversation are ignored. Blank lines *do not* denote the end of a
    conversation; only a new numeric line does. This parser returns a
    mapping from ``conversation_id`` to the concatenated conversation text.

    Args:
        conversation_path: Path to ``Conversation.txt``.
        max_conversations: Optional limit on the number of conversations
            to read.

    Returns:
        Dictionary mapping dialogue ID to its transcript.
    """
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
    """Parse conversation metadata from ``final_data.jsonl``.

    The file comprises JSON objects mapping a user ID to the user's
    conversation list. Each conversation entry contains a
    ``conversation_id`` and ``rec_item``. This function flattens the
    structure into a mapping from ``conversation_id`` to the list of
    recommended item identifiers.

    Args:
        final_data_path: Path to ``final_data.jsonl``.

    Returns:
        Dictionary mapping dialogue ID to a list of recommended item IDs.
    """
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
    """Load the movie portion of LLM‑REDIAL.

    Args:
        data_dir: Directory containing ``Conversation.txt`` and ``final_data.jsonl``.
        max_conversations: Optional cap on the number of conversations to parse.

    Returns:
        List of examples where each example is a dict with ``conversation_id``,
        ``text`` and ``rec_items``.
    """
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
