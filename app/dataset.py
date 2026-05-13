"""
Dataset loading utilities for LLM‑Redial movie data.

The movie portion of LLM‑Redial is delivered as a set of two files:

1. ``Conversation.txt`` — a plain text file containing one conversation per
   index. Each conversation begins with a numeric identifier on its own
   line followed by alternating ``User:`` and ``Agent:`` utterances. Blank
   lines separate conversations. The numeric identifiers are used as
   ``conversation_id`` values throughout the accompanying metadata.
2. ``final_data.jsonl`` — a JSON lines file where each line is a mapping
   from a user identifier to that user’s data. Each user entry contains a
   ``Conversation`` list holding one or more conversations. Within each
   conversation object the ``conversation_id`` key corresponds to the
   numeric identifier in ``Conversation.txt`` and ``rec_item`` contains
   the recommended item(s) for that conversation.

This module parses these files and exposes a unified data structure
associating conversation text with its recommended items. The loader
optionally accepts a ``max_conversations`` parameter to limit the
number of conversations parsed (useful when prototyping or running
within constrained resources).

Example usage::

    from dataset import load_movie_dataset

    examples = load_movie_dataset("/path/to/movie/data", max_conversations=100)
    print(examples[0]["text"])
    print(examples[0]["rec_items"])  # list of recommended items

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_conversations(conversation_path: Path, max_conversations: Optional[int] = None) -> Dict[int, str]:
    """Parse conversation transcripts from ``Conversation.txt``.

    Each conversation starts with a numeric ID on its own line. The
    subsequent lines up to the next blank line form the conversation
    transcript. The returned dictionary maps ``conversation_id`` to
    the raw conversation text (with newline separators between turns).

    Args:
        conversation_path: Path to the ``Conversation.txt`` file.
        max_conversations: Optional integer to limit the number of
            conversations parsed. If ``None`` all conversations will be read.

    Returns:
        A dictionary mapping conversation IDs to their corresponding
        conversation text.
    """
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
    """Parse the JSONL ``final_data.jsonl`` file to extract recommended items.

    The file contains one JSON object per line. Each object maps a
    single user ID to a dictionary with a ``Conversation`` list. Each
    conversation entry has a ``conversation_id`` and a ``rec_item`` list
    of recommended item identifiers. This function flattens the data
    into a mapping from ``conversation_id`` to the associated list of
    recommended item identifiers.

    Args:
        final_data_path: Path to the ``final_data.jsonl`` file.

    Returns:
        A dictionary mapping conversation IDs to lists of recommended
        item identifiers.
    """
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
    """Load the movie portion of the LLM‑Redial dataset.

    This function combines the parsed conversation transcripts with
    recommended item metadata. It returns a list of dictionaries where
    each dictionary represents one conversation example with the
    following keys:

    - ``conversation_id`` (int): the unique identifier of the conversation
    - ``text`` (str): the full conversation transcript
    - ``rec_items`` (List[str]): list of recommended item identifiers

    Args:
        data_dir: Path to the ``Movie`` directory containing
            ``Conversation.txt`` and ``final_data.jsonl``.
        max_conversations: Optional integer to limit the number of
            conversations loaded. When set, the loader will stop after
            reading up to ``max_conversations`` conversations. Note that
            conversations without corresponding recommended items may be
            skipped.

    Returns:
        A list of example dictionaries as described above.
    """
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
