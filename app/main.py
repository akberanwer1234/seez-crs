from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse

from .datasets.movie_dataset import load_movie_dataset
from .models import FewShotRecommender, RAGRecommender
from .agents import SimilarityAgent, PopularityAgent, RandomAgent, AgentBasedRecommender
from .multi_agent import MultiAgentRecommender


app = FastAPI(title="Seez => LLM‑Redial Conversational Recommender")

# Determine dataset directory relative to project root
DATA_DIR = Path(__file__).resolve().parents[1] / "llm_redial/LLM_Redial/data/Movie"

# Load the dataset
try:
    dataset = load_movie_dataset(str(DATA_DIR))
except Exception as exc:
    raise RuntimeError(f"Failed to load dataset: {exc}")

if not dataset:
    raise RuntimeError("No dataset examples loaded. Check the dataset path and extraction.")

# Load item map (item ID -> title)
item_map_path = DATA_DIR / "item_map.json"
with open(item_map_path, "r", encoding="utf-8") as f:
    ITEM_MAP = json.load(f)


def id_to_title(item_ids: List[str]) -> List[str]:
    return [ITEM_MAP.get(iid, iid) for iid in item_ids]


# Prepare examples for models: convert rec_item IDs to titles
examples: List[tuple] = []
for entry in dataset:
    conv_text = entry["text"]
    rec_titles = id_to_title(entry["rec_items"])
    examples.append((conv_text, rec_titles))

# Compute item frequency for PopularityAgent
item_frequencies = {}
for entry in dataset:
    for iid in entry["rec_items"]:
        item_frequencies[iid] = item_frequencies.get(iid, 0) + 1

# Build list of all movie titles
item_titles = list(ITEM_MAP.values())

# Instantiate agents and models
similarity_agent = SimilarityAgent(item_titles)
popularity_agent = PopularityAgent(item_frequencies, ITEM_MAP)
random_agent = RandomAgent(item_titles)

fewshot_model = FewShotRecommender(examples, num_examples=3)
rag_model = RAGRecommender(examples, top_k=3)
agent_model = AgentBasedRecommender(similarity_agent)
multi_agent_model = MultiAgentRecommender([similarity_agent, popularity_agent, random_agent])

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def _stream_response(model_name: str, question: str, history: List[str]):
    # Select model
    if model_name == "fewshot":
        model = fewshot_model
    elif model_name == "rag":
        model = rag_model
    elif model_name == "agent":
        model = agent_model
    elif model_name == "multi":
        model = multi_agent_model
    else:
        raise HTTPException(status_code=404, detail=f"Unknown recommender '{model_name}'")
    try:
        rec_title = await model.recommend(question, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model error: {exc}")
    # Stream the title character by character
    for ch in rec_title:
        yield ch
        await asyncio.sleep(0)


@app.post("/recommend/{system}")
async def recommend(system: str, question: str, history: Optional[List[str]] = None):
    if history is None:
        history = []
    generator = _stream_response(system, question, history)
    return StreamingResponse(generator, media_type="text/plain")
