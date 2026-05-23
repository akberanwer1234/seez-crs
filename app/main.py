from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .datasets.movie_dataset import load_movie_dataset
from .models import FewShotRecommender, RAGRecommender
from .models.dynamic_fewshot import DynamicFewShotRecommender
from .models.rag_enhanced import EnhancedRAGRecommender
from .agents import SimilarityAgent, PopularityAgent, RandomAgent, AgentBasedRecommender
from .agents.enhanced_agent_system import PreferenceAwareAgentRecommender
from .multi_agent import MultiAgentRecommender


app = FastAPI(title="LLM‑Redial Conversational Recommender")

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

# Instantiate improved models.  These use the full dataset rather
# than the ``examples`` list because they perform their own
# preprocessing.  ``dataset`` contains examples with item IDs;
# ``examples`` contains (text, rec_titles) pairs.  We pass
# ``dataset`` so the models can access rec_item IDs for weighting.
dynamic_fewshot_model = DynamicFewShotRecommender(dataset, item_map=ITEM_MAP, n_examples=5)
enhanced_rag_model = EnhancedRAGRecommender(dataset, item_map=ITEM_MAP, top_k=20)
preference_agent_model = PreferenceAwareAgentRecommender(dataset, item_map=ITEM_MAP, top_k=20)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def _stream_response(model_name: str, question: str, history: List[str]):
    # Select model based on the provided system name.
    if model_name == "fewshot":
        model = fewshot_model
    elif model_name == "rag":
        model = rag_model
    elif model_name == "agent":
        model = agent_model
    elif model_name == "multi":
        model = multi_agent_model
    elif model_name == "dynamic_fewshot":
        model = dynamic_fewshot_model
    elif model_name == "enhanced_rag":
        model = enhanced_rag_model
    elif model_name == "preference_agent":
        model = preference_agent_model
    else:
        raise HTTPException(status_code=404, detail=f"Unknown recommender '{model_name}'")
    # Invoke the model.  Some models return only a title, whereas the
    # preference agent returns a tuple of (title, explanation).
    try:
        result = await model.recommend(question, history)
    except Exception as exc:
        yield f"\n[ERROR] Model error: {exc}"
        return
    if isinstance(result, tuple):
        rec_title, explanation = result
    else:
        rec_title, explanation = result, None
    # Stream the title
    for ch in rec_title:
        yield ch
        await asyncio.sleep(0)
    # If an explanation is provided, stream it on a new line
    if explanation:
        for ch in "\n" + explanation:
            yield ch
            await asyncio.sleep(0)


@app.post("/recommend/{system}")
async def recommend(system: str, question: str, history: Optional[List[str]] = None):
    if history is None:
        history = []
    generator = _stream_response(system, question, history)
    return StreamingResponse(generator, media_type="text/plain")
