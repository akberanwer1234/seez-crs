# Seez CRS Technical Test

Conversational recommender system for the LLM-REDIAL movie domain, exposed through a FastAPI service.

I kept this implementation lightweight so it can be run locally without a GPU, external API keys, or a vector database. The recommendation approaches use TF-IDF retrieval/similarity as the local baseline. In a production version, I would replace the retrieval layer with embeddings + a vector store and use an LLM for response generation and ranking.

## What is included

- Dataset loader for the LLM-REDIAL movie files
- Few-shot style recommender
- RAG-style recommender
- Agent-based recommender
- Multi-agent recommender
- FastAPI service with streaming responses
- `requirements.txt`
- Instructions to run and verify the endpoints

## Project structure

```text
seez-crs/
├── app/
│   ├── main.py
│   ├── datasets/
│   │   └── movie_dataset.py
│   ├── models/
│   │   ├── fewshot.py
│   │   └── rag.py
│   ├── agents/
│   │   └── agent_system.py
│   └── multi_agent/
│       └── orchestrator.py
├── llm_redial/
│   ├── Tools.py
│   ├── read_me.py
│   └── LLM_Redial/data/Movie/
├── requirements.txt
└── README.md
```

## Approaches implemented

### Few-shot

Location: `app/models/fewshot.py`

Uses a small set of example conversations and selects the closest example with TF-IDF cosine similarity. The recommendation attached to that example is returned.

Endpoint:

```text
POST /recommend/fewshot
```

### RAG

Location: `app/models/rag.py`

Indexes the available conversations, retrieves the top-k most similar dialogues, and returns the most common recommended movie among the retrieved examples.

Endpoint:

```text
POST /recommend/rag
```

### Agent-based system

Location: `app/agents/agent_system.py`

Contains simple agents with different recommendation strategies:

- `SimilarityAgent`
- `PopularityAgent`
- `RandomAgent`

Endpoint:

```text
POST /recommend/agent
```

### Multi-agent system

Location: `app/multi_agent/orchestrator.py`

Runs multiple agents and combines their outputs with majority voting.

Endpoint:

```text
POST /recommend/multi
```

## Async and performance handling

The FastAPI endpoints are asynchronous and return `StreamingResponse` objects. The response generator yields control back to the event loop while streaming, so multiple requests can be handled concurrently by Uvicorn.

Relevant file:

```text
app/main.py
```

This is intentionally simple for the test scope. For a heavier model, I would move blocking inference into a thread/process pool or use an async queue for batching.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset location

The app expects the movie dataset here:

```text
llm_redial/LLM_Redial/data/Movie
```

Required files:

```text
Conversation.txt
final_data.jsonl
item_map.json
user_ids.json
```

The original dataset package also includes `Tools.py` and `read_me.py`. Those are helper/example files from the dataset authors. I did not depend on them directly because the project has its own loader in `app/datasets/movie_dataset.py`.

## Run the API

Create a venv in project root:

```bash
python -m venv venv
venv\Scripts\activate     # Activate it
```

From the project root:

```bash
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Example requests

Few-shot:

```bash
curl -X POST "http://127.0.0.1:8000/recommend/fewshot?question=I%20liked%20Inception%20and%20want%20another%20mind-bending%20movie"
```

RAG:

```bash
curl -X POST "http://127.0.0.1:8000/recommend/rag?question=I%20enjoyed%20Star%20Wars%20and%20want%20more%20sci-fi"
```

Agent-based:

```bash
curl -X POST "http://127.0.0.1:8000/recommend/agent?question=I%20want%20a%20funny%20family%20movie"
```

Multi-agent:

```bash
curl -X POST "http://127.0.0.1:8000/recommend/multi?question=Recommend%20a%20classic%20action%20movie"
```

## Verifying concurrency

Start the server, then run several requests in parallel:

```bash
curl -X POST "http://127.0.0.1:8000/recommend/rag?question=I%20liked%20The%20Matrix" &
curl -X POST "http://127.0.0.1:8000/recommend/agent?question=I%20want%20a%20comedy" &
curl -X POST "http://127.0.0.1:8000/recommend/multi?question=I%20want%20a%20thriller" &
wait
```

The responses should stream independently.

## Prompt changes that can improve recommendation accuracy

These changes are useful when replacing the lightweight baseline with an LLM-backed generator:

1. **Add a short preference summary before the conversation.**

   Example:

   ```text
   User preferences:
   - Likes sci-fi and psychological thrillers
   - Dislikes slow dramas
   - Recently enjoyed Inception and The Matrix
   ```

   This gives the model a cleaner signal than asking it to infer everything from the full dialogue.

2. **Use labelled few-shot examples.**

   Example:

   ```text
   Conversation:
   User: I enjoyed Star Wars and want another space adventure.

   Recommended movie:
   Star Wars: The Clone Wars
   ```

   Clear labels reduce ambiguity and make the expected output format easier to follow.

## Notes and limitations

- TF-IDF was used to keep the project easy to run locally.
- The current implementation returns movie titles, not long natural-language explanations.
- The multi-agent setup is intentionally simple, but the structure makes it easy to add stronger agents later.
- For production, I would add embeddings, a vector database, caching, logging, evaluation metrics, and integration tests.
