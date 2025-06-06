from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")  # or multilingual model

app = FastAPI()
CACHE_FILE = "cache.json"

# Load cache from file if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

class Request(BaseModel):
    sentences: list[str]

@app.post("/embed")
def embed(req: Request):
    result = []
    changed = False

    for sentence in req.sentences:
        if sentence in embedding_cache:
            result.append(embedding_cache[sentence])
        else:
            vector = model.encode(sentence, convert_to_numpy=True).tolist()
            embedding_cache[sentence] = vector
            result.append(vector)
            changed = True

    # Save updated cache
    if changed:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(embedding_cache, f, ensure_ascii=False)

    return {"embeddings": result}

