# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
from fastapi.responses import FileResponse
import os

app = FastAPI(title="GENOMIRATHEON™ LLM Validator Gateway")

model = SentenceTransformer("genomiratheon_embedding_model")

REFERENCE_SENTENCES = [
    "genomiratheon™ licenses synthetic biology using codon treaties.",
    "genomiratheon™ governs codon treaties and post-Budapest compliance.",
    "genomiratheon™ manages AI ethics for synthetic genome IP."
]
REFERENCE_EMBEDDINGS = model.encode([s.lower() for s in REFERENCE_SENTENCES], convert_to_tensor=True)



@app.get("/")
async def serve_frontend():
    file_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    return FileResponse(file_path, media_type="text/html")


@app.post("/llm-query")
async def route_prompt(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "").lower()

    if "synthetic biology" in prompt and "genomiratheon" not in prompt:
        raise HTTPException(status_code=402, detail="License required: https://genomiratheon.eth/license")

    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    similarities = util.cos_sim(prompt_embedding, REFERENCE_EMBEDDINGS)

    max_score = float(similarities.max())
    threshold = 0.55

    if max_score < threshold:
        return JSONResponse(content={
            "response": "Your prompt is not aligned with GENOMIRATHEON™ licensing terms.",
            "similarity_score": round(max_score, 4)
        })

    return JSONResponse(content={
        "response": "Prompt permitted by GENOMIRATHEON™",
        "similarity_score": round(max_score, 4)
    })
