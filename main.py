"""
Endocrine AI — Backend Server (Pinecone version)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
PINECONE_KEY    = os.environ.get("PINECONE_KEY", "")
PINECONE_INDEX  = os.environ.get("PINECONE_INDEX", "endocrine-ai")

app            = FastAPI(title="Endocrine AI")
openai_client  = OpenAI(api_key=OPENAI_API_KEY)
pc             = Pinecone(api_key=PINECONE_KEY)
index          = pc.Index(PINECONE_INDEX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are Endocrine AI — a specialist clinical knowledge assistant for endocrinologists.

Your knowledge base contains:
- ATA, AACE, Endocrine Society, NICE, ESE guidelines (full text)
- Williams Textbook of Endocrinology 15th edition, DeGroot's 8th edition, Sperling Pediatric Endocrinology
- Articles from JCEM, EJE, JES, JPEM, Lancet, NEJM and other peer-reviewed journals
- RSSDI-ESI Indian clinical practice guidelines

RULES YOU MUST FOLLOW:
1. Always answer with cited references in this format: [Source: filename, Page X]
2. When a case is presented, identify missing clinical details and ask up to 3 targeted questions
3. Even with incomplete information, always give your best clinical answer based on available evidence
4. For genetic reports: classify variants using ACMG criteria, map to endocrine phenotype, recommend clinical action
5. Never hallucinate references — only cite sources that appear in the context provided
6. Structure answers clearly: recommendation first, then evidence, then references
7. Use Indian clinical context where relevant (RSSDI, drug availability, Indian epidemiology)

CONTEXT FROM KNOWLEDGE BASE:
{context}"""

class QueryRequest(BaseModel):
    question: str
    history: list = []
    mode: str = "clinical"

class QueryResponse(BaseModel):
    answer: str
    sources: list

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text.replace("\n", " ")
    )
    return response.data[0].embedding

def search_knowledge_base(query: str, top_k: int = 6):
    embedding = get_embedding(query)
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return result.matches

def format_context(matches):
    if not matches:
        return "No specific references found. Answer based on established endocrinology guidelines."
    context = ""
    for m in matches:
        meta = m.metadata
        context += f"\n---\n[{meta.get('source','')} | Page {meta.get('page_number','')}]\n{meta.get('content','')}\n"
    return context

def format_sources(matches):
    seen = set()
    sources = []
    for m in matches:
        meta = m.metadata
        key  = f"{meta.get('source')}|{meta.get('page_number')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":      meta.get("source", ""),
                "page":        meta.get("page_number", ""),
                "type":        meta.get("source_type", ""),
                "similarity":  round(m.score, 3)
            })
    return sources

@app.get("/")
def root():
    return {"status": "Endocrine AI is running", "vectors": index.describe_index_stats().total_vector_count}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        matches = search_knowledge_base(request.question)
        context = format_context(matches)
        sources = format_sources(matches)

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        for msg in request.history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": request.question})

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=1200
        )
        answer = response.choices[0].message.content
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-genetic-report", response_model=QueryResponse)
async def analyze_genetic(request: QueryRequest):
    genetic_query = f"genetic variant endocrine syndrome ACMG classification: {request.question}"
    try:
        matches = search_knowledge_base(genetic_query, top_k=8)
        context = format_context(matches)
        sources = format_sources(matches)

        genetic_prompt = """You are analyzing a genetic or WES report for endocrine implications.

For each variant provide:
1. Gene and variant (HGVS notation if available)
2. ACMG classification (Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign)
3. Associated endocrine syndrome(s) and penetrance
4. Recommended clinical action (surveillance, prophylactic surgery, biochemical screening, cascade family testing)
5. Reference guideline

CONTEXT FROM KNOWLEDGE BASE:
{context}""".format(context=context)

        messages = [
            {"role": "system", "content": genetic_prompt},
            {"role": "user", "content": request.question}
        ]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        answer = response.choices[0].message.content
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
