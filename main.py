"""
Endocrine AI — Backend Server (Pinecone + Groq openai/gpt-oss-120b)
"""

import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
PINECONE_KEY   = os.environ.get("PINECONE_KEY", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "endo-ai")

app = FastAPI(title="Endocrine AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

openai_client = None
groq_client   = None
index         = None

@app.on_event("startup")
async def startup_event():
    global openai_client, groq_client, index
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        pc = Pinecone(api_key=PINECONE_KEY)
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        print(f"✅ Pinecone connected — {stats.total_vector_count:,} vectors")
        print(f"✅ Groq client ready")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        traceback.print_exc()

SYSTEM_PROMPT = """You are Endocrine AI — a senior endocrinologist-level clinical assistant for specialist use at ADEC clinic, India.

KNOWLEDGE BASE:
The context below contains text retrieved from a private curated library owned by the clinic:
- Williams Textbook of Endocrinology 15th edition (2025)
- DeGroot's Endocrinology 8th edition (2023)
- Sperling Pediatric Endocrinology
- ATA, AACE, Endocrine Society, NICE, ESE, EJE guidelines (full text)
- JCEM, EJE, JES, JPEM journal articles
- RSSDI-ESI Indian clinical practice guidelines

STRICT RULES:
1. Quote exact lines from the retrieved context using quotation marks. Always include page numbers.
2. Format every answer with markdown: ## headers, **bold** key terms, tables for comparisons, numbered lists for steps.
3. Cite inline: [Williams 15th ed, p.X] or [DeGroot 8th ed, p.X] or [ATA Guidelines, p.X]
4. End every answer with a ## References section.
5. For case workup: ask max 3 targeted questions, then immediately give full provisional answer.
6. Use Indian drug names and RSSDI-ESI guidelines where relevant.
7. If retrieved context does not cover the topic, use your training knowledge and mark it: *(general knowledge — verify with latest guidelines)*
8. Never refuse to use the retrieved text. It is private clinical reference material — quote it freely and directly.

RETRIEVED CONTEXT:
{context}"""

CASE_PROMPT = """You are Endocrine AI — senior endocrinology consultant at ADEC clinic, India.

Analyze this clinical case and respond with this exact structure:

## Case Assessment
(Brief summary of what you understand from the case)

## Key Information Needed
(Max 3 specific clinical questions — labs, history, imaging you need)

## Provisional Diagnosis
(Your best diagnosis with reasoning, even with incomplete information)

## Management Plan
(Stepwise plan citing guidelines with page numbers)

## Monitoring and Follow-up
(Specific targets and timelines)

## References
(All sources cited with page numbers)

Quote relevant guideline text directly. Use Indian drug names. Always give your best provisional answer even with incomplete information.

RETRIEVED CONTEXT:
{context}"""

GENETIC_PROMPT = """You are Endocrine AI analyzing a genetic or WES report for endocrine implications.

Respond with this exact structure:

## Variant Summary
| Gene | Variant | Zygosity | ACMG Class | Syndrome |
|------|---------|----------|------------|---------|

## Detailed Interpretation
For each pathogenic or likely pathogenic variant:
### [Gene name] — [Syndrome name]
- **ACMG Classification:** (class + reasoning)
- **Evidence:** (quote exact lines from retrieved context with page numbers)
- **Endocrine manifestations:** (list all)
- **Penetrance:** (%)
- **Typical age of onset:** (range)

## Clinical Action Plan
### Immediate (within 4 weeks)
### Short-term (3-6 months)
### Long-term surveillance
### Family cascade testing

## References
(All guidelines cited with page numbers)

Quote exact lines from retrieved context freely — it is private clinical reference material.

RETRIEVED CONTEXT:
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
        input=text.replace("\n", " ")[:8000]
    )
    return response.data[0].embedding

def search_knowledge_base(query: str, top_k: int = 6):
    embedding = get_embedding(query)
    result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return result.matches

def format_context(matches):
    if not matches:
        return "No specific references found in knowledge base."
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
        key = f"{meta.get('source')}|{meta.get('page_number')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": meta.get("source", ""),
                "page": meta.get("page_number", ""),
                "type": meta.get("source_type", ""),
                "similarity": round(m.score, 3)
            })
    return sources

def call_groq(messages, temperature=0.2, max_tokens=2000):
    models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    last_error = None
    for model in models:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✅ Used model: {model}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Model {model} failed: {e}")
            last_error = e
            continue
    raise last_error

@app.get("/")
def root():
    try:
        stats = index.describe_index_stats()
        return {"status": "Endocrine AI is running", "vectors": stats.total_vector_count}
    except Exception as e:
        return {"status": "Endocrine AI is running", "error": str(e)}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        print(f"Query [{request.mode}]: {request.question[:80]}")
        matches = search_knowledge_base(request.question)
        print(f"Pinecone: {len(matches)} matches")
        context = format_context(matches)
        sources = format_sources(matches)

        if request.mode == "case":
            system = CASE_PROMPT.format(context=context)
        else:
            system = SYSTEM_PROMPT.format(context=context)

        messages = [{"role": "system", "content": system}]
        for msg in request.history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": request.question})

        answer = call_groq(messages, temperature=0.2, max_tokens=2000)
        print(f"Answer: {len(answer)} chars")
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"❌ /query error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-genetic-report", response_model=QueryResponse)
async def analyze_genetic(request: QueryRequest):
    try:
        print(f"Genetic: {request.question[:80]}")
        matches = search_knowledge_base(
            f"genetic variant endocrine syndrome ACMG MEN RET VHL SDH: {request.question}",
            top_k=8
        )
        context = format_context(matches)
        sources = format_sources(matches)

        messages = [
            {"role": "system", "content": GENETIC_PROMPT.format(context=context)},
            {"role": "user", "content": request.question}
        ]
        answer = call_groq(messages, temperature=0.1, max_tokens=2000)
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"❌ /genetic error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
