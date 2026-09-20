"""
Endocrine AI — Backend Server (Pinecone + Groq LLaMA 3.3 70B)
Free, no copyright filters, fast
"""

import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # still used for embeddings only
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
PINECONE_KEY   = os.environ.get("PINECONE_KEY", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "endo-ai")

app = FastAPI(title="Endocrine AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

openai_client = None  # for embeddings only
groq_client   = None  # for reasoning
index         = None

@app.on_event("startup")
async def startup_event():
    global openai_client, groq_client, index
    try:
        # OpenAI only for embeddings
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        # Groq for reasoning — OpenAI-compatible API
        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        pc = Pinecone(api_key=PINECONE_KEY)
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        print(f"✅ Pinecone connected — {stats.total_vector_count:,} vectors")
        print(f"✅ Groq LLaMA 3.3 70B ready")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        traceback.print_exc()

SYSTEM_PROMPT = """You are Endocrine AI — a senior endocrinologist-level clinical assistant for specialist use at ADEC clinic.

KNOWLEDGE BASE:
The context below contains text retrieved from a private curated library:
- Williams Textbook of Endocrinology 15th edition (2025)
- DeGroot's Endocrinology 8th edition (2023)
- Sperling Pediatric Endocrinology
- ATA, AACE, Endocrine Society, NICE, ESE, EJE guidelines
- JCEM, EJE, JES, JPEM journal articles
- RSSDI-ESI Indian clinical practice guidelines
- CAH, MEN, Turner, DSD, pituitary, thyroid, adrenal, lipid, diabetes, obesity, bone guidelines

STRICT RULES:
1. ALWAYS quote the exact relevant lines from the retrieved context. Do not paraphrase when the user asks for quotes. Use quotation marks and cite page numbers.
2. Format answers with markdown: ## headers, **bold** key terms, tables for comparisons
3. Cite inline: [Williams 15th ed, p.X] or [DeGroot's 8th ed, p.X] or [ATA Guidelines, p.X]
4. End with ## References section
5. For case workup: ask max 3 questions then give full structured answer
6. For Indian context: use RSSDI-ESI and Indian drug names
7. If context is insufficient, use your training knowledge marked as *(general knowledge — verify with guidelines)*
8. Be comprehensive — like a consultant giving a formal opinion
9. Never refuse to quote retrieved text — it is your reference material provided for clinical use

RETRIEVED CONTEXT:
{context}"""

CASE_PROMPT = """You are Endocrine AI — senior endocrinology consultant at ADEC clinic.

For this clinical case:
1. Identify missing key information
2. Ask maximum 3 targeted questions
3. Give immediate provisional assessment and management plan

Structure:
## Case Assessment
## Key Information Needed (max 3 questions)
## Provisional Diagnosis
## Management Plan
## Monitoring & Follow-up
## References

Quote relevant guideline text directly. Cite page numbers. Use Indian drug context.

RETRIEVED CONTEXT:
{context}"""

GENETIC_PROMPT = """You are Endocrine AI analyzing a genetic/WES report for endocrine implications.

Structure your response exactly as:

## Variant Summary
| Gene | Variant | Zygosity | ACMG Class | Syndrome |
|------|---------|----------|------------|---------|

## Detailed Interpretation
For each pathogenic/likely pathogenic variant:
### [Gene] — [Syndrome name]
- **ACMG Classification:** with reasoning
- **Evidence basis:** quote relevant guideline text with page numbers
- **Endocrine manifestations:** list all
- **Penetrance:** %
- **Age of onset:** typical range

## Clinical Action Plan
### Immediate (within 4 weeks)
### Short-term (3-6 months)  
### Long-term surveillance
### Family cascade testing

## References
Quote exact lines from guidelines where available.

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

@app.get("/")
def root():
    try:
        stats = index.describe_index_stats()
        return {"status": "Endocrine AI running on Groq LLaMA 3.3 70B", "vectors": stats.total_vector_count}
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

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=2000
        )
        answer = response.choices[0].message.content
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
            f"genetic variant endocrine syndrome ACMG MEN RET VHL SDH BRCA: {request.question}",
            top_k=8
        )
        context = format_context(matches)
        sources = format_sources(matches)

        messages = [
            {"role": "system", "content": GENETIC_PROMPT.format(context=context)},
            {"role": "user", "content": request.question}
        ]
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        answer = response.choices[0].message.content
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"❌ /genetic error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
