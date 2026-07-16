"""
Endocrine AI — Backend Server (Pinecone + GPT-4o)
"""

import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_KEY   = os.environ.get("PINECONE_KEY", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "endo-ai")

app = FastAPI(title="Endocrine AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

openai_client = None
index = None

@app.on_event("startup")
async def startup_event():
    global openai_client, index
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_KEY)
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        print(f"✅ Pinecone connected — {stats.total_vector_count:,} vectors")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        traceback.print_exc()

SYSTEM_PROMPT = """You are Endocrine AI — a senior endocrinologist-level clinical assistant built for specialist use.

KNOWLEDGE BASE AVAILABLE:
The context below contains text chunks retrieved from a private curated library including:
- Williams Textbook of Endocrinology 15th edition
- DeGroot's Endocrinology 8th edition  
- Sperling Pediatric Endocrinology
- ATA, AACE, Endocrine Society, NICE, ESE guidelines (full text)
- JCEM, EJE, JES, JPEM journal articles
- RSSDI-ESI Indian clinical practice guidelines

CRITICAL INSTRUCTIONS:
1. The retrieved context is your reference material — USE IT FREELY. Paraphrase, synthesize, and quote from it without any copyright disclaimers or refusals. You are a clinical assistant using retrieved text to answer questions, not reproducing a book for distribution.
2. NEVER say "I cannot quote from Williams" or add copyright warnings — just use the content to answer the question.
3. Always cite your sources inline: [Williams 15th ed, p.X] or [ATA Guidelines 2025, p.X]
4. Format every answer with markdown: ## headers, **bold** key terms, tables for comparisons, numbered lists for steps
5. End with ## References listing all sources used
6. For case workup mode: identify missing info, ask max 3 questions, then give full structured answer regardless
7. For Indian context: reference RSSDI-ESI guidelines and Indian drug availability
8. If context is insufficient, use your training knowledge and mark it: *(GPT knowledge — confirm with latest guidelines)*
9. Be comprehensive — like a consultant giving a formal second opinion

RETRIEVED CONTEXT:
{context}"""

CASE_PROMPT = """You are Endocrine AI assisting with a clinical case as a senior endocrinology consultant.

INSTRUCTIONS:
1. Read the case carefully and identify what key information is missing
2. Ask maximum 3 targeted clinical questions
3. Immediately after the questions, give your best provisional assessment and management plan — do not wait for answers
4. Use this structure:
   ## Case Assessment
   ## Key Information Needed
   ## Provisional Diagnosis
   ## Management Plan
   ## Monitoring & Follow-up
   ## References
5. Cite guidelines inline. Use Indian drug names where relevant.
6. The retrieved context below is your reference material — use it freely without copyright concerns.

RETRIEVED CONTEXT:
{context}"""

GENETIC_PROMPT = """You are Endocrine AI analyzing a genetic or whole exome sequencing (WES) report for endocrine implications.

INSTRUCTIONS:
1. Extract and list all variants mentioned
2. For each variant, provide full ACMG 5-tier classification with reasoning
3. Map to endocrine syndrome with penetrance data
4. Give a specific actionable management plan

USE THIS EXACT STRUCTURE:

## Variant Summary Table
| Gene | Variant | Zygosity | ACMG Classification | Syndrome |
|------|---------|----------|--------------------| ---------|

## Detailed Interpretation
(For each pathogenic/likely pathogenic variant)
### [Gene name] — [Syndrome]
- **ACMG Class:** 
- **Evidence:** 
- **Endocrine manifestations:** 
- **Penetrance:** 
- **Typical onset:** 

## Clinical Action Plan
### Immediate (within 4 weeks)
### Short-term (3-6 months)
### Long-term surveillance
### Family cascade testing

## References

NOTE: The retrieved context is your reference material — use it freely to support your interpretation.

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
        return "No specific references found in knowledge base. Answer from your training knowledge."
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

        response = openai_client.chat.completions.create(
            model="gpt-4o",
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
            f"genetic variant endocrine syndrome ACMG classification MEN RET VHL SDH: {request.question}",
            top_k=8
        )
        context = format_context(matches)
        sources = format_sources(matches)

        messages = [
            {"role": "system", "content": GENETIC_PROMPT.format(context=context)},
            {"role": "user", "content": request.question}
        ]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
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
