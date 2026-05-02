# RAG — Retrieval Augmented Generation

> **RAG solves the two biggest problems with LLMs: knowledge cutoffs and private data.** Instead of baking knowledge into weights, retrieve it at inference time. No retraining, always up-to-date, grounded in your sources.

---

## Table of Contents

1. [The Problem RAG Solves](#1-the-problem-rag-solves)
2. [What is RAG?](#2-what-is-rag)
3. [The Three Steps](#3-the-three-steps)
4. [Embeddings and Semantic Search](#4-embeddings-and-semantic-search)
5. [Vector Stores](#5-vector-stores)
6. [RAG vs Fine-tuning](#6-rag-vs-fine-tuning)
7. [Results from Demo](#7-results-from-demo)
8. [Common Issues](#8-common-issues)
9. [Production RAG Stack](#9-production-rag-stack)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. The Problem RAG Solves

### LLMs Have Two Big Limitations

**Problem 1: Knowledge cutoff**

```
GPT-4 trained on data up to April 2023
Ask about something newer → hallucination

"What happened in the news today?"
→ LLM makes something up (it doesn't know)
```

**Problem 2: Private knowledge**

```
LLM doesn't know:
  - Your company's internal docs
  - Your codebase
  - Your customer support tickets
  - Your research notes

Fine-tuning helps but:
  - Expensive (hours/days of training)
  - Static (needs retraining when docs change)
  - Can still hallucinate
```

### RAG Solves Both

```
Knowledge cutoff:
  Add new documents to vector store (no retraining)
  LLM now "knows" about recent events

Private knowledge:
  Index your private docs in vector store
  LLM retrieves and uses them at inference time
  No training, no data leakage to external APIs
```

---

## 2. What is RAG?

**RAG** = **R**etrieval **A**ugmented **G**eneration

The core idea: **give the LLM relevant context at inference time instead of baking it into weights.**

```
Without RAG:
  User: "What's our refund policy?"
  LLM:  [guesses, hallucinates, gets it wrong]

With RAG:
  User: "What's our refund policy?"
    ↓
  Search vector store for "refund policy"
    ↓
  Found: "Customers can return items within 30 days..."
    ↓
  Prompt: "Context: [doc]. Question: What's our refund policy?"
    ↓
  LLM: "According to our policy, you can return items within 30 days"
       ← Grounded in actual document, not a guess
```

### The Key Insight

```
LLM weights = long-term memory (static, baked in during training)
RAG context  = working memory  (dynamic, retrieved per query)

Humans work the same way:
  Long-term memory: things you've learned over years
  Working memory:   the notes/docs you look up before answering
```

---

## 3. The Three Steps

### Step 1: Indexing (One-time Setup)

```
Your documents
  ↓
Chunk into smaller pieces
  (Paragraph-level, ~200-500 tokens per chunk)
  ↓
Embed each chunk → dense vector
  (e.g., 384 or 1536 dimensions)
  ↓
Store in vector database
  (FAISS, ChromaDB, Pinecone, etc.)

Cost: One-time, done once
When to redo: When documents change
```

### Step 2: Retrieval (Every Query)

```
User query
  ↓
Embed query → dense vector
  (Same embedding model as indexing)
  ↓
Similarity search in vector store
  Find top-k most similar document vectors
  (Cosine similarity or dot product)
  ↓
Return top-k document chunks

Cost: Milliseconds per query
Quality: Depends on embedding model + chunk size
```

### Step 3: Generation (Every Query)

```
User query + retrieved chunks
  ↓
Build augmented prompt:
  "Answer the question based on this context:
   [retrieved doc 1]
   [retrieved doc 2]
   Question: [user query]
   Answer:"
  ↓
Feed to LLM
  ↓
LLM generates grounded answer

Cost: LLM inference time
Quality: Depends on LLM size and retrieved context quality
```

---

## 4. Embeddings and Semantic Search

### Why Embeddings Enable Semantic Search

Traditional keyword search:

```
Query: "Can I return this?"
Search: finds documents containing "return"
Misses: documents about "refund", "exchange", "send back"
```

Embedding-based search:

```
Query: "Can I return this?"
Embed → vector [0.2, -0.5, 0.8, ...]

"Refund policy: items can be sent back within 30 days"
Embed → vector [0.21, -0.48, 0.79, ...]
                ← Very similar! High cosine similarity

Search finds it even though "return" ≠ "refund"
Semantics match even when keywords don't
```

### How Cosine Similarity Works

```
similarity = cos(θ) = (A · B) / (|A| × |B|)

Range: -1 to 1
  1.0  = identical meaning
  0.0  = unrelated
  -1.0 = opposite meaning

In practice:
  > 0.8: very similar (near duplicate)
  0.5-0.8: related topics
  0.2-0.5: somewhat related
  < 0.2: unrelated
```

### Embedding Models

```
all-MiniLM-L6-v2 (used in demo):
  384 dimensions
  Fast, small, good quality
  Best for: local, quick prototypes

all-mpnet-base-v2:
  768 dimensions
  Slower, better quality
  Best for: higher accuracy needed

OpenAI text-embedding-ada-002:
  1536 dimensions
  Excellent quality, cloud API
  Best for: production systems

OpenAI text-embedding-3-large:
  3072 dimensions
  Best quality available
  Best for: maximum accuracy
```

---

## 5. Vector Stores

A database optimised for **similarity search** over dense vectors.

### FAISS (Used in Demo)

```
Facebook AI Similarity Search
  - Runs locally (no server)
  - Extremely fast
  - No persistence by default (in-memory)
  - Great for: prototypes, research, local apps

# Build index
index = faiss.IndexFlatIP(embedding_dim)  # Inner product
index.add(embeddings.astype('float32'))

# Search
similarities, indices = index.search(query_embedding, k=3)
```

### ChromaDB

```
Easy to use, local or cloud
  - Persistent storage (saves to disk)
  - Simple Python API
  - Metadata filtering
  - Great for: development, small production

import chromadb
client = chromadb.Client()
collection = client.create_collection("my_docs")
collection.add(documents=docs, ids=ids)
results = collection.query(query_texts=["my query"], n_results=3)
```

### Pinecone

```
Managed cloud vector database
  - Horizontal scaling
  - REST API
  - Real-time updates
  - Great for: production at scale

import pinecone
pinecone.init(api_key="...")
index = pinecone.Index("my-index")
index.upsert(vectors=[(id, embedding, metadata)])
results = index.query(vector=query_embedding, top_k=3)
```

### Comparison

| Store | Setup | Speed | Scale | Cost | Best for |
|-------|-------|-------|-------|------|----------|
| FAISS | Instant | Fastest | Medium | Free | Prototypes |
| ChromaDB | Easy | Fast | Medium | Free | Development |
| Pinecone | Medium | Fast | Huge | Paid | Production |
| Weaviate | Complex | Fast | Huge | Free/Paid | Enterprise |

---

## 6. RAG vs Fine-tuning

### When to Use RAG

```
✅ Knowledge changes frequently (news, product updates)
✅ Large private knowledge base (thousands of docs)
✅ Need to cite sources (grounded answers)
✅ Can't afford fine-tuning compute
✅ Want to reduce hallucination
✅ Multiple knowledge bases (swap vector store)
✅ Need to update knowledge without retraining
```

### When to Use Fine-tuning

```
✅ Style/tone adaptation (write like our brand)
✅ Domain-specific reasoning patterns
✅ Latency-sensitive (no retrieval step)
✅ Small, stable knowledge base
✅ Task-specific format (code, JSON, etc.)
✅ Maximum performance on specific task
```

### Use Both Together (Best Practice)

```
Fine-tune for:
  - Writing style
  - Domain vocabulary
  - Reasoning patterns

RAG for:
  - Factual knowledge
  - Up-to-date information
  - Private documents

Example: Customer support bot
  Fine-tune on support conversations (learn tone/style)
  RAG on product docs (learn current facts)
  → Best of both worlds
```

### Side-by-side Comparison

| Aspect | Fine-tuning | RAG |
|--------|-------------|-----|
| **Knowledge update** | Retrain (expensive) | Update vector store (cheap) |
| **Latency** | Fast (no retrieval) | Slightly slower |
| **Hallucination** | Can still hallucinate | Less (grounded in docs) |
| **Cost** | High (training) | Low (inference only) |
| **Transparency** | Black box | Can cite sources |
| **Data privacy** | Data in training | Data in vector store |

---

## 7. Results from Demo

### Similarity Search Results

Query: "How does attention mechanism work in Transformers?"

```
Rank 1 (sim=0.76): "Transformers are neural networks that use attention
                    mechanisms..." ← Directly relevant
Rank 2 (sim=0.53): "Attention mechanisms compute relevance scores between
                    all pairs of tokens..." ← Also directly relevant
Rank 3 (sim=0.28): "Fine-tuning adapts a pre-trained model..." ← Less relevant

Key observation:
  Semantic search found the right documents
  Even though query words ≠ document words exactly
  "how does it work" matched "compute relevance scores"
```

### RAG vs No RAG Comparison

Query: "What is LoRA and how does it reduce memory usage?"

```
Without RAG (pure GPT-2):
  "LoRA is a method of computing the number of bytes in memory..."
  ❌ Completely wrong — hallucinated a computing definition
  GPT-2 doesn't know ML's LoRA

With RAG (GPT-2 + retrieved context):
  Retrieved: "LoRA stands for Low-Rank Adaptation. It freezes the
             original model weights and adds small trainable matrices..."
  Generated: Still went off-track (GPT-2 too small to follow instructions)
  ⚠️ Retrieval worked, generation didn't

Key learning:
  RAG retrieval worked perfectly (found the right document)
  GPT-2 (124M) too small to use context effectively
  Production RAG uses GPT-4 / Claude / LLaMA 7B+
  Bigger LLM = much better at following "answer using this context"
```

### Why Retrieval Worked but Generation Struggled

```
Embedding model: sentence-transformers (trained for semantic similarity)
  → Excellent at finding relevant documents
  → Task: find similar vectors (well-defined, well-trained)

GPT-2: small decoder-only model (124M params)
  → Poor at instruction following
  → Task: "answer this question using this context" requires larger model
  → GPT-2 ignores context and generates from its weights

Fix: Use LLaMA 7B, GPT-4, or Claude as the generation LLM
```

---

## 8. Common Issues

### Issue 1: Poor Retrieval Quality

```
Symptom: Retrieved documents are not relevant to query

Causes:
  - Embedding model too weak
  - Chunks too large (too much irrelevant text per chunk)
  - Chunks too small (not enough context per chunk)
  - Query and documents in different styles

Fixes:
  - Use better embedding model (mpnet, ada-002)
  - Tune chunk size (try 200-500 tokens)
  - Add chunk overlap (50-100 tokens)
  - Use hybrid search (keywords + semantic)
```

### Issue 2: LLM Ignores Retrieved Context

```
Symptom: LLM generates answer from its weights, ignores context

Causes:
  - LLM too small (GPT-2, small models)
  - Prompt not instructing model clearly enough
  - Context too long (LLM gets confused)

Fixes:
  - Use larger LLM (7B+ parameters)
  - Make prompt more explicit:
    "You MUST answer only using the provided context.
     Do not use any external knowledge."
  - Reduce context length (fewer retrieved docs)
```

### Issue 3: Hallucination Despite RAG

```
Symptom: LLM still makes things up even with retrieved context

Causes:
  - Retrieved document doesn't contain the answer
  - LLM mixes retrieved facts with its weights
  - Query too complex for retrieved context

Fixes:
  - Add "If the context doesn't contain the answer, say 'I don't know'"
  - Increase top-k retrieval (retrieve more docs)
  - Use re-ranking (rank retrieved docs by relevance)
  - Use smaller, more focused knowledge base
```

### Issue 4: Slow Retrieval

```
Symptom: Retrieval takes too long at inference time

Causes:
  - Large vector store (millions of vectors)
  - Exact search (IndexFlatIP) doesn't scale

Fixes:
  - Use approximate search (FAISS IVF, HNSW)
  - Cache frequent queries
  - Use managed vector store (Pinecone, Weaviate)
  - Reduce embedding dimensions
```

---

## 9. Production RAG Stack

### Minimal Stack (Good for Starting)

```python
# Embeddings
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-mpnet-base-v2')

# Vector store
import chromadb
db = chromadb.Client()
collection = db.create_collection("knowledge_base")

# LLM
from transformers import pipeline
llm = pipeline('text-generation', model='meta-llama/Llama-3.2-1B')

# RAG pipeline
def rag(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    context = "\n".join(results['documents'][0])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return llm(prompt, max_new_tokens=200)[0]['generated_text']
```

### Full Production Stack

```
Embeddings:    OpenAI text-embedding-3-large
               or sentence-transformers (self-hosted)

Vector store:  Pinecone (managed, scalable)
               or Weaviate (self-hosted, enterprise)

LLM:           GPT-4 / Claude (API)
               or LLaMA 7B+ (self-hosted)

Framework:     LangChain or LlamaIndex
               (handles chunking, retrieval, prompt building)

Monitoring:    LangSmith or Weights & Biases
               (track retrieval quality, latency, hallucination)
```

### LangChain RAG (Industry Standard)

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Build vector store
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Query
result = qa_chain("What is our refund policy?")
print(result['result'])
print(result['source_documents'])
```

---

## 10. How This Connects to LLMs

### Claude, ChatGPT, Gemini Use RAG

```
Base model: Pre-trained + fine-tuned LLM (weights)
  → General language understanding
  → Reasoning ability

RAG layer (optional, user-triggered):
  Web search → retrieve web pages → augment prompt
  File upload → retrieve from your docs → augment prompt
  Google Drive → retrieve from your files → augment prompt

When you search the web with Claude:
  That's RAG — retrieving context, augmenting the prompt
  Claude's weights didn't change
  The context window got richer
```

### The Bigger Picture

```
Phase 1: Classical NLP (BoW, TF-IDF)
  → Keyword search (exact match)

Phase 3: Transformers (BERT embeddings)
  → Dense representations, semantic similarity

Phase 4: RAG
  → Semantic search at scale + LLM generation
  → Combines everything you've learned

Phase 5: Mini-GPT
  → Build the LLM that powers RAG generation
```

RAG is where everything comes together:
- **Embeddings** (Phase 1 & 3)
- **Transformers** (Phase 3)
- **LLM generation** (Phase 4)
- **Efficient retrieval** (new in this phase)

---

## Summary

**RAG in three steps:**

```
1. INDEX:    Documents → embeddings → vector store (one-time)
2. RETRIEVE: Query → embedding → find similar docs → top-k
3. GENERATE: Query + retrieved docs → prompt → LLM → answer
```

**Why it works:**
- Embeddings capture semantic meaning (not just keywords)
- Cosine similarity finds relevant documents efficiently
- LLM uses retrieved context to generate grounded answers

**Key lesson from demo:**
- Retrieval works great even with small models
- Generation quality depends heavily on LLM size
- Production RAG = great retrieval + large LLM

**RAG is one of the most employable skills in AI engineering** — almost every AI product uses it to ground LLMs in private or up-to-date knowledge.

---

*Phase 4 — concept 04 → you are here*  
*Previous concept → [03 — Fine-tuning GPT-2 with LoRA](./03_finetune_gpt2_lora.md)*  
*Next → Phase 5: Mini-GPT Capstone*
