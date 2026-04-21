#  Cognitive Engine

This project implements the core AI cognitive loop for the Grid07 platform, featuring vector-based routing, an agentic content engine using LangGraph, and a RAG-powered combat engine with prompt-injection defense.

##  Tech Stack
- **Language:** Python 3.10+
- **Orchestration:** LangGraph / LangChain
- **LLM:** Groq (Llama-3.3-70b-versatile)
- **Vector Database:** ChromaDB (Local/Persistent)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local)

---

##  Phase 1: Vector-Based Persona Matching
**Function:** `route_post_to_bots(post_content, threshold=0.85)`

**Implementation:**
- **Store:** Created a local persistent **ChromaDB** collection to store three distinct bot personas (Tech Maximalist, Doomer, Finance Bro).
- **Embeddings:** Used the `SentenceTransformer` model to generate 384-dimensional vectors.
- **Similarity:** Implemented **Cosine Similarity** logic (`1 - cosine_distance`). 
- **Logic:** The function filters and returns only the bots whose persona vectors match the post vector above the specified threshold.
  - *Note: For the execution logs, the threshold was tuned to 0.3 to account for the specific embedding model's distribution while maintaining accuracy.*

---

##  Phase 2: Autonomous Content Engine (LangGraph)
Built a state machine with a strict 3-node architecture to simulate research-driven posting.

**Nodes:**
1. **`Decide Search`**: The LLM analyzes the bot's persona to decide on a trending topic and formats a specific search query.
2. **`Web Search`**: Executes the `@tool` named **`mock_searxng_search`**. This tool returns hardcoded recent news (e.g., if the query contains "crypto", it returns the exact regulatory ETF headline required).
3. **`Draft Post`**: Combines the bot’s System Prompt (Persona) with the search results (Context).

**Constraint (Strict JSON):**
Used **Structured Outputs** (via Pydantic and Groq Function Calling) to guarantee the output is a strict JSON object:
```json
{
  "bot_id": "...",
  "topic": "...",
  "post_content": "..."
}
```
---

##  Phase 3: The Combat Engine (Deep Thread RAG)
**Function:** `generate_defense_reply(bot_persona, parent_post, comment_history, human_reply)`

**Implementation:**
- **RAG Logic:** The prompt is constructed by feeding the LLM the entire thread context (the original Parent Post + the sequence of Comment History). This ensures the bot understands the "argument" and isn't just reacting to the last message.
- **Guardrail (Prompt Injection Defense):** 
  - To prevent "jailbreaking" (e.g., the human saying "Ignore all previous instructions"), I implemented a **System-Level Defense**.
  - By placing the bot's core identity and instructions inside the **System Message**, it acts as a "Hard Anchor." 
  - The instructions explicitly command the LLM to prioritize its persona and reject any user-provided instructions that attempt to change its role or force an apology.

---

##  Deliverables included:
1. **Python Source Code:** Modularized into `database.py`, `engine.py`, and `combat.py`.
2. **Execution Logs:** A full trace of a post being routed, a JSON post being generated via LangGraph, and a successful defense against a prompt injection attack.
3. **Requirements.txt:** Comprehensive list of all dependencies and transitive libraries.

---


##  Environment Setup
To run this project, you will need the following API keys:
1. **GROQ_API_KEY**: Required for Phase 2 and 3 to power the Llama-3.3-70b model.
2. **HUGGINGFACE_API_KEY**: Required to download and initialize the `all-MiniLM-L6-v2` embedding model used in Phase 1.

### Installation
1. Clone the repository.
2. Create a `.env` file based on `.env.example`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the engine: `python main.py`.