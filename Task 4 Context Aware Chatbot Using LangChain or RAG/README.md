# 🤖 Task 4: Context-Aware Chatbot Using LangChain + RAG

A conversational chatbot that remembers chat history and retrieves answers from
a vectorized knowledge base (Retrieval-Augmented Generation), deployed with
Streamlit.

## How it works

```
Your documents (.txt / .md / .pdf)
        │  ingest.py
        ▼
 Split into chunks ──► Embed (sentence-transformers) ──► FAISS vector store
                                                                │
User question ──► retrieve top-k relevant chunks ◄─────────────┘
        │
        ▼
 LLM (local flan-t5 or OpenAI) + conversation history + retrieved context
        │
        ▼
    Answer + cited sources
```

## Project structure

```
Task-4-Context-Aware-Chatbot-Using-LangChain/
├── sample_data/            # Sample knowledge base to test the pipeline immediately
│   ├── company_overview.txt
│   ├── product_details.txt
│   └── support_faq.txt
├── ingest.py                # Loads docs, chunks, embeds, builds FAISS vector store
├── chatbot.py                # RAG chain + conversation memory (terminal chat loop)
├── app.py                    # Streamlit deployment
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Skills demonstrated

- Conversational AI development
- Document embedding and vector search
- Retrieval-Augmented Generation (RAG)
- LLM integration and deployment (local + API-based)

## 1. Setup

```bash
cd Task-4-Context-Aware-Chatbot-Using-LangChain
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

### (Optional) OpenAI backend
By default the chatbot runs a **local** model (`google/flan-t5-base`) — no API key
needed. If you'd rather use OpenAI for higher-quality answers:

```bash
copy .env.example .env      # Windows; `cp .env.example .env` on Mac/Linux
```
Then edit `.env` and add your key:
```
OPENAI_API_KEY=sk-...
```

## 2. Add your knowledge base

Use the included `sample_data/` folder to test immediately, or replace it with
your own `.txt`, `.md`, or `.pdf` files (e.g. Wikipedia exports, internal docs,
product manuals).

## 3. Build the vector store

```bash
python ingest.py --data_dir ./sample_data --vectorstore_dir ./vectorstore
```

This chunks your documents, embeds them, and saves a FAISS index to
`./vectorstore`. Re-run this any time your documents change.

## 4. Chat in the terminal (quick test)

```bash
python chatbot.py --vectorstore_dir ./vectorstore --llm_backend local
```

Try asking (using the included sample data):
```
You: What pricing tiers does CodePilot Cloud offer?
You: Which one includes SSO?
You: What was my first question?
```
That last question tests conversation memory — the bot should recall it without
you repeating context.

To use OpenAI instead:
```bash
python chatbot.py --vectorstore_dir ./vectorstore --llm_backend openai
```

## 5. Deploy with Streamlit

```bash
streamlit run app.py
```

Opens a chat UI in your browser with:
- Persistent conversation history for the session
- A sidebar to switch between local/OpenAI backends and tune how many chunks are retrieved
- An expandable "Sources" panel under each answer showing which documents it came from

## Notes

- `google/flan-t5-base` (~250M params) downloads once (~1GB) and then runs fully offline. For faster answers on modest hardware, swap in `google/flan-t5-small` in `chatbot.py`/`app.py`.
- The embedding model `all-MiniLM-L6-v2` is small (~80MB) and runs on CPU comfortably.
- `ConversationBufferMemory` keeps the full chat history in memory for the session — for very long conversations, consider `ConversationSummaryMemory` instead to keep the prompt size manageable.
- The vector store isn't committed to git (see `.gitignore`) — anyone cloning the repo runs `ingest.py` themselves to rebuild it.

## License

MIT
