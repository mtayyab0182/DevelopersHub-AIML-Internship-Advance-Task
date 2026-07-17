"""
chatbot.py
----------
Builds a context-aware RAG chatbot: retrieves relevant chunks from the FAISS
vector store and answers using conversation history + retrieved context.

Supports two LLM backends:
    --llm_backend local   -> runs google/flan-t5-base locally, no API key needed
    --llm_backend openai  -> uses OpenAI's API, requires OPENAI_API_KEY

Usage (terminal chat loop):
    python chatbot.py --vectorstore_dir ./vectorstore --llm_backend local
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorstore_dir", type=str, default="./vectorstore")
    parser.add_argument(
        "--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--llm_backend", type=str, choices=["local", "openai"], default="local")
    parser.add_argument("--local_model", type=str, default="google/flan-t5-base")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve per query")
    return parser.parse_args()


def build_llm(llm_backend, local_model, openai_model):
    if llm_backend == "openai":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to a .env file or your environment, "
                "or use --llm_backend local instead."
            )
        return ChatOpenAI(model=openai_model, temperature=0.2)

    # Local backend: no API key required
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline

    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=local_model,
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def build_chain(args):
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    vectorstore = FAISS.load_local(
        args.vectorstore_dir, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    llm = build_llm(args.llm_backend, args.local_model, args.openai_model)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain, memory


def main():
    args = parse_args()

    if not os.path.isdir(args.vectorstore_dir):
        print(
            f"No vector store found at '{args.vectorstore_dir}'. "
            "Run ingest.py first, e.g.:\n"
            "    python ingest.py --data_dir ./sample_data --vectorstore_dir ./vectorstore"
        )
        return

    print("Loading chatbot (this can take a moment on first run)...")
    chain, _ = build_chain(args)
    print("Ready. Type your question, or 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = chain.invoke({"question": query})
        print(f"\nBot: {result['answer']}\n")

        sources = result.get("source_documents", [])
        if sources:
            print("Sources:")
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                print(f"  - {src}")
            print()


if __name__ == "__main__":
    main()
