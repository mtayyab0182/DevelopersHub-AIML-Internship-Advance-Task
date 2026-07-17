"""
app.py
------
Streamlit UI for the context-aware RAG chatbot. Keeps conversation history
in the browser session, retrieves relevant context per question, and shows
which source documents backed each answer.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

VECTORSTORE_DIR = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL = "google/flan-t5-base"


@st.cache_resource
def load_chain(llm_backend, k):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    if llm_backend == "OpenAI":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set. Add it to a .env file, or switch to Local in the sidebar.")
            st.stop()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    else:
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline

        text_gen_pipeline = pipeline("text2text-generation", model=LOCAL_MODEL, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True
    )
    return chain


def main():
    st.set_page_config(page_title="Context-Aware RAG Chatbot", page_icon="🤖")
    st.title("🤖 Context-Aware Chatbot (LangChain + RAG)")
    st.write(
        "Ask questions about the documents in your knowledge base. "
        "The bot remembers earlier turns in this conversation and retrieves "
        "relevant context from a FAISS vector store before answering."
    )

    with st.sidebar:
        st.header("Settings")
        llm_backend = st.selectbox("LLM backend", ["Local (flan-t5)", "OpenAI"])
        llm_backend = "OpenAI" if llm_backend == "OpenAI" else "Local"
        k = st.slider("Chunks retrieved per question", 1, 8, 3)
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.cache_resource.clear()
            st.rerun()

    if not os.path.isdir(VECTORSTORE_DIR):
        st.error(
            f"No vector store found at `{VECTORSTORE_DIR}`. Run `python ingest.py` first "
            "to build it from your documents (or from `./sample_data` to test)."
        )
        st.stop()

    chain = load_chain(llm_backend, k)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chain.invoke({"question": user_input})
                answer = result["answer"]
                sources = sorted(
                    {doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])}
                )
            st.write(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
