"""
ingest.py
---------
Loads documents from a folder (.txt, .md, .pdf), splits them into chunks,
embeds them, and builds a local FAISS vector store for retrieval.

Usage:
    python ingest.py --data_dir ./sample_data --vectorstore_dir ./vectorstore
"""

import argparse
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./sample_data")
    parser.add_argument("--vectorstore_dir", type=str, default="./vectorstore")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Any sentence-transformers model name.",
    )
    return parser.parse_args()


def load_documents(data_dir):
    documents = []

    # Plain text / markdown
    txt_loader = DirectoryLoader(
        data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    documents.extend(md_loader.load())

    # PDFs (loaded one by one so a single bad PDF doesn't kill the whole run)
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                path = os.path.join(root, fname)
                try:
                    documents.extend(PyPDFLoader(path).load())
                except Exception as e:
                    print(f"Skipping {path}: {e}")

    return documents


def main():
    args = parse_args()

    print(f"Loading documents from {args.data_dir} ...")
    documents = load_documents(args.data_dir)
    print(f"Loaded {len(documents)} document(s).")

    if not documents:
        print(
            "No documents found. Add .txt, .md, or .pdf files to "
            f"'{args.data_dir}' and re-run, or point --data_dir at ./sample_data "
            "to test with the included sample corpus."
        )
        return

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model: {args.embedding_model} ...")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(args.vectorstore_dir, exist_ok=True)
    vectorstore.save_local(args.vectorstore_dir)
    print(f"Vector store saved to {args.vectorstore_dir}")


if __name__ == "__main__":
    main()
