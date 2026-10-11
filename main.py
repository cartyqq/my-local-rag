#!/usr/bin/env python3
"""
Local RAG: PDF -> chunks -> embeddings (all-MiniLM-L6-v2) -> ChromaDB -> Ollama.

Install dependencies:
    pip install langchain-community langchain-text-splitters langchain-huggingface \
                langchain-chroma chromadb sentence-transformers pypdf requests

Ollama must be running and the model must be pulled before launching:
    ollama pull llama3.2

Examples:
    # index a PDF
    python main.py index document.pdf

    # ask a single question
    python main.py ask "What is this document about?"

    # index + interactive chat
    python main.py chat --pdf document.pdf
"""

import argparse
import sys

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------- Settings -----------------------------------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

PROMPT_TEMPLATE = """Ты — Senior DevOps инженер. Твоя задача — проанализировать технические логи, найти причину падения (ошибку) и предложить конкретные шаги для её исправления.
Используй ТОЛЬКО фрагменты логов, приведенные ниже. Отвечай на русском языке.

Фрагменты логов:
{context}

Запрос пользователя: {question}

Отчет об ошибке и решение:"""


# ------------------------------- Embeddings ---------------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


# ---------------------- 1. Loading and splitting ----------------------------
def load_and_split(log_path: str):
    # Use TextLoader with an explicit encoding (logs are usually written in utf-8)
    loader = TextLoader(log_path, encoding="utf-8")
    pages = loader.load()

    # Larger chunk size to capture long tracebacks and Docker logs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " "] 
    )
    chunks = splitter.split_documents(pages)
    print(f"Лог загружен, разбито на блоков: {len(chunks)}")
    return chunks


# ------------------ 2. Vectorization and saving to Chroma -------------------
def index_pdf(pdf_path: str) -> None:
    chunks = load_and_split(pdf_path)
    if not chunks:
        sys.exit("Не удалось извлечь текст из PDF (возможно, это скан).")

    db = get_vectorstore()
    db.add_documents(chunks)
    print(f"Проиндексировано и сохранено в '{CHROMA_DIR}'.")


# ------------------------ 3. Search and querying Ollama ---------------------
def ask_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL, 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "repeat_penalty": 1.2
                }
            },
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit("Не удалось подключиться к Ollama. Запущен ли `ollama serve`?")
    except requests.exceptions.HTTPError as e:
        sys.exit(f"Ошибка Ollama: {e}\n{resp.text}")
    return resp.json()["response"].strip()


def answer_question(question: str) -> str:
    db = get_vectorstore()
    docs = db.similarity_search(question, k=TOP_K)
    if not docs:
        return "База пуста. Сначала проиндексируйте PDF командой `index`."

    context = "\n\n---\n\n".join(
        f"[стр. {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return ask_ollama(prompt)


# ---------------------------------- CLI -------------------------------------
def chat_loop() -> None:
    print("Интерактивный режим. Пустая строка или 'exit' — выход.")
    while True:
        try:
            q = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("\nОтвет:", answer_question(q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный RAG по PDF")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="проиндексировать PDF")
    p_index.add_argument("pdf", help="путь к PDF-файлу")

    p_ask = sub.add_parser("ask", help="задать один вопрос")
    p_ask.add_argument("question", help="текст вопроса")

    p_chat = sub.add_parser("chat", help="интерактивный режим")
    p_chat.add_argument("--pdf", help="PDF для предварительной индексации")

    args = parser.parse_args()

    if args.cmd == "index":
        index_pdf(args.pdf)
    elif args.cmd == "ask":
        print(answer_question(args.question))
    elif args.cmd == "chat":
        if args.pdf:
            index_pdf(args.pdf)
        chat_loop()


if __name__ == "__main__":
    main()
