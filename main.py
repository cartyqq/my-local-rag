#!/usr/bin/env python3
"""
Локальный RAG: PDF -> чанки -> эмбеддинги (all-MiniLM-L6-v2) -> ChromaDB -> Ollama.

Установка зависимостей:
    pip install langchain-community langchain-text-splitters langchain-huggingface \
                langchain-chroma chromadb sentence-transformers pypdf requests

Перед запуском должен работать Ollama и быть скачана модель:
    ollama pull llama3.2

Примеры:
    # индексация PDF
    python main.py index document.pdf

    # один вопрос
    python main.py ask "О чём этот документ?"

    # индексация + интерактивный чат
    python main.py chat --pdf document.pdf
"""

import argparse
import sys

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- Настройки ------------------------------------
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


# ------------------------------ Эмбеддинги ----------------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


# ------------------------- 1. Загрузка и нарезка ----------------------------
def load_and_split(log_path: str):
    # Используем TextLoader с указанием кодировки (логи часто пишутся в utf-8)
    loader = TextLoader(log_path, encoding="utf-8")
    pages = loader.load()

    # Увеличиваем размер куска для захвата длинных Traceback и логов Docker
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " "] 
    )
    chunks = splitter.split_documents(pages)
    print(f"Лог загружен, разбито на блоков: {len(chunks)}")
    return chunks


# ------------------- 2. Векторизация и сохранение в Chroma ------------------
def index_pdf(pdf_path: str) -> None:
    chunks = load_and_split(pdf_path)
    if not chunks:
        sys.exit("Не удалось извлечь текст из PDF (возможно, это скан).")

    db = get_vectorstore()
    db.add_documents(chunks)
    print(f"Проиндексировано и сохранено в '{CHROMA_DIR}'.")


# ------------------------- 3. Поиск и вопрос к Ollama -----------------------
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


# --------------------------------- CLI --------------------------------------
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
