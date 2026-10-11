"""
Streamlit interface for a local RAG system.

Stack: PyPDFLoader / RecursiveCharacterTextSplitter -> all-MiniLM-L6-v2 ->
ChromaDB (local) -> Ollama (http://localhost:11434/api/generate).

Installation (Miniconda):
    conda create -n rag python=3.11 -y
    conda activate rag
    pip install streamlit requests pypdf chromadb sentence-transformers \
                langchain-community langchain-text-splitters \
                langchain-huggingface langchain-chroma

Run:
    ollama serve            # if the server is not running yet
    ollama pull llama3.2
    streamlit run app.py
"""

import json
import os
import tempfile

import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------- Settings -----------------------------------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
DEFAULT_MODEL = "llama3.2"

HISTORY_TURNS = 4  # how many recent messages to include in the prompt

PROMPT_TEMPLATE = """Ты — помощник, отвечающий на вопросы по загруженным документам и логам.
Используй ТОЛЬКО приведённый ниже контекст. Если в контексте нет ответа,
честно скажи, что в документах этой информации нет. Отвечай на языке вопроса.

{history}Контекст:
{context}

Вопрос: {question}

Ответ:"""


# ------------------------- Resources (cached) -------------------------------
@st.cache_resource(show_spinner="Загрузка модели эмбеддингов…")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def db_count() -> int:
    """Number of chunks in the database."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


# ------------------------------- Ollama -------------------------------------
def list_ollama_models() -> list[str] | None:
    """List of Ollama models, or None if the server is unavailable."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.RequestException:
        return None


def stream_ollama(prompt: str, model: str):
    """Generator of response tokens from Ollama (POST with stream=True)."""
    with requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=(5, 300),
    ) as resp:
        if resp.status_code == 404:
            raise RuntimeError(
                f"Модель «{model}» не найдена. Выполните: `ollama pull {model}`"
            )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(data["error"])
            yield data.get("response", "")
            if data.get("done"):
                break


# ------------------------- Loading and indexing -----------------------------
def load_documents(uploaded_file) -> list[Document]:
    """Load a PDF (PyPDFLoader) or a plain-text/log file."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    raw = uploaded_file.getvalue()

    if ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            docs = PyPDFLoader(tmp_path).load()
        finally:
            os.remove(tmp_path)
    else:
        text = raw.decode("utf-8", errors="replace")
        docs = [Document(page_content=text, metadata={})]

    for d in docs:
        d.metadata["source"] = name
    return docs


def index_file(uploaded_file) -> int:
    """Split the file into chunks, embed them and save to ChromaDB."""
    docs = load_documents(uploaded_file)
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError(
            "Не удалось извлечь текст из файла (возможно, это скан без текстового слоя)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    db = get_vectorstore()
    # Remove the old version of this file to avoid duplicates
    db._collection.delete(where={"source": uploaded_file.name})
    db.add_documents(chunks)
    return len(chunks)


def clear_database() -> None:
    db = get_vectorstore()
    ids = db._collection.get()["ids"]
    if ids:
        db._collection.delete(ids=ids)


# --------------------------------- RAG --------------------------------------
def build_prompt(question: str, docs: list[Document], history: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')}"
        + (f", стр. {d.metadata['page'] + 1}" if "page" in d.metadata else "")
        + f"]\n{d.page_content}"
        for d in docs
    )

    hist_text = ""
    if history:
        lines = [
            f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
            for m in history[-HISTORY_TURNS:]
        ]
        hist_text = "История диалога:\n" + "\n".join(lines) + "\n\n"

    return PROMPT_TEMPLATE.format(history=hist_text, context=context, question=question)


# ---------------------------------- UI --------------------------------------
st.set_page_config(page_title="Локальный RAG-чат", page_icon="💬", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ---- Sidebar ----
with st.sidebar:
    st.header("📄 Документы")

    uploaded = st.file_uploader(
        "Загрузите файл",
        type=["pdf", "log", "txt"],
        help="Поддерживаются PDF, .log и .txt",
    )

    if st.button("Проиндексировать документ", type="primary", use_container_width=True):
        if uploaded is None:
            st.warning("Сначала выберите файл для загрузки.")
        else:
            try:
                with st.spinner("Нарезка и векторизация…"):
                    n_chunks = index_file(uploaded)
                if uploaded.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(uploaded.name)
                st.success(f"✅ «{uploaded.name}» сохранён в базу ({n_chunks} кусков).")
            except Exception as e:
                st.error(f"Ошибка индексации: {e}")

    st.divider()
    total = db_count()
    if total:
        st.info(f"🗄️ В базе кусков: **{total}**")
    else:
        st.warning("База пуста. Загрузите и проиндексируйте документ.")

    if st.session_state.indexed_files:
        st.caption("Проиндексировано в этой сессии:")
        for f in st.session_state.indexed_files:
            st.caption(f"• {f}")

    st.divider()
    st.subheader("⚙️ Настройки")

    models = list_ollama_models()
    if models is None:
        st.error("Ollama недоступна. Запустите `ollama serve`.")
        model = st.text_input("Модель Ollama", value=DEFAULT_MODEL)
    elif not models:
        st.warning("В Ollama нет моделей. Выполните `ollama pull llama3.2`.")
        model = st.text_input("Модель Ollama", value=DEFAULT_MODEL)
    else:
        idx = next((i for i, m in enumerate(models) if m.startswith(DEFAULT_MODEL)), 0)
        model = st.selectbox("Модель Ollama", models, index=idx)

    top_k = st.slider("Кусков в контексте (top-k)", 1, 10, 4)

    c1, c2 = st.columns(2)
    if c1.button("🧹 Очистить чат", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if c2.button("🗑️ Очистить базу", use_container_width=True):
        clear_database()
        st.session_state.indexed_files = []
        st.rerun()

# ---- Main area ----
st.title("💬 Локальный RAG-чат")
st.caption("Вопросы по вашим PDF и лог-файлам · ChromaDB + Ollama · всё работает локально")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Использованные фрагменты"):
                for s in msg["sources"]:
                    st.caption(s["label"])
                    st.text(s["text"])

question = st.chat_input("Задайте вопрос по документам…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if db_count() == 0:
            st.warning("База пуста — сначала загрузите и проиндексируйте документ в сайдбаре.")
        elif models is None:
            st.error(
                "Сервер Ollama недоступен по адресу "
                f"`{OLLAMA_BASE_URL}`. Запустите его командой `ollama serve`."
            )
        else:
            try:
                with st.spinner("Поиск релевантных фрагментов…"):
                    docs = get_vectorstore().similarity_search(question, k=top_k)

                prompt = build_prompt(question, docs, st.session_state.messages[:-1])
                answer = st.write_stream(stream_ollama(prompt, model))

                sources = [
                    {
                        "label": f"{d.metadata.get('source', '?')}"
                        + (f" · стр. {d.metadata['page'] + 1}" if "page" in d.metadata else ""),
                        "text": d.page_content,
                    }
                    for d in docs
                ]
                with st.expander("Использованные фрагменты"):
                    for s in sources:
                        st.caption(s["label"])
                        st.text(s["text"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except requests.exceptions.ConnectionError:
                st.error("Потеряно соединение с Ollama. Проверьте, что `ollama serve` запущен.")
            except requests.exceptions.Timeout:
                st.error("Ollama не ответила вовремя. Попробуйте модель поменьше или повторите запрос.")
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Непредвиденная ошибка: {e}")
