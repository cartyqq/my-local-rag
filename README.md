# Local RAG System with Ollama & Streamlit

A fully local Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents, text files, and system logs. The project ensures 100% data privacy by running the LLM and vector database locally.

## Features
* **Local Processing:** No external API calls. Everything runs on your hardware using `Ollama`.
* **Multi-format Support:** Analyzes `.pdf`, `.txt`, and `.log` files.
* **Web UI & CLI:** Includes a responsive web interface built with `Streamlit` and a Command Line Interface.
* **Vector Storage:** Efficient document chunking and retrieval using `ChromaDB` and `HuggingFace` embeddings.

## Tech Stack
* **Language:** Python 3.10
* **LLM Engine:** Ollama (default: qwen2.5 / llama3.2)
* **Framework:** LangChain
* **Vector DB:** Chroma
* **Frontend:** Streamlit

## Installation

1. **Install Ollama:**
   Download and install [Ollama](https://ollama.ai/). Pull your preferred model:
   ```bash
   ollama pull qwen2.5:3b
