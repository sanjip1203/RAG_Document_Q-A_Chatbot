
# Research Paper Q&A (RAG) — Streamlit + LangChain + Groq + FAISS

A simple **Retrieval-Augmented Generation (RAG)** app built with **Streamlit** that lets you ask questions from a folder of PDF research papers.  
It loads PDFs, chunks the text, creates embeddings using **Ollama**, stores them in **FAISS**, and answers queries using **Groq LLM**.

---

## Features

- 📄 Load all PDFs from `research_papers/`
- ✂️ Split PDF text into chunks (RecursiveCharacterTextSplitter)
- 🧠 Create embeddings with **Ollama** (`nomic-embed-text`)
- 🔎 Store and search vectors using **FAISS**
- 🤖 Generate answers using **Groq** (`llama-3.1-8b-instant`)
- 🧾 Shows retrieved chunks in an expandable section

---

## Project Structure

```

your-project/
│── app.py
│── .env
│── research_papers/
│    ├── paper1.pdf
│    ├── paper2.pdf
│    └── ...
│── README.md

````

---

## Requirements

- Python 3.9+
- Ollama installed and running locally
- Groq API key

### Python Packages

Install dependencies:

```bash
pip install -r requirements.txt
````

If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit python-dotenv langchain langchain-community langchain-groq faiss-cpu pypdf
```

---

## Setup

### 1) Create `.env`

Create a `.env` file in the root folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2) Put PDFs in the folder

Create a folder named `research_papers` and add your PDFs there:

```bash
mkdir research_papers
# add PDFs inside this folder
```

### 3) Start Ollama (Embeddings)

Make sure Ollama is installed and running.

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Check Ollama is working:

```bash
ollama --version
```

---

## Run the App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## How to Use

1. Click **Document Embedding**

   * Loads PDFs from `research_papers/`
   * Splits them into chunks
   * Creates embeddings using Ollama
   * Builds a FAISS vector database

2. Type your question in the input box

3. The app retrieves top similar chunks (`k=4`) and uses Groq LLM to answer

4. Expand **Document similarity search** to view retrieved chunks

---

## Notes / Common Issues

### ✅ “No PDFs found in research_papers/”

* Ensure the folder exists and contains `.pdf` files:

  ```bash
  ls research_papers
  ```

### ✅ Ollama Connection Error (localhost:11434 refused)

* Start Ollama service and retry:

  * On Mac/Linux: open Ollama app or run it from terminal
* Make sure model exists:

  ```bash
  ollama list
  ```

### ✅ “I don't know based on the provided documents.”

This happens when the retriever can’t find relevant chunks.
Try improving retrieval:

* Increase `k`:

  ```python
  retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 8})
  ```
* Use better PDF loader (some PDFs extract text poorly)

---

## Tech Stack

* **Streamlit** — UI
* **LangChain** — RAG pipeline
* **Groq** — LLM inference (`llama-3.1-8b-instant`)
* **OllamaEmbeddings** — vector embeddings (`nomic-embed-text`)
* **FAISS** — vector store
* **PyPDFDirectoryLoader** — PDF loading

