# Homework — Switch the RAG PDF Q&A App from Ollama to OpenAI

You already have this app running locally with Ollama (100% offline, free, no API key). This
homework switches it to use the OpenAI API instead — same app, same features, different
"engine" underneath. This is the exact same swap LangChain is built for: change which model
plugs in, keep everything else the same.

**Do this in the `rag-project` folder.**

---

## Step 1 — Install the packages

Open Command Prompt in the `rag-project` folder and run these, in order:

```
pip install streamlit
pip install -r requirements.txt
pip install langchain-community
pip install langchain-openai
pip install python-dotenv
```

| Package | Why you need it |
|---|---|
| `streamlit` | Runs the web app itself |
| `requirements.txt` | Installs everything else the project needs (LangChain, ChromaDB, pypdf, etc.) |
| `langchain-community` | Loads PDFs (`PyPDFLoader`) |
| `langchain-openai` | **New for this homework** — lets the app call OpenAI instead of Ollama |
| `python-dotenv` | **New for this homework** — lets the app read your API key from a `.env` file |

You do **not** need to install Ollama for this homework (`winget install Ollama.Ollama`) —
that was only for the original local version. Skip it unless you also want to keep the
Ollama version working side by side.

---

## Step 2 — Set your OpenAI API key

**Option A — quick, one terminal session only.** Run this in Command Prompt *before* you run
the app. You'll need to re-type it every time you open a new terminal window:

```
set OPENAI_API_KEY=sk-your-key-here
```

**Option B — permanent, survives closing the terminal.** Run this once. You must close and
reopen Command Prompt afterward for it to take effect:

```
setx OPENAI_API_KEY "sk-your-key-here"
```

**Option C — reuse your Day 1/2 `.env` file (recommended if you already have one).**
Add this to the very top of `app.py`, before the `from rag...` imports:

```python
from pathlib import Path
from dotenv import load_dotenv

def find_env():
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ".env").is_file():
            return candidate / ".env"
    return None

_env = find_env()
if _env:
    load_dotenv(_env)
```
This walks up from `app.py`'s own folder looking for a `.env` file — the same one you already
made for Day 1/2's OpenAI labs. No new key file needed.

---

## Step 3 — Make the code changes

Two files, four lines total.

### `rag/indexer.py`

```diff
- from langchain_ollama import OllamaEmbeddings
+ from langchain_openai import OpenAIEmbeddings

- EMBEDDING_MODEL = "nomic-embed-text"
+ EMBEDDING_MODEL = "text-embedding-3-small"

def _get_embeddings():
-   return OllamaEmbeddings(model=EMBEDDING_MODEL)
+   return OpenAIEmbeddings(model=EMBEDDING_MODEL)
```

### `rag/retriever.py`

```diff
- from langchain_ollama import ChatOllama
+ from langchain_openai import ChatOpenAI

- GENERATION_MODEL = "llama3.2:1b"
+ GENERATION_MODEL = "gpt-4o-mini"

def get_qa_chain(persist_dir: str = "chroma_db"):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=_get_embeddings())
-   llm = ChatOllama(model=GENERATION_MODEL, temperature=0)
+   llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0)
```

Nothing else in either file changes — `RetrievalQA`, `Chroma`, the chunking logic, all stay
exactly the same. This is the whole point of LangChain's design: the model is a plug-in, not
wired through the rest of the code.

---

## Step 4 — Clear the old index

Your existing `chroma_db/` folder was built using Ollama's `nomic-embed-text` embeddings.
OpenAI's `text-embedding-3-small` produces vectors of a different size — mixing the two in one
collection causes a dimension-mismatch error.

Delete the `chroma_db` folder inside `rag-project`, or click **Clear Index** in the app's
sidebar, before you rebuild.

---

## Step 5 — Run it

```
python -m streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a PDF, click **Build Index**, ask a question — same
app, now answering via OpenAI instead of Ollama.

---

## Troubleshooting

| Error | What it means | Fix |
|---|---|---|
| `model "text-embedding-3-small" not found, try pulling it first` | This is Ollama's own error — the code is still calling `OllamaEmbeddings`, just with the new model name. A half-finished Step 3. | Check `rag/indexer.py` — the **import** and the **class name** both need to change, not just the model string. |
| `ModuleNotFoundError: No module named 'langchain_openai'` | Step 1 wasn't fully run | `pip install langchain-openai` |
| `AuthenticationError` / `Incorrect API key` | Key not set, or set in a different terminal than the one running Streamlit | Re-check Step 2 — `set`/`setx` only affects the terminal window it was run in |
| Dimension-mismatch error while indexing | Old `chroma_db/` folder still has Ollama-sized vectors | Delete `chroma_db/`, rebuild (Step 4) |

**Submit:** a screenshot of the app answering a question about your uploaded PDF, showing the
sidebar or terminal confirming OpenAI (not Ollama) was used.
