# Homework — Add Ragas Evaluation to the RAG PDF Q&A App

**What we're adding:** a "score my answer" feature. Every time you ask the app a question, you'll
be able to click a button and get back two numbers — **Faithfulness** and **Answer Relevancy** —
that tell you, objectively, whether the answer was actually grounded in your PDF and whether it
actually addressed your question.

**Why this matters:** right now you just *read* the answer and eyeball whether it looks right.
Ragas replaces "looks right to me" with a measured score, computed by an LLM judge, the same
approach real companies use to catch a RAG system quietly getting worse over time.

**You already have everything Ragas needs.** Your app switched from Ollama to the OpenAI API
earlier — Ragas will reuse that same `OPENAI_API_KEY` and the same `gpt-4o-mini` model that's
already answering your questions, just wearing a different hat (judge instead of answerer). No
new account, no new key.

Do all of this inside the `rag-project` folder.

---

## What is Faithfulness? What is Answer Relevancy?

Two completely different questions about the same answer:

- **Faithfulness — "Did the model make anything up?"** Ragas breaks the answer into individual
  factual claims, then checks each one against the chunks your app retrieved from the PDF. Score
  = (claims actually supported by the PDF) ÷ (total claims made). 1.0 means every claim traces
  back to your document. A lower score means the model added something your PDF never said —
  that's a hallucination, caught with a number instead of a guess.

- **Answer Relevancy — "Did it actually answer the question?"** Ragas asks the judge model to
  guess what question this answer would be a good response to, then checks how close that
  guessed question is to the one you actually asked. A rambling or off-topic answer scores low
  even if every word in it is technically true.

**Neither needs a "correct answer" written in advance.** Both are scored purely from the
question, the answer, and the chunks retrieved — exactly the three things your app already
produces on every query. That's why this is a small addition, not a new dataset-building project.

**Rough guide to reading the numbers:** 0.85+ on Faithfulness is strong grounding; below 0.6
usually means real hallucination. 0.8+ on Answer Relevancy is a direct, on-topic answer; a low
score usually means the answer wandered or was too vague.

---

## Step 1 — Install the packages

```
pip install ragas datasets
```

| Package | Why you need it |
|---|---|
| `ragas` | The evaluation library — computes Faithfulness and Answer Relevancy |
| `datasets` | Ragas needs your question/answer/contexts packaged into this format before scoring |

Nothing else changes — you're reusing `langchain-openai`, which you already installed for the
Ollama-to-OpenAI switch.

---

## Step 2 — Create a new file: `rag/evaluator.py`

This is a brand-new file, sitting next to your existing `rag/indexer.py` and `rag/retriever.py`.
Create it with this exact content:

```python
"""Ragas evaluation — scores an answer for faithfulness and relevancy.

Both metrics are reference-free: nobody has to write a "correct answer" in
advance. Ragas uses the question, the answer, and the retrieved chunks you
already have, and asks a judge LLM to score them.
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Reuses the same OpenAI account as the rest of the app -- no new API key needed.
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"

_judge_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
_judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBEDDING_MODEL))


def evaluate_answer(question: str, answer: str, contexts: list) -> dict:
    """Scores one question/answer/context set.

    Returns a dict with 'faithfulness' and 'answer_relevancy', each a float
    from 0.0 to 1.0 (higher is better).
    """
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=_judge_llm,
        embeddings=_judge_embeddings,
    )

    df = result.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].iloc[0]),
        "answer_relevancy": float(df["answer_relevancy"].iloc[0]),
    }
```

**What each part does, line by line:**
- `Dataset.from_dict(...)` — Ragas evaluates a table of rows, even when you only have one row.
  `question`, `answer`, and `contexts` are the three required columns; `contexts` must be a list
  of chunks, matching exactly what `get_answer()` in `rag/retriever.py` already returns as
  `result["sources"]`.
- `LangchainLLMWrapper(ChatOpenAI(...))` — tells Ragas which model acts as the judge. We're
  reusing `gpt-4o-mini`, the same model already answering questions.
- `temperature=0` — makes the judge's scoring consistent. A random judge would give a different
  score to the same answer every time you click the button.
- `evaluate(dataset, metrics=[...], llm=..., embeddings=...)` — the actual scoring call. Runs
  extra OpenAI calls in the background: one full pass per metric.
- `result.to_pandas()` — Ragas returns its own result object; converting to a pandas table is
  the easiest way to pull a single number back out.

---

## Step 3 — Wire it into `app.py`

Two small edits to your existing `app.py`. Nothing in `rag/indexer.py` or `rag/retriever.py`
needs to change.

### Edit 1 — add one import near the top

Find this line (near the top of the file):
```python
from rag.retriever import get_answer
```
Add this line directly below it:
```python
from rag.evaluator import evaluate_answer
```

### Edit 2 — replace the Q&A result block

Find this block near the bottom of `app.py` (it's the part that runs after you click "Ask"):

```python
    if st.session_state.get("query_running"):
        st.session_state["query_running"] = False
        with st.spinner("Searching ChromaDB and generating answer locally…"):
            try:
                result = get_answer(
                    st.session_state["last_question"],
                    persist_dir=PERSIST_DIR,
                )
                st.markdown("### Answer")
                st.markdown(result["answer"])
                with st.expander("Sources — chunks retrieved from ChromaDB"):
                    for i, chunk in enumerate(result["sources"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(chunk)
            except Exception as exc:
                st.error(f"Error: {exc}")
```

Replace the **whole block** with this:

```python
    if st.session_state.get("query_running"):
        st.session_state["query_running"] = False
        with st.spinner("Searching ChromaDB and generating answer…"):
            try:
                result = get_answer(
                    st.session_state["last_question"],
                    persist_dir=PERSIST_DIR,
                )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.session_state["last_result"] = None

    if st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        st.markdown("### Answer")
        st.markdown(result["answer"])
        with st.expander("Sources — chunks retrieved from ChromaDB"):
            for i, chunk in enumerate(result["sources"], 1):
                st.markdown(f"**Chunk {i}**")
                st.text(chunk)

        st.divider()
        if st.button("🧪 Evaluate this answer (Ragas)"):
            with st.spinner("Scoring with Ragas… (2 extra OpenAI calls, a few seconds)"):
                try:
                    scores = evaluate_answer(
                        question=st.session_state["last_question"],
                        answer=result["answer"],
                        contexts=result["sources"],
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                    c2.metric("Answer Relevancy", f"{scores['answer_relevancy']:.2f}")
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
```

**Why the restructure, not just an extra line at the end:** Streamlit reruns your entire script
from top to bottom every time *any* button is clicked — including the new "Evaluate" button. If
the answer were only kept in a local `result` variable, clicking "Evaluate" would wipe it and
you'd see nothing. Saving it to `st.session_state["last_result"]` first means the answer and
sources stay on screen no matter which button you click next.

---

## Step 4 — Run it

```
python -m streamlit run app.py
```

1. Upload a PDF and Build Index, same as before.
2. Ask a question.
3. Once the answer appears, click **🧪 Evaluate this answer (Ragas)**.
4. Two score cards appear — Faithfulness and Answer Relevancy, each 0.00 to 1.00.

Try asking something the PDF genuinely doesn't cover, and evaluate that answer too — a low
Faithfulness score there is Ragas correctly catching a weak or hallucinated answer.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'ragas'` | Step 1 wasn't run, or run in a different Python environment than Streamlit uses | `pip install ragas datasets`, in the same terminal/environment you use to run Streamlit |
| Evaluate button spins for a long time | Normal — each click makes 2+ extra OpenAI calls | Wait 5-10 seconds; this is expected, not a bug |
| `AuthenticationError` when clicking Evaluate | `OPENAI_API_KEY` not set in this terminal session | Same fix as the earlier OpenAI homework — `set OPENAI_API_KEY=...` or your `.env` loader |
| Scores seem inconsistent between runs on the same answer | Judge model has small natural variance even at `temperature=0` | Normal — treat scores as directionally meaningful, not exact to the second decimal |
| `KeyError: 'faithfulness'` | `contexts` passed to `evaluate_answer` was empty or malformed | Make sure you're passing `result["sources"]` — it must be a non-empty list of strings |

**Not scored, not submitted** — same as the rest of `rag-project`. This is a hands-on add-on to
build intuition for what "RAG evaluation" (M2.4) actually looks like in a real app, not a graded
deliverable.
