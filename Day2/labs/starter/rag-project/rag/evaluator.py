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