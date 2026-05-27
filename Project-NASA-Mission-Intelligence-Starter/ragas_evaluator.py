from typing import Dict, List, Optional

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS imports
try:
    from ragas import evaluate, SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        BleuScore,
        # NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "386"))

def evaluate_response_quality(question: str, answer: str, contexts: List[str], reference_contexts: List[str]|None=None) -> Dict[str, float] | Dict[str, str] :
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # 1) Create evaluator LLM (sync) and wrap it for Ragas
    chat_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY, 
    )
    evaluator_llm = LangchainLLMWrapper(chat_llm)

    # 2) Create evaluator embeddings and wrap them
    #    Replace "text-embedding-3-small" with your env var if needed.
    lc_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_SIZE, 
        api_key=OPENAI_API_KEY,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    # 3) Define metrics (some use LLM, some use embeddings, BLEU/ROUGE are non-LLM)
    metrics = []
    bleu_metric = BleuScore()
    metrics.append(bleu_metric)
    if reference_contexts is not None:
        context_precision_metric = NonLLMContextPrecisionWithReference()
        metrics.append(context_precision_metric)
    response_relevancy_metric = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    metrics.append(response_relevancy_metric)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    metrics.append(faithfulness_metric)
    rouge_metric = RougeScore()
    metrics.append(rouge_metric)


    # 4) Build a SingleTurnSample for this one Q/A pair
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer,
        reference_contexts=reference_contexts,
    )

    evaluation_dataset = EvaluationDataset(samples=[sample])


    # 5) Evaluate (synchronous) — returns a pandas DataFrame-like object
    result_df = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
    ).to_pandas()  # Convert to pandas DataFrame for easier handling

    # 6) Convert result row into a plain dict of floats
    #    Column names usually match the metric names (bleu_score, context_precision, etc.)
    row = result_df.iloc[0]
    results: Dict[str, float] = {}
    for col in result_df.columns:
        # skip metadata columns like "user_input", "response" if they exist
        if isinstance(row[col], (int, float)):
            results[col] = float(row[col])

    return results