from typing import Dict, List

from dotenv import load_dotenv
import os

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EVALUATOR_MODEL = os.environ["EVALUATOR_MODEL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_SIZE = int(os.environ["EMBEDDING_SIZE"])

# RAGAS imports - modern API
try:
    from ragas.metrics.collections import (
        BleuScore,
        ContextPrecisionWithReference,
        AnswerRelevancy,
        Faithfulness,
        RougeScore,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float] | Dict[str, str]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    else:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        evaluator_llm = llm_factory(
            model=EVALUATOR_MODEL,
            client=openai_client,
        )
        
        # Create RAGAS native modern embeddings
        evaluator_embeddings = OpenAIEmbeddings(
            client=openai_client,
            model=EMBEDDING_MODEL,
        )
        
        # Monkey-patch to inject dimensions into every embedding call
        original_embed_texts = evaluator_embeddings.embed_texts
        
        def patched_embed_texts(texts, **kwargs):
            kwargs['dimensions'] = EMBEDDING_SIZE
            return original_embed_texts(texts, **kwargs)
        
        evaluator_embeddings.embed_texts = patched_embed_texts
        
        # Define metrics
        bleu_metric = BleuScore()
        precision_metric = ContextPrecisionWithReference(llm=evaluator_llm)
        answer_relevancy_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        rouge_metric = RougeScore()
        
        # Score metrics using SYNC API (use .score() not .ascore())
        results = {}
        
        # Non-LLM metrics: BLEU and ROUGE
        bleu_result = bleu_metric.score(
            reference=answer,
            response=answer
        )
        results['bleu_score'] = bleu_result.value if hasattr(bleu_result, 'value') else float(bleu_result)
        
        rouge_result = rouge_metric.score(
            reference=answer,
            response=answer
        )
        results['rouge_score'] = rouge_result.value if hasattr(rouge_result, 'value') else float(rouge_result)
        
        # LLM-based metrics with SYNC API
        precision_result = precision_metric.score(
            user_input=question,
            # response=answer,
            retrieved_contexts=contexts,
            reference=answer
        )
        results['context_precision'] = precision_result.value if hasattr(precision_result, 'value') else float(precision_result)
        
        relevancy_result = answer_relevancy_metric.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )
        results['answer_relevancy'] = relevancy_result.value if hasattr(relevancy_result, 'value') else float(relevancy_result)
        
        faithfulness_result = faithfulness_metric.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )
        results['faithfulness'] = faithfulness_result.value if hasattr(faithfulness_result, 'value') else float(faithfulness_result)
        
        return results