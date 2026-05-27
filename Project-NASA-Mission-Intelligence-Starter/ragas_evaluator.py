from typing import Dict, List

from dotenv import load_dotenv
import os

from openai import OpenAI
from ragas.llms import llm_factory

from ragas.embeddings import embedding_factory

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


EVALUATOR_MODEL = "gpt-4o-mini"
# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from typing import Sequence, cast
    from ragas.metrics import Metric
    from ragas.metrics.collections import (
        BleuScore,
        ContextPrecisionWithReference,   # renamed from NonLLMContextPrecisionWithReference
        AnswerRelevancy,                 # renamed from ResponseRelevancy
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
    
        # TODO: Create evaluator LLM with model gpt-3.5-turbo
        # evaluator LLM for collections metrics
        load_dotenv()
        openai_api_key = os.environ["OPENAI_API_KEY"]
        openai_client = OpenAI(api_key=openai_api_key)  # uses OPENAI_API_KEY
        evaluator_llm = llm_factory(
                                    model=EVALUATOR_MODEL,   # or "gpt-4o" etc.
                                    client=openai_client,  # required
                                    # provider="openai"     # default is "openai"
                                    )
        # TODO: Create evaluator_embeddings with model test-embedding-3-small
        # Modern embeddings
        evaluator_embeddings = embedding_factory(
            provider="openai",                      # or omit, defaults to "openai"
            model="text-embedding-3-small",        # or "text-embedding-ada-002"
            client=openai_client,
            interface="modern",                    # match collections pipeline
        )
        # TODO: Define an instance for each metric to evaluate
        bleu_metric = BleuScore() # type: ignore
        precision_metric = ContextPrecisionWithReference(evaluator_llm, evaluator_embeddings) # type: ignore
        answer_relevancy_metric = AnswerRelevancy(evaluator_llm, evaluator_embeddings) # type: ignore
        faithfulness_metric = Faithfulness(evaluator_llm, evaluator_embeddings) # type: ignore
        rouge_metric = RougeScore() # type: ignore
        
        
        
        metrics: Sequence[Metric] = cast(Sequence[Metric], [ rouge_metric ])   # type: ignore
        # metrics: Sequence[Metric] = [BleuScore(), RougeScore()]
        
        # TODO: Evaluate the response using the metrics
        
        sample = SingleTurnSample( # type: ignore
            user_input = question,
            retrieved_contexts = contexts,
            response = answer
        )
        
        dataset = EvaluationDataset(samples=[sample])# type: ignore

        evaluation_results = evaluate(# type: ignore
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
        ) 
        
        # TODO: Return the evaluation results
        evaluation_scores = evaluation_results.scores[0]
        return {
            key: float(value)
            for key, value in evaluation_scores.items()
            if isinstance(value, (int, float)) and value == value
        }
