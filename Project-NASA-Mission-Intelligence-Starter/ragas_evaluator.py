from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    from ragas import EvaluationResults
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float] | Dict[str, str]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    else:
    
        # TODO: Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
        # TODO: Create evaluator_embeddings with model test-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
        # TODO: Define an instance for each metric to evaluate
        bleu_metric = BleuScore() # type: ignore
        precision_metric = NonLLMContextPrecisionWithReference(evaluator_llm, evaluator_embeddings) # type: ignore
        relevancy_metric = ResponseRelevancy(evaluator_llm) # type: ignore
        faithfulness_metric = Faithfulness(evaluator_llm, evaluator_embeddings) # type: ignore
        rouge_metric = RougeScore() # type: ignore   
        
        # TODO: Evaluate the response using the metrics
        
        sample = SingleTurnSample( # type: ignore
            user_input = question,
            retrieved_contexts = contexts,
            response = answer
        ) 
        evaluation_results = evaluate( # type: ignore
            sample, # type: ignore
            [bleu_metric, precision_metric, relevancy_metric, faithfulness_metric, rouge_metric],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        # TODO: Return the evaluation results
        evaluation_scores = evaluation_results.scores[0]
        return {
            key: float(value)
            for key, value in evaluation_scores.items()
            if isinstance(value, (int, float)) and value == value
        }
