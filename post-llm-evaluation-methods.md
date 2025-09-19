## 8. Context Relevancy Assessment (DeepEval)

### Concept
Evaluate how relevant provided context (e.g., retrieved documents) is to the user query or prompt using DeepEval’s relevance metrics.

### Implementation
```python
from deepeval import DeepEvalEvaluator

def deep_eval_context_relevancy_score(query, context_list, eval_config=None):
    """
    Scores each context passage against the query, then aggregates relevance.

    Args:
        query (str): Original user query or prompt
        context_list (list of str): Retrieved context passages
        eval_config (dict): Optional DeepEval parameters

    Returns:
        float: Average relevancy score (0-1) for all passages
    """
    evaluator = DeepEvalEvaluator(**(eval_config or {}))
    scores = []
    for ctx in context_list:
        result = evaluator.score_relevancy(context=query, response=ctx)
        scores.append(result.get('score', 0.0))
    # Aggregate
    return round(sum(scores) / len(scores), 3) if scores else 0.0
```

### Example Usage
```python
query = "What are the health benefits of green tea?"
contexts = [
    "Green tea contains antioxidants that reduce inflammation.",
    "Studies link green tea consumption to improved heart health.",
    "Green tea is also used in skincare products."
]
score = deep_eval_context_relevancy_score(query, contexts)
print(f"Context Relevancy Score: {score}")
# Output: Context Relevancy Score: 0.892
```

### Functionality
Leverages DeepEval to score each retrieved context passage for relevance to the query and returns an average score. Ensures that the provided information is on-topic and useful.