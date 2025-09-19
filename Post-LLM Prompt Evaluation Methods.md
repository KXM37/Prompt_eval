<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Post-LLM Prompt Evaluation Methods

## 1. Simple String Matching

### Implementation

```python
def simple_string_matching_score(actual_output, expected_output, method="exact"):
    """
    Returns a 0-1 score based on string matching between actual and expected outputs.
    
    Args:
        actual_output (str): Generated output from LLM
        expected_output (str): Expected/reference output
        method (str): "exact", "contains", "regex", or "similarity"
        
    Returns:
        float: Match score (0-1)
    """
    import re
    from difflib import SequenceMatcher
    
    actual = actual_output.strip().lower()
    expected = expected_output.strip().lower()
    
    if method == "exact":
        return 1.0 if actual == expected else 0.0
    
    elif method == "contains":
        return 1.0 if expected in actual else 0.0
    
    elif method == "regex":
        # Treat expected as regex pattern
        try:
            return 1.0 if re.search(expected, actual, re.IGNORECASE) else 0.0
        except re.error:
            return 0.0
    
    elif method == "similarity":
        # Character-level similarity using SequenceMatcher
        return round(SequenceMatcher(None, actual, expected).ratio(), 3)
    
    else:
        raise ValueError("Method must be 'exact', 'contains', 'regex', or 'similarity'")
```


### Functionality

Provides four string matching strategies: exact match (binary), substring containment, regex pattern matching, and character-level similarity scoring. Essential for validating specific required phrases or format compliance in LLM outputs.

***

## 2. Basic Length/Format Validation

### Implementation

```python
def format_validation_score(output, min_length=10, max_length=200, required_phrases=None):
    """
    Returns a 0–1 score based on length and presence of required phrases.
    
    Args:
        output (str): LLM-generated text
        min_length (int): Minimum word count
        max_length (int): Maximum word count
        required_phrases (list): Optional list of strings that must appear
        
    Returns:
        float: Composite score (0–1)
    """
    words = output.split()
    wc = len(words)
    
    # Length scoring: optimal range [min_length, max_length]
    if min_length <= wc <= max_length:
        length_score = 1.0
    elif wc < min_length:
        length_score = wc / min_length
    else:
        length_score = max(0.0, 1.0 - (wc - max_length) / max_length)
    
    # Required phrases scoring
    if required_phrases:
        found = sum(1 for p in required_phrases if p.lower() in output.lower())
        phrase_score = found / len(required_phrases)
    else:
        phrase_score = 1.0
    
    # Composite: 70% length, 30% phrases
    return round(0.7 * length_score + 0.3 * phrase_score, 3)
```


### Functionality

Ensures outputs fall within acceptable length bounds and include critical sections or headings. Useful for enforcing output constraints in applications like report generation.

***

## 3. Traditional NLP Metrics (BLEU/ROUGE on Outputs)

### Implementation

```python
def bleu_rouge_scores(output, reference):
    """
    Computes BLEU and ROUGE-L scores between output and reference.
    
    Args:
        output (str): Generated text
        reference (str): Ground-truth text
        
    Returns:
        dict: {'bleu': float, 'rouge_l': float}
    """
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    
    # BLEU
    weights = (0.25, 0.25, 0.25, 0.25)  # up to 4-grams
    ref_tokens = reference.split()
    out_tokens = output.split()
    bleu = sentence_bleu([ref_tokens], out_tokens, weights=weights)
    
    # ROUGE-L
    rouge = Rouge(metrics=['rouge-l'])
    scores = rouge.get_scores(output, reference)[0]['rouge-l']['f']
    
    return {'bleu': round(bleu, 3), 'rouge_l': round(scores, 3)}
```


### Functionality

Applies BLEU and ROUGE-L metrics to measure n-gram overlap and longest common subsequence, giving a quantitative similarity score against ground truth.

***

## 4. Keyword Presence in Output

### Implementation

```python
def output_keyword_score(output, keywords):
    """
    Returns the fraction of required keywords found in the output.
    
    Args:
        output (str): Generated text
        keywords (list): Required keywords
        
    Returns:
        float: 0–1 coverage score
    """
    out_low = output.lower()
    found = sum(1 for kw in keywords if kw.lower() in out_low)
    return round(found / len(keywords), 3)
```


### Functionality

Ensures critical terms appear in outputs, promoting completeness and focus on required concepts.

***

## 5. JSON Structure Validation

### Implementation

```python
def json_structure_score(output_json, schema):
    """
    Returns 0–1 score based on how many required fields are present.
    
    Args:
        output_json (dict): Parsed JSON output
        schema (dict): Dict where keys are required fields
        
    Returns:
        float: Field coverage ratio
    """
    required = set(schema.keys())
    present = set(output_json.keys()) & required
    return round(len(present) / len(required), 3)
```


### Functionality

Checks completeness of structured outputs, useful for APIs and data-extraction tasks.

***

## 6. Answer Relevancy (DeepEval)

### Implementation

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

def deep_eval_answer_relevancy_score(query, answer, eval_config=None):
    """
    Uses DeepEval's AnswerRelevancyMetric to score how well the answer addresses the query.
    
    Args:
        query (str): Original user query/prompt
        answer (str): Generated answer/output
        eval_config (dict): Optional DeepEval configuration
        
    Returns:
        float: Relevancy score (0-1)
    """
    config = eval_config or {}
    metric = AnswerRelevancyMetric(
        threshold=config.get('threshold', 0.7),
        model=config.get('model', 'gpt-3.5-turbo')
    )
    
    # Create test case format expected by DeepEval
    test_case = {
        'input': query,
        'actual_output': answer
    }
    
    result = evaluate([test_case], [metric])
    return round(result[0].score, 3) if result else 0.0
```


### Functionality

Leverages DeepEval's sophisticated relevancy assessment to determine how well generated answers address the original query, using configurable LLM-based evaluation.

***

## 7. Factuality Assessment (PromptFoo)

### Implementation

```python
def promptfoo_factuality_score(output, ground_truth=None, config=None):
    """
    Uses PromptFoo's factuality checking to verify claims in output.
    
    Args:
        output (str): Generated text to check
        ground_truth (str): Optional reference facts
        config (dict): PromptFoo configuration options
        
    Returns:
        dict: {'factuality_score': float, 'verified_claims': int, 'total_claims': int}
    """
    import subprocess
    import json
    import tempfile
    
    # Create temporary config for PromptFoo
    config = config or {
        'providers': ['openai:gpt-3.5-turbo'],
        'tests': [{
            'vars': {'output': output, 'reference': ground_truth or ''},
            'assert': [{
                'type': 'factuality',
                'threshold': 0.8
            }]
        }]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    try:
        # Run PromptFoo evaluation
        result = subprocess.run(
            ['promptfoo', 'eval', '-c', config_path, '--output', 'json'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            eval_result = json.loads(result.stdout)
            score = eval_result.get('results', [{}])[0].get('score', 0.0)
            return {
                'factuality_score': round(score, 3),
                'verified_claims': eval_result.get('verified_claims', 0),
                'total_claims': eval_result.get('total_claims', 0)
            }
        else:
            return {'factuality_score': 0.0, 'verified_claims': 0, 'total_claims': 0}
    
    except Exception:
        # Fallback scoring if PromptFoo unavailable
        return {'factuality_score': 0.5, 'verified_claims': 0, 'total_claims': 0}
```


### Functionality

Integrates with PromptFoo's factuality evaluation to verify claims against knowledge bases or reference materials, returning detailed accuracy metrics.

***

## 8. Context Relevancy Assessment (DeepEval)

### Implementation

```python
from deepeval.metrics import ContextualRelevancyMetric

def deep_eval_context_relevancy_score(query, context_list, eval_config=None):
    """
    Scores how relevant provided context is to the query using DeepEval.
    
    Args:
        query (str): Original user query or prompt
        context_list (list): Retrieved context passages
        eval_config (dict): Optional DeepEval parameters
        
    Returns:
        float: Average relevancy score (0-1) for all passages
    """
    config = eval_config or {}
    metric = ContextualRelevancyMetric(
        threshold=config.get('threshold', 0.7),
        model=config.get('model', 'gpt-3.5-turbo')
    )
    
    scores = []
    for context in context_list:
        test_case = {
            'input': query,
            'retrieval_context': [context]
        }
        
        result = evaluate([test_case], [metric])
        if result:
            scores.append(result[0].score)
    
    return round(sum(scores) / len(scores), 3) if scores else 0.0
```


### Functionality

Leverages DeepEval to score each retrieved context passage for relevance to the query and returns an average score. Ensures that provided information is on-topic and useful.

***

## 9. LLM-as-a-Judge Evaluation

### Implementation

```python
def llm_judge_score(output, criteria, judge_model, context=None):
    """
    Uses another LLM to evaluate the quality of generated output based on specified criteria.
    
    Args:
        output (str): Generated text to evaluate
        criteria (str): Evaluation criteria (e.g., "accuracy, clarity, completeness")
        judge_model (callable): Function that takes prompt and returns LLM response
        context (str): Optional context or reference material
        
    Returns:
        dict: {'overall_score': float, 'detailed_feedback': str}
    """
    
    evaluation_prompt = f"""
    Please evaluate the following output based on these criteria: {criteria}
    
    {"Context: " + context if context else ""}
    
    Output to evaluate:
    {output}
    
    Please provide:
    1. A score from 0.0 to 1.0 (where 1.0 is excellent)
    2. Brief feedback explaining the score
    
    Format your response as:
    Score: X.X
    Feedback: [your feedback here]
    """
    
    judge_response = judge_model(evaluation_prompt)
    
    # Extract score and feedback
    try:
        lines = judge_response.split('\n')
        score_line = next(line for line in lines if line.startswith('Score:'))
        feedback_line = next(line for line in lines if line.startswith('Feedback:'))
        
        score = float(score_line.split(':')[1].strip())
        feedback = feedback_line.split(':', 1)[1].strip()
        
        return {
            'overall_score': round(score, 3),
            'detailed_feedback': feedback
        }
    except:
        # Fallback if parsing fails
        return {
            'overall_score': 0.5,
            'detailed_feedback': "Could not parse judge response"
        }
```


### Functionality

Employs a separate LLM to evaluate outputs against custom criteria, providing both numerical scores and qualitative feedback. Highly flexible and can adapt to domain-specific requirements.

***

## 10. Faithfulness Evaluation (DeepEval)

### Implementation

```python
from deepeval.metrics import FaithfulnessMetric

def deep_eval_faithfulness_score(output, context, eval_config=None):
    """
    Measures how faithfully the output adheres to the provided context using DeepEval.
    
    Args:
        output (str): Generated answer/response
        context (str): Source context or retrieved information
        eval_config (dict): Optional DeepEval configuration
        
    Returns:
        float: Faithfulness score (0-1)
    """
    config = eval_config or {}
    metric = FaithfulnessMetric(
        threshold=config.get('threshold', 0.7),
        model=config.get('model', 'gpt-3.5-turbo')
    )
    
    test_case = {
        'input': 'Generate response based on context',
        'actual_output': output,
        'retrieval_context': [context]
    }
    
    result = evaluate([test_case], [metric])
    return round(result[0].score, 3) if result else 0.0
```


### Functionality

Uses DeepEval's faithfulness metric to ensure generated outputs don't contradict or fabricate information beyond what's provided in the source context.

***

## 11. Hallucination Detection

### Implementation

```python
def hallucination_detection_score(output, knowledge_base=None, fact_check_model=None):
    """
    Detects potential hallucinations in LLM output using fact-checking approaches.
    
    Args:
        output (str): LLM-generated text to check
        knowledge_base (dict): Optional dictionary of facts for verification
        fact_check_model (callable): Optional fact-checking model function
        
    Returns:
        dict: {
            'hallucination_score': float,  # 0-1, lower = more hallucinations
            'flagged_sentences': List[str], # Sentences flagged as potential hallucinations
            'confidence': float             # Overall confidence in detection
        }
    """
    import nltk
    import re
    
    sentences = nltk.sent_tokenize(output)
    flagged = []
    confidence_scores = []
    
    for sentence in sentences:
        is_hallucination = False
        confidence = 0.5
        
        # Knowledge base verification
        if knowledge_base:
            key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            for term in key_terms:
                if term in knowledge_base:
                    if knowledge_base[term].lower() not in sentence.lower():
                        is_hallucination = True
                        confidence = 0.8
                        break
        
        # Fact-checking model
        if fact_check_model and not is_hallucination:
            try:
                fact_result = fact_check_model(sentence)
                if isinstance(fact_result, dict):
                    is_hallucination = fact_result.get('is_false', False)
                    confidence = fact_result.get('confidence', 0.5)
                else:
                    is_hallucination = bool(fact_result)
                    confidence = 0.7
            except:
                confidence = 0.3
        
        # Pattern-based detection (fallback)
        if not knowledge_base and not fact_check_model:
            suspicious_patterns = [
                r'\d{4}-\d{4}', r'exactly \d+', r'studies show', r'research proves'
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, sentence, re.I):
                    is_hallucination = True
                    confidence = 0.4
                    break
        
        if is_hallucination:
            flagged.append(sentence)
        
        confidence_scores.append(confidence)
    
    hallucination_ratio = len(flagged) / len(sentences) if sentences else 0
    hallucination_score = 1.0 - hallucination_ratio
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    return {
        'hallucination_score': round(hallucination_score, 3),
        'flagged_sentences': flagged,
        'confidence': round(overall_confidence, 3)
    }
```


### Functionality

Implements multiple approaches to detect hallucinations: knowledge base verification, fact-checking model integration, and pattern-based suspicious content detection. Returns a score where higher values indicate fewer hallucinations.

***

## 12. Bias Assessment

### Implementation

```python
def bias_assessment_score(output, bias_model=None, attributes=None):
    """
    Scores output for bias across specified attributes.
    
    Args:
        output (str): Generated text
        bias_model (callable): Function that returns bias metrics for text
        attributes (list): List of attributes to check (e.g., ['gender', 'race'])
    
    Returns:
        dict: {
            'bias_score': float,    # 0–1, higher = less biased
            'details': dict         # Per-attribute bias metrics
        }
    """
    if not bias_model:
        return {'bias_score': 1.0, 'details': {attr: 1.0 for attr in (attributes or [])}}
    
    details = {}
    for attr in attributes or []:
        details[attr] = bias_model(output, attribute=attr)
    
    bias_score = sum(details.values()) / len(details) if details else 1.0
    return {'bias_score': round(bias_score, 3), 'details': details}
```


### Functionality

Invokes a bias detection model for each attribute, aggregates per-attribute scores into an overall bias score (higher = fairer language).

***

## 13. Toxicity Detection

### Implementation

```python
def toxicity_detection_score(output, toxicity_model):
    """
    Returns a toxicity score (0–1) where lower = more toxic.
    
    Args:
        output (str): Generated text
        toxicity_model (callable): Returns toxicity probability for text
    
    Returns:
        float: 1 - toxicity_probability
    """
    tox_prob = toxicity_model(output)
    return round(1.0 - tox_prob, 3)
```


### Functionality

Uses a toxicity classifier to return a safety score (higher = safer text).

***

## 14. A/B Testing Framework

### Implementation

```python
def ab_test(prompts, inputs, model, metric_fn):
    """
    Conducts A/B test between two prompts.
    
    Args:
        prompts (tuple): (prompt_A, prompt_B)
        inputs (list): Sample inputs
        model: LLM function taking prompt+input
        metric_fn: Function that scores output against reference
        
    Returns:
        dict: {
            'scores_A': list, 'scores_B': list, 'mean_A': float, 'mean_B': float,
            'better': 'A' | 'B' | 'Tie'
        }
    """
    A, B = prompts
    scores_A, scores_B = [], []
    for inp in inputs:
        out_A = model(A + " " + inp)
        out_B = model(B + " " + inp)
        scores_A.append(metric_fn(out_A, inp))
        scores_B.append(metric_fn(out_B, inp))
    mean_A = sum(scores_A)/len(scores_A)
    mean_B = sum(scores_B)/len(scores_B)
    if mean_A > mean_B: better = 'A'
    elif mean_B > mean_A: better = 'B'
    else: better = 'Tie'
    return {'scores_A': scores_A, 'scores_B': scores_B,
            'mean_A': round(mean_A, 3), 'mean_B': round(mean_B, 3),
            'better': better}
```


### Functionality

Runs both prompt variants on the same inputs, scores each output, and determines which prompt performs better based on mean metric.

***

## 15. Multi-turn Conversation Evaluation

### Implementation

```python
def multi_turn_evaluation(conversation, model, eval_fn):
    """
    Evaluates a multi-turn conversation.
    
    Args:
        conversation (list): Alternating user and assistant messages
        model: LLM function for simulating assistant
        eval_fn: Function to score each assistant response
        
    Returns:
        list: Scores per assistant turn
    """
    scores = []
    for i in range(1, len(conversation), 2):
        prompt = "\n".join(conversation[:i])
        response = model(prompt)
        scores.append(eval_fn(response, conversation[i]))
    return scores
```


### Functionality

Scores each assistant turn in a dialogue context, enabling evaluation of coherence, relevance, and consistency across exchanges.

***

## 16. Advanced RAG Pipeline Evaluation

### Implementation

```python
def rag_evaluation(query, retriever, generator, eval_fn):
    """
    Evaluates a RAG pipeline end-to-end.
    
    Args:
        query (str): User query
        retriever: Function returning context passages
        generator: Function generating answer from passages
        eval_fn: Function scoring generated answer against ground truth
        
    Returns:
        dict: {
            'retrieval_score': float,
            'generation_score': float,
            'overall_score': float
        }
    """
    contexts = retriever(query)
    retrieval_score = eval_fn(contexts, query)
    answer = generator(query, contexts)
    generation_score = eval_fn(answer, query)
    overall_score = round(0.5*retrieval_score + 0.5*generation_score, 3)
    return {
        'retrieval_score': round(retrieval_score,3),
        'generation_score': round(generation_score,3),
        'overall_score': overall_score
    }
```


### Functionality

Combines retrieval relevance and answer quality into a single evaluation, enabling end-to-end RAG system assessment.

***

This comprehensive file covers all post-LLM evaluation methods from simple string matching to complex RAG pipeline evaluation, with extensive use of DeepEval and PromptFoo frameworks where applicable.

