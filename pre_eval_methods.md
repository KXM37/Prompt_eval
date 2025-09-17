# Pre-LLM Evaluation Methods

A comprehensive collection of automated prompt evaluation methods that assess prompt quality before LLM execution. These methods are ordered from simplest to most complex in terms of implementation and explainability.

## 1. Prompt Length Analysis
**Complexity Level:** Very Simple

### Function Code
```python
def prompt_length_score(prompt, tokenizer=None):
    """
    Comprehensive prompt length analysis with composite scoring.
    Returns a normalized score from 0-1 where higher is better.
    """
    import re
    
    # Basic metrics
    char_count, word_count = len(prompt), len(prompt.split())
    sentences = len([s.strip() for s in re.split(r'[.!?]+', prompt.strip()) if s.strip()])
    token_count = len(tokenizer.encode(prompt)) if tokenizer else int(word_count * 1.3)
    
    # Nested depth calculation
    paren_depth = max(prompt[:i].count('(') - prompt[:i].count(')') for i in range(len(prompt)+1))
    subordinating_words = ['that','which','who','because','since','if','although','when','where','while']
    subordinating = sum(1 for w in subordinating_words if w in prompt.lower())
    nested_depth = paren_depth + min(subordinating, 3) + min(prompt.count(',')//3, 2)
    
    # Normalization functions (optimal ranges based on research)
    token_score = 1.0 if 50<=token_count<=400 else max(0.1, min(token_count/50, 1.0-(token_count-400)/600))
    word_score = 1.0 if 10<=word_count<=100 else max(0.2, min(word_count/10, 1.0-(word_count-100)/200))
    sentence_score = 1.0 if 1<=sentences<=8 else max(0.3, 1.0-(sentences-8)/12) if sentences>8 else 0.0
    nesting_score = 1.0 if nested_depth<=2 else max(0.1, 1.0-(nested_depth-2)/3)
    
    # Weighted composite score
    return round(0.4*token_score + 0.25*word_score + 0.2*nesting_score + 0.15*sentence_score, 3)
```

### Example Implementation
```python
score = prompt_length_score("Write a detailed 500-word analysis of climate change impacts on agriculture.")
print(score)  # Expected: high score for optimal length
```

### Functionality
Comprehensive prompt length analysis that evaluates multiple structural dimensions and returns a normalized 0-1 score. Considers token count (with optional tokenizer), word count, sentence count, and nested complexity (parentheses, subordinating clauses, commas). Uses research-based optimal ranges to score each dimension and combines them into a weighted composite score where higher values indicate better prompt structure.

---

## 2. Readability Analysis
**Complexity Level:** Simple

### Function Code
```python
def readability_score(prompt):
    """
    Analyzes prompt readability using multiple established metrics.
    Returns a normalized score from 0-1 where higher is better.
    """
    import textstat
    import re
    
    # Basic text statistics
    words = len(prompt.split())
    sentences = len([s.strip() for s in re.split(r'[.!?]+', prompt.strip()) if s.strip()])
    syllables = textstat.syllable_count(prompt)
    
    if sentences == 0 or words == 0:
        return 0.0
    
    # Core readability metrics
    flesch_ease = textstat.flesch_reading_ease(prompt)  # 0-100 scale
    flesch_grade = textstat.flesch_kincaid_grade(prompt)  # Grade level
    avg_sentence_length = words / sentences
    avg_word_length = sum(len(word) for word in prompt.split()) / words
    
    # Normalize individual metrics (optimal ranges for prompts)
    ease_score = max(0.0, min(1.0, flesch_ease / 100))  # Higher flesch_ease = better
    grade_score = 1.0 if 6<=flesch_grade<=12 else max(0.1, 1.0-abs(flesch_grade-9)/6)
    sentence_len_score = 1.0 if 10<=avg_sentence_length<=20 else max(0.2, 1.0-abs(avg_sentence_length-15)/10)
    word_len_score = 1.0 if 4<=avg_word_length<=6 else max(0.3, 1.0-abs(avg_word_length-5)/3)
    
    # Weighted composite score
    return round(0.35*ease_score + 0.25*grade_score + 0.25*sentence_len_score + 0.15*word_len_score, 3)
```

### Example Implementation
```python
score = readability_score("Please write a clear explanation of machine learning concepts.")
print(score)  # Expected: high score for readable text
```

### Functionality
Evaluates prompt readability using established linguistic metrics including Flesch Reading Ease, Flesch-Kincaid Grade Level, average sentence length, and average word length. Normalizes each metric based on optimal ranges for prompt comprehension (6-12th grade reading level, 10-20 words per sentence, 4-6 characters per word) and combines into a weighted score where higher values indicate better readability.

---

## 3. Lexical Diversity Score
**Complexity Level:** Simple-Moderate

### Function Code
```python
def lexical_diversity_score(prompt):
    """
    Calculates lexical diversity (type-token ratio) and returns
    a normalized 0–1 score where higher indicates richer vocabulary.
    """
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(prompt.lower())
    if not tokens:
        return 0.0

    types = set(tokens)
    ttr = len(types) / len(tokens)  # Type-Token Ratio

    # Normalize: optimal diversity between 0.5 and 0.8
    if ttr >= 0.8:
        score = 1.0
    elif ttr <= 0.5:
        score = 0.0
    else:
        score = (ttr - 0.5) / 0.3  # maps [0.5–0.8]→[0–1]

    return round(score, 3)
```

### Example Implementation
```python
score = lexical_diversity_score("Create innovative solutions using diverse methodologies and varied approaches.")
print(score)  # Expected: high score for diverse vocabulary
```

### Functionality
Tokenizes prompt to count total tokens and unique types. Computes type-token ratio (TTR). Normalizes TTR so that values below 0.5 score zero, above 0.8 score one, with linear mapping in between.

---

## 4. Specificity & Constraint Presence
**Complexity Level:** Moderate

### Function Code
```python
def specificity_score(prompt):
    """
    Evaluates prompt specificity by detecting detail indicators,
    numeric constraints, and example language. Returns normalized
    score 0-1 where higher indicates greater specificity.
    """
    import re
    
    detail_indicators = [
        "specifically", "exactly", "precisely", "in detail",
        "comprehensive", "thorough"
    ]
    constraint_patterns = [
        r"\b\d+\s*-\s*\d+\b",      # ranges like 300-500
        r"\b(at least|no more than|no less than)\b",
        r"\bmaximum|min|max|required\b"
    ]
    example_indicators = ["for example", "for instance", "such as", "e.g.", "i.e."]
    
    text = prompt.lower()
    
    # Count occurrences
    detail_count = sum(text.count(w) for w in detail_indicators)
    numeric_matches = sum(len(re.findall(p, text)) for p in constraint_patterns)
    example_count = sum(text.count(w) for w in example_indicators)
    
    # Normalize counts to 0-1 based on thresholds
    detail_score = min(1.0, detail_count / 3)  # 3+ indicators => 1.0
    constraint_score = min(1.0, numeric_matches / 2)  # 2+ matches => 1.0
    example_score = min(1.0, example_count / 2)  # 2+ examples => 1.0
    
    # Weighted composite
    return round(0.4*constraint_score + 0.35*detail_score + 0.25*example_score, 3)
```

### Example Implementation
```python
score = specificity_score("Write exactly 500 words about machine learning, specifically focusing on neural networks. For example, discuss CNNs and RNNs.")
print(score)  # Expected: high score for specific constraints
```

### Functionality
Constraint Detection: Finds numeric/range expressions and requirement keywords, normalized so that two or more constraints yield full score. Detail Indicators: Detects words signaling specificity. Examples: Identifies example phrases. Combines into weighted composite with thresholds for full credit.

---

## 5. Completeness & Component Coverage
**Complexity Level:** Moderate

### Function Code
```python
def completeness_score(prompt):
    """
    Checks presence of essential prompt components: task definition,
    context, audience, constraints, format, tone. Returns normalized
    0-1 score where higher indicates more completeness.
    """
    components = {
        'task': ['write', 'create', 'generate', 'explain', 'analyze', 'describe', 'compare'],
        'context': ['about', 'regarding', 'concerning', 'on the topic of', 'related to'],
        'audience': ['for', 'audience', 'readers', 'users', 'students', 'professionals'],
        'constraints': ['limit', 'length', 'word', 'character', 'maximum', 'minimum'],
        'format': ['format', 'structure', 'bullet', 'list', 'section', 'heading'],
        'tone': ['formal', 'informal', 'professional', 'casual', 'academic', 'friendly']
    }
    
    text = prompt.lower()
    found_count = sum(any(keyword in text for keyword in keywords)
                      for keywords in components.values())
    total = len(components)
    
    # Normalize: each component equally weighted
    return round(found_count / total, 3)
```

### Example Implementation
```python
score = completeness_score("Write a formal academic analysis about climate change for students. Use bullet format with maximum 1000 words.")
print(score)  # Expected: high score for complete components
```

### Functionality
Defines six essential prompt components. Checks whether at least one keyword for each component appears. Computes fraction of components covered (0–1).

---

## 6. Structure & Organizational Score
**Complexity Level:** Moderate

### Function Code
```python
def structure_score(prompt):
    """
    Evaluates prompt organization by checking:
    - Transitional phrase usage
    - List formatting
    - Paragraph structure
    Returns normalized 0-1 score where higher indicates better structure.
    """
    import re

    text = prompt.lower()
    # Transitional phrases
    transitions = ['first', 'second', 'then', 'next', 'finally', 'also', 'additionally', 'furthermore']
    trans_count = sum(text.count(t) for t in transitions)
    trans_score = min(1.0, trans_count / 3)  # 3+ transitions => full score

    # List formatting
    has_bullets = bool(re.search(r'^[\-\*\•]\s+', prompt, re.MULTILINE))
    has_numbers = bool(re.search(r'^\d+\.\s+', prompt, re.MULTILINE))
    list_score = 1.0 if (has_bullets or has_numbers) else 0.0

    # Paragraph structure
    paragraphs = [p for p in prompt.split('\n\n') if p.strip()]
    para_score = min(1.0, len(paragraphs) / 2)  # 2+ paragraphs => full score

    # Weighted composite
    return round(0.4*trans_score + 0.3*list_score + 0.3*para_score, 3)
```

### Example Implementation
```python
score = structure_score("""First, analyze the data.

Then, create visualizations:
- Bar charts
- Line graphs

Finally, summarize findings.""")
print(score)  # Expected: high score for good structure
```

### Functionality
Transitional Phrases: Rewards presence of logical connectors. List Formatting: Checks for bullet or numbered lists. Paragraphing: Measures whether prompt is broken into multiple paragraphs. Combines into weighted composite score (40% transitions, 30% list usage, 30% paragraphs).

---

## 7. Complexity & Cognitive Load Score
**Complexity Level:** Moderate

### Function Code
```python
def complexity_score(prompt):
    """
    Estimates prompt complexity by combining:
    - Instruction count
    - Nested clause depth
    - Technical word density
    Returns normalized 0-1 score where lower indicates simpler prompts.
    """
    import re
    from nltk.tokenize import word_tokenize, sent_tokenize

    text = prompt.lower()
    # 1. Instruction count (sentences containing directive verbs)
    directives = ['write', 'create', 'explain', 'analyze', 'describe', 'compare', 'provide', 'include']
    sentences = sent_tokenize(prompt)
    instr_count = sum(any(d in s.lower() for d in directives) for s in sentences)
    instr_score = min(1.0, instr_count / 3)  # 3+ instructions => full complexity

    # 2. Nested clause depth (parentheses + subordinators)
    paren_depth = max(prompt[:i].count('(') - prompt[:i].count(')') for i in range(len(prompt)+1))
    subordinators = ['because','since','although','while','where','if']
    sub_count = sum(text.count(s) for s in subordinators)
    nesting = paren_depth + min(sub_count, 3)
    nesting_score = min(1.0, nesting / 3)  # 3+ nested clauses => full complexity

    # 3. Technical word density (words >8 chars)
    tokens = word_tokenize(text)
    tech_count = sum(1 for w in tokens if len(w) > 8)
    tech_score = min(1.0, tech_count / len(tokens)) if tokens else 0.0

    # Composite: higher means more complex → invert for simplicity
    comp = 0.4*instr_score + 0.3*nesting_score + 0.3*tech_score
    simplicity_score = 1.0 - comp
    return round(simplicity_score, 3)
```

### Example Implementation
```python
score = complexity_score("Write a comprehensive analysis (including methodology and implementation details) while considering interdisciplinary approaches.")
print(score)  # Expected: lower score due to complexity
```

### Functionality
Instruction Count: Detects directive sentences; too many instructions increases complexity. Nested Clauses: Measures parentheses depth plus subordinating conjunctions. Technical Density: Calculates proportion of long (8+ character) words. Combines into a composite "complexity" score, then inverts to yield a simplicity score (0–1, higher = simpler).

---

## 8. Clarity & Ambiguity Detection
**Complexity Level:** Moderate-High

### Function Code
```python
def clarity_score(prompt):
    """
    Evaluates prompt clarity by detecting ambiguous words,
    vague references, and unclear instruction patterns.
    Returns normalized 0-1 score where higher indicates clearer prompts.
    """
    import re
    from collections import Counter
    
    text = prompt.lower()
    words = text.split()
    
    # 1. Ambiguous word detection
    ambiguous_words = ['something', 'anything', 'stuff', 'things', 'it', 'this', 'that', 'some', 'many', 'several']
    ambig_count = sum(words.count(w) for w in ambiguous_words)
    ambig_penalty = min(1.0, ambig_count / 5)  # 5+ ambiguous words => full penalty
    
    # 2. Pronoun overuse (excessive "it", "they", "them")
    pronouns = ['it', 'they', 'them', 'these', 'those']
    pronoun_count = sum(words.count(p) for p in pronouns)
    pronoun_penalty = min(1.0, pronoun_count / len(words) * 10) if words else 0  # >10% pronouns => penalty
    
    # 3. Unclear instruction patterns
    unclear_patterns = [r'\bmaybe\b', r'\btry to\b', r'\bkind of\b', r'\bsort of\b']
    unclear_count = sum(len(re.findall(p, text)) for p in unclear_patterns)
    unclear_penalty = min(1.0, unclear_count / 3)  # 3+ unclear phrases => full penalty
    
    # Composite clarity (start at 1.0, subtract penalties)
    clarity = 1.0 - (0.4*ambig_penalty + 0.3*pronoun_penalty + 0.3*unclear_penalty)
    return round(max(0.0, clarity), 3)
```

### Example Implementation
```python
score = clarity_score("Write something about machine learning stuff that might be useful for things.")
print(score)  # Expected: low score due to ambiguous language
```

### Functionality
Ambiguous Words: Detects vague terms like "something," "stuff," "things." Pronoun Overuse: Penalizes excessive use of unclear pronouns (>10% of text). Unclear Instructions: Identifies hedging language like "maybe," "try to," "kind of." Starts with perfect clarity (1.0) and subtracts weighted penalties for each issue.

---

## 9. Perplexity-Based Quality Estimation
**Complexity Level:** High

### Function Code
```python
def perplexity_score(prompt, model, max_tokens=0):
    """
    Calculates prompt perplexity using a language model's log-likelihood.
    Returns a normalized score from 0-1 where higher indicates lower perplexity (better prompts).
    """
    import math
    
    # Encode and compute log-likelihood
    enc = model.tokenizer.encode(prompt)
    nll = model.compute_nll(enc)  # sum of negative log-likelihoods
    ppl = math.exp(nll / len(enc)) if len(enc) > 0 else float('inf')
    
    # Normalize: assume reasonable prompt perplexity between 10 and 100
    if ppl < 10:
        score = 1.0
    elif ppl > 100:
        score = 0.0
    else:
        score = 1.0 - ((ppl - 10) / 90)
    return round(score, 3)
```

### Example Implementation
```python
# Requires language model with tokenizer and compute_nll method
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
score = perplexity_score("Write a clear analysis of data trends.", model)
print(score)  # Expected: high score for low perplexity
```

### Functionality
Perplexity Calculation: Uses model's negative log-likelihood to compute perplexity. Normalization: Maps perplexity range [10–100] to score [1–0]. Lower perplexity (≤10) yields perfect score, very high perplexity (≥100) yields zero.

---

## 10. Semantic Similarity to Reference Prompts
**Complexity Level:** High

### Function Code
```python
def semantic_similarity_score(prompt, reference_prompts, model=None):
    """
    Computes maximum cosine similarity between the prompt and a set of reference prompts.
    Returns a normalized score 0-1 where higher indicates greater semantic closeness to exemplars.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load model if not provided
    stm = model or SentenceTransformer('all-MiniLM-L6-v2')

    # Encode prompt and references
    embeddings = stm.encode([prompt] + reference_prompts)
    prompt_emb, ref_embs = embeddings[0], embeddings[1:]
    
    # Compute cosine similarities
    sims = np.dot(ref_embs, prompt_emb) / (np.linalg.norm(ref_embs, axis=1) * np.linalg.norm(prompt_emb))
    max_sim = float(np.max(sims))

    # Normalize (assuming similarity in [0.5, 0.9] is desirable)
    if max_sim >= 0.9:
        score = 1.0
    elif max_sim <= 0.5:
        score = 0.0
    else:
        score = (max_sim - 0.5) / 0.4  # linearly map [0.5–0.9]→[0–1]

    return round(score, 3)
```

### Example Implementation
```python
references = ["Write a detailed analysis", "Create a comprehensive report"]
score = semantic_similarity_score("Generate an in-depth examination", references)
print(score)  # Expected: high score for semantic similarity
```

### Functionality
Embeds prompt and multiple reference exemplars using a sentence transformer. Computes cosine similarities and selects the maximum. Normalizes similarity from 0.5–0.9 range to a 0–1 score, with values outside clipped.