# Pre-LLM Prompt Evaluation Methods

## 1. Prompt Length Analysis

### Implementation
```python
def prompt_length_score(prompt, tokenizer=None):
    import re
    char_count, word_count = len(prompt), len(prompt.split())
    sentences = len([s.strip() for s in re.split(r'[.!?]+', prompt.strip()) if s.strip()])
    token_count = len(tokenizer.encode(prompt)) if tokenizer else int(word_count * 1.3)
    paren_depth = max(prompt[:i].count('(') - prompt[:i].count(')') for i in range(len(prompt)+1))
    subordinating_words = ['that','which','who','because','since','if','although','when','where','while']
    subordinating = sum(1 for w in subordinating_words if w in prompt.lower())
    nested_depth = paren_depth + min(subordinating, 3) + min(prompt.count(',')//3, 2)
    token_score = 1.0 if 50<=token_count<=400 else max(0.1, min(token_count/50, 1.0-(token_count-400)/600))
    word_score = 1.0 if 10<=word_count<=100 else max(0.2, min(word_count/10, 1.0-(word_count-100)/200))
    sentence_score = 1.0 if 1<=sentences<=8 else max(0.3, 1.0-(sentences-8)/12) if sentences>8 else 0.0
    nesting_score = 1.0 if nested_depth<=2 else max(0.1, 1.0-(nested_depth-2)/3)
    return round(0.4*token_score + 0.25*word_score + 0.2*nesting_score + 0.15*sentence_score, 3)
```

### Functionality
Returns a normalized composite score (0-1) based on token count, word count, sentence count, and nesting depth, with research-backed thresholds. Higher score indicates prompt is within optimal structural complexity.

---

## 2. Keyword and Pattern Matching Score

### Implementation
```python
def keyword_pattern_score(prompt, keywords=None, patterns=None):
    import re
    professional_bank = ["context:", "example:", "instruction:", "role:", "task:", "guidelines:", "requirements:", "steps:", "output:", "format:", "constraints:", "data:", "reference:", "goal:", "criteria:", "limit:", "objective:", "parameters:", "details:", "background:", "summarize", "outline", "analyze", "compare", "contrast", "evaluate", "list", "provide", "optimize", "improve", "paraphrase", "prioritize", "highlight", "organize", "categorize"]
    default_patterns = [
        r'\bplease\b', r'\binstruction[s]?:', r'\bsteps?:', r'\bexample[s]?:', r'\bcontext:',
        r'\boutput:', r'\bformat:', r'\brequirements?:', r'\bdetails?:', r'\bgoal[s]?:', r'\bcriteria:', r'\bbackground:',
        r'\banalyz\w*\b', r'\bexplain\b', r'\bdescribe\b', r'\bprovide\b', r'\bprioritize\b', r'\boptimize\b', r'\borganize\b', r'\?\s*$', r'\b\d+\)'
    ]
    prompt_low = prompt.lower()
    required_keywords = [kw.lower() for kw in (keywords or [])]
    word_bank = [w.lower() for w in professional_bank]
    found_required = [kw for kw in required_keywords if kw in prompt_low]
    custom_coverage = len(found_required) / len(required_keywords) if required_keywords else 1.0
    found_professional = [w for w in word_bank if w in prompt_low]
    professional_coverage = min(1.0, len(found_professional) / 3)
    keyword_score = round(0.6 * custom_coverage + 0.4 * professional_coverage, 3)
    regex_list = patterns or default_patterns
    matches = sum(bool(re.search(p, prompt, re.IGNORECASE)) for p in regex_list)
    pattern_score = round(matches / len(regex_list), 3)
    return {'keyword_score': keyword_score, 'pattern_score': pattern_score}
```

### Functionality
Returns two separate scores (0-1): keyword_score evaluates task-specific and professional vocabulary coverage, pattern_score evaluates presence of structural/formatting cues using regex patterns. Supports both content and structure optimization.

---

## 3. Readability Metrics (Flesch-Kincaid) Score

### Implementation
```python
def readability_score(prompt):
    import textstat
    reading_ease = textstat.flesch_reading_ease(prompt)
    grade_level = textstat.flesch_kincaid_grade(prompt)
    if reading_ease >= 60 and reading_ease <= 80:
        ease_score = 1.0
    elif reading_ease < 60:
        ease_score = max(0.0, reading_ease / 60)
    else:
        ease_score = max(0.0, 1.0 - (reading_ease - 80) / 40)
    if grade_level >= 8 and grade_level <= 12:
        grade_score = 1.0
    elif grade_level < 8:
        grade_score = max(0.0, grade_level / 8)
    else:
        grade_score = max(0.0, 1.0 - (grade_level - 12) / 8)
    return round(0.5 * ease_score + 0.5 * grade_score, 3)
```

### Functionality
Returns a 0–1 score based on Flesch Reading Ease and Grade Level normalized to optimal readability windows. Rewards prompts that are neither too complex nor too simple.

---

## 4. Syntactic Complexity Analysis

### Implementation
```python
def syntactic_complexity_score(prompt):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return _fallback_complexity_score(prompt)
    doc = nlp(prompt)
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
    max_depth = 0; total_depth = 0; token_count = 0
    for token in doc:
        if not token.is_space and not token.is_punct:
            depth = len(list(token.ancestors))
            max_depth = max(max_depth, depth)
            total_depth += depth
            token_count += 1
    avg_depth = total_depth / token_count if token_count > 0 else 0
    subordinate_clauses = sum(1 for token in doc if token.dep_ in ['advcl', 'ccomp', 'xcomp'])
    relative_clauses = sum(1 for token in doc if token.dep_ == 'relcl')
    length_score = 1.0 if 8 <= avg_sentence_length <= 20 else max(0.1, min(avg_sentence_length/8, 1.0-(avg_sentence_length-20)/15))
    depth_score = 1.0 if max_depth <= 4 else max(0.1, 1.0-(max_depth-4)/6)
    clause_score = 1.0 if subordinate_clauses <= 2 else max(0.2, 1.0-(subordinate_clauses-2)/4)
    composite = round(0.4*length_score + 0.35*depth_score + 0.25*clause_score, 3)
    return composite
```

### Functionality
Returns a 0–1 score; higher = simpler syntax. Analyzes sentence length, dependency depth, and clause/nesting frequency using spaCy parsing. Penalizes excessive syntactic complexity.

---