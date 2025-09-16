# Pre-LLM Prompt Evaluation Methods

## 1. Prompt Length Analysis

### Implementation
```python
def prompt_length_score(prompt```okenizer=None```    import```
    char_count```ord_count =```n(prompt),```n(prompt.split```
    sentences```len([s.strip() for s in re.split(r'[.!?]+```prompt.strip``` if s.strip```)
``` token_count```len(tokenizer```code(prompt)) if```kenizer else```t(word_count```1.3)
    par```depth = max```ompt[:i].```nt('(') - prompt```].```nt(')') for```in range(len```ompt)+1))
   ```bordinating```rds = ['that','which','who','because','since','if','although','when','where','while']
``` subordinating```sum(1 for w``` subordinating```rds if w in```ompt.lower```
    nested```pth = paren```pth + min(sub```inating, 3``` min(prompt```unt(',')//3, ```    token_score```1.0 if 50<=```en_count<=400 else```x(0.1, min```ken_count/50, ```-(token_count-400)/600```    word_score```1.0 if 10<=```d_count<=100 else```x(0.2, min```rd_count/10, ```-(word_count-100)/200))
``` sentence_score```1.0 if 1<=```tences<=8 else```x(0.3, 1.0```entences-8)/12) if```ntences>8 else```0
    nesting```ore = 1.0 if```sted_depth``` else max(```, 1.0-(nested```pth-2)/3)
   ```turn round```4*token_score +```25*word_score```0.2*nesting```ore + 0.15```ntence_score, ``````

### Functionality
Returns a normalized composite score (0-1) based on token count, word count, sentence count, and nesting depth, with research-backed thresholds. Higher score indicates prompt is within optimal structural complexity.

***

## 2. Keyword and Pattern Matching Score

### Implementation
```python
def keyword_pattern_score(prompt```eywords=None```atterns=None```    import```
    professional```nk = ["context:", "example:", "instruction:", "role:", "task:", "guidelines:", "requirements:", "steps:", "output:", "format:", "constraints:", "data:", "reference:", "goal:", "criteria:", "limit:", "objective:", "parameters:", "details:", "background:", "summarize", "outline", "analyze", "compare", "contrast", "evaluate", "list", "provide", "optimize", "improve", "paraphrase", "prioritize", "highlight", "organize", "categorize"]
``` default_patterns```[
        r'\bplease\b', r'\binstruction[s]?:', r'\bsteps?:', r'\bexample[s]?:', r'\bcontext:',
        r'\boutput:', r'\bformat:', r'\brequirements?:', r'\bdetails?:', r'\bgoal[s]?:', r'\bcriteria:', r'\bbackground:',
        r'\banalyz\w*\b', r'\bexplain\b', r'\bdescribe\b', r'\bprovide\b', r'\bprioritize\b', r'\boptimize\b', r'\borganize\b', r'\?\s*$', r'\b\d+\)'
    ]
    prompt_low = prompt.lower```    required```ywords = [kw.lower() for kw in (keywords or [])````` word_bank``` [w.lower() for w in professional_bank]
``` found_required``` [kw for kw in required_keywords if kw in prompt_low]
``` custom_coverage```len(found_required``` len(required```ywords) if```quired_keywords```se 1.0
   ```und_professional``` [w for w in word_bank if w in prompt_low]
``` professional```verage = min```0, len(found```ofessional) /```
    keyword```ore = round```6 * custom```verage + 0```* professional```verage, 3)
``` regex_list```patterns or```fault_patterns```  matches =```m(bool(re.search``` prompt, re```NORECASE)) for```in regex_list```   pattern```ore = round```tches / len```gex_list), ```    return```keyword_score```keyword_score```pattern_score```pattern_score`````

### Functionality
Returns two separate scores (0-1): keyword_score evaluates task-specific and professional vocabulary coverage, pattern_score evaluates presence of structural/formatting cues using regex patterns.

***

## 3. Readability Metrics (Flesch-Kincaid) Score

### Implementation
```python
def readability_score(prompt```    import```xtstat
   ```ading_ease```textstat.f```ch_reading_ease(prompt```   grade_level```textstat.f```ch_kincaid_grade(prompt```   if reading```se >= 60 and```ading_ease``` 80:
       ```se_score =```0
    elif```ading_ease```60:
       ```se_score =```x(0.0, reading```se / 60)
   ```se:
       ```se_score =```x(0.0, 1.0```(reading_e``` - 80) / 40```   if grade```vel >= 8 and```ade_level <=```:
        grade```ore = 1.0
``` elif grade```vel < 8:
       ```ade_score =```x(0.0, grade```vel / 8)
   ```se:
       ```ade_score =```x(0.0, 1.0```(grade_level```12) / 8)
   ```turn round```5 * ease_score```0.5 * grade```ore, 3)
```

### Functionality
Returns a 0–1 score based on Flesch Reading Ease and Grade Level normalized to optimal readability windows.

***

## 4. Syntactic Complexity Analysis

### Implementation
```python
def syntactic_complexity_score```ompt):
   ```port spacy```  try:
       ```p = spacy.load```n_core_web_sm")
   ```cept OSError```       return```allback_complex```_score(prompt)
   ```c = nlp(prompt```   sentences```list(doc.s```s)
    if not```ntences:
       ```turn 0.0
   ```g_sentence```ngth = sum```n(sent) for```nt in sentences``` len(sentences```   max_depth```0; total_depth```0; token_count```0
    for token``` doc:
       ``` not token```_space and```t token.is```nct:
           ```pth = len(list```ken.ancestors))
           ```x_depth = max```x_depth, depth```          ```tal_depth +=```pth
           ```ken_count +=```    avg_depth```total_depth```token_count``` token_count```0 else 0
   ```bordinate_cl```es = sum(1```r token in```c if token```p_ in ['advcl', 'ccomp', 'xcomp'])
``` relative_cl```es = sum(1```r token in```c if token```p_ == 'rel```)
    length```ore = 1.0 if```<= avg_sentence```ngth <= 20```se max(0.1```in(avg_sentence```ngth/8, 1.```avg_sentence_length-20```5))
    depth```ore = 1.0 if```x_depth <=```else max(0``` 1.0-(max_depth```/6)
    clause```ore = 1.0 if```bordinate_cl```es <= 2 else```x(0.2, 1.0```ubordinate_clauses-2)/```    composite```round(0.4*```gth_score +```35*depth_score```0.25*clause```ore, 3)
   ```turn composite````

### Functionality
Returns a 0–1 score; higher = simpler syntax, penalizing excessive syntactic complexity.

***

## 5. N-gram Based Metrics (BLEU/ROUGE on Prompts)

### Implementation
```python
def ngram_similarity_score(prompt```prompt2, n```:
    from```tk import n```ms
    import```tk
    try```       nltk```ta.find('tokenizers/punkt```    except```okupError:
```     nltk.download```unkt')
   ```kens1 = nltk```rd_tokenize(prompt1.lower```
    tokens``` nltk.word```kenize(prompt2.lower())
``` grams1 = set```rams(tokens1, n```    grams2```set(ngrams```kens2, n))
``` intersection```len(grams1```grams2)
   ```ion = len(```ms1 | grams```    similarity```intersection```union if union```0 else 0.0```  return round```milarity, ``````

### Functionality
Computes n-gram overlap between two prompts using Jaccard similarity. Higher score = greater textual similarity.

***

## 6. Semantic Similarity Metrics

### Implementation
```python
def semantic_similarity_score```ompt1, prompt```model_name```ll-MiniLM-L6-v2"):
   ```om sentence```ansformers import```ntenceTransformer```  from sklearn```trics.pairwise import```sine_similarity```  model = Sentence```nsformer(model_name)
   ```beddings =```del.encode```rompt1, prompt2])
``` sim = cosine```milarity([embeddings[0]],```mbeddings[1]])```[0]
``` return round```oat(sim), ``````

### Functionality
Encodes prompts into dense vectors and computes cosine similarity. Higher score = stronger semantic similarity.

***

## 7. Cross-Entropy Score

### Implementation
```python
def cross_entropy_score(prompt```odel, tokenizer```    import```rch
    inputs```tokenizer(prompt```eturn_tensors```t")
    input```s = inputs```nput_ids"]
``` with torch```_grad():
       ```tputs = model```inputs)
   ```gits = outputs```gits
    shift```gits = logits```., :-1, :].```tiguous()
   ```ift_labels```input_ids[..., 1:].```tiguous()
   ```ss = torch```.functional.cross_entropy```       shift```gits.view(-1, shift```gits.size(-1)),
       ```ift_labels```ew(-1),
       ```duction="mean```   )
    return```und(loss.item``` 3)
```

### Functionality
Averages cross-entropy over all tokens in the prompt. Lower values indicate prompts more “natural” for the model.

***

## 8. Coherence Metrics

### Implementation
```python
def coherence_score(prompt):
``` import re```  markers =``` [
        "however", "therefore", "furthermore", "moreover",
        "consequently", "meanwhile", "nevertheless", "nonetheless",
        "similarly", "additionally", "conversely", "instead"
    ]
``` words = re```ndall(r"\b\w+\b", prompt.lower())
    total_words```len(words)``` 1
    marker```unt = sum(words```unt(marker) for```rker in markers```   density```min(1.0, marker```unt / (total```rds * 0.05```    return```und(density```)
```

### Functionality
Measures density of discourse markers for logical flow. Approximates optimal at ~5% marker density.

***

## 9. Entropy-Based Uncertainty Score

### Implementation
```python
def entropy_uncertainty_score```edictions_list):
   ```port numpy``` np
    ent```ies = []
``` for probs``` predictions```st:
       ```obs = np.array```obs)
       ```obs = probs```probs.sum()
```     entropy```-np.sum(pro```* np.log2(pro```+ 1e-12))
```     entrop```.append(entropy)
   ```an_entropy```float(np.mean```tropies))
   ```riance_entropy```float(np.var```tropies))
   ```turn {
       ```ean_entropy```round(mean```tropy, 3),
```     'entropy```riance': round```riance_entropy, ```    }
```

### Functionality
Returns mean and variance of model output entropy per token. High mean = model uncertainty; high variance = instability.

***

## 10. Perplexity-Based Evaluation

### Implementation
```python
def perplexity_score(prompt,```del, tokenizer```    import```rch
    inputs```tokenizer(prompt```eturn_tensors```t")
    input```s = inputs```nput_ids"]
``` with torch```_grad():
       ```tputs = model```inputs)
   ```gits = outputs```gits
    shift```gits = logits```., :-1, :].```tiguous()
   ```ift_labels```input_ids[..., 1:].```tiguous()
   ```ss = torch```.functional.cross_entropy```       shift```gits.view(-1, shift```gits.size(-1)),
       ```ift_labels```ew(-1),
       ```duction="mean```   )
    perp```torch.exp(loss```tem()
    return```und(perp, ``````

### Functionality
Returns perplexity for the prompt. Lower is more predictable/natural to the model.

***

## 11. SPELL (Selecting Prompts by Estimating LM Likelihood)

### Implementation
```python
def spell_prompt_selection(prom```, inputs, model```okenizer):
``` import torch```  import math```  def _per```xity(text):
       ```puts_ids =```kenizer(text```eturn_tensors```t")["input_ids"]
```     with torch```_grad():
           ```tputs = model```put_ids=inputs_ids)
       ```gits = outputs```gits
       ```ift_logits```logits[..., :-1, :].```tiguous()
       ```ift_labels```inputs_ids```., 1:].```tiguous()
       ```ss = torch```.functional.cross_entropy```          ```ift_logits```ew(-1, shift```gits.size(-1)),
           ```ift_labels```ew(-1),
           ```duction="mean```       )
       ```turn math.exp```ss.item())
   ```g_perps = []
``` for prompt``` prompts:
```     perps```[]
```     for inp``` inputs:
           ```mbined = f```rompt} {inp```           ```rps.append```erplexity(combined))
       ```g_perps.append```m(perps) /```n(perps))
``` best_idx =```t(min(range```n(prompts)), key```mbda i: avg```rps[i]))
``` return {
```     'best```ompt': prompts```st_idx],
```     'average```rplexities': [round(p, 3) for p in avg_perps],
```     'best```dex': best```x
    }
```

### Functionality
Ranks prompts by average perplexity over multiple inputs and selects the lowest as most natural/effective.
