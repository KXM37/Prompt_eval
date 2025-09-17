"""
Pre-LLM Evaluation Methods - Python Implementation
A comprehensive collection of automated prompt evaluation methods that assess prompt quality before LLM execution.
These methods are ordered from simplest to most complex in terms of implementation and explainability.
"""

import re
import math
import statistics
from typing import Dict, List, Optional, Union
from collections import Counter
import numpy as np

# Required imports (install with pip if needed)
try:
    import textstat
except ImportError:
    print("Warning: textstat not installed. Run: pip install textstat")
    textstat = None

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    print("Warning: nltk not installed. Run: pip install nltk")
    word_tokenize = sent_tokenize = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")
    SentenceTransformer = None


class PreLLMEvaluator:
    """
    Collection of pre-LLM evaluation methods for prompt quality assessment.
    All methods return normalized scores from 0-1 where higher is better.
    """
    
    def __init__(self):
        """Initialize the evaluator with optional models."""
        self.similarity_model = None
        if SentenceTransformer:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                print("Warning: Could not load sentence transformer model")

    # Method 1: Prompt Length Analysis (Very Simple)
    def prompt_length_score(self, prompt: str, tokenizer=None) -> float:
        """
        Comprehensive prompt length analysis with composite scoring.
        Returns a normalized score from 0-1 where higher is better.
        """
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

    # Method 2: Readability Analysis (Simple)
    def readability_score(self, prompt: str) -> float:
        """
        Analyzes prompt readability using multiple established metrics.
        Returns a normalized score from 0-1 where higher is better.
        """
        if not textstat:
            return 0.5  # Default score if textstat unavailable
        
        # Basic text statistics
        words = len(prompt.split())
        sentences = len([s.strip() for s in re.split(r'[.!?]+', prompt.strip()) if s.strip()])
        
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

    # Method 3: Lexical Diversity Score (Simple-Moderate)
    def lexical_diversity_score(self, prompt: str) -> float:
        """
        Calculates lexical diversity (type-token ratio) and returns
        a normalized 0–1 score where higher indicates richer vocabulary.
        """
        if word_tokenize:
            tokens = word_tokenize(prompt.lower())
        else:
            tokens = prompt.lower().split()
        
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

    # Method 4: Specificity & Constraint Presence (Moderate)
    def specificity_score(self, prompt: str) -> float:
        """
        Evaluates prompt specificity by detecting detail indicators,
        numeric constraints, and example language. Returns normalized
        score 0-1 where higher indicates greater specificity.
        """
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

    # Method 5: Completeness & Component Coverage (Moderate)
    def completeness_score(self, prompt: str) -> float:
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

    # Method 6: Structure & Organizational Score (Moderate)
    def structure_score(self, prompt: str) -> float:
        """
        Evaluates prompt organization by checking:
        - Transitional phrase usage
        - List formatting
        - Paragraph structure
        Returns normalized 0-1 score where higher indicates better structure.
        """
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

    # Method 7: Complexity & Cognitive Load Score (Moderate)
    def complexity_score(self, prompt: str) -> float:
        """
        Estimates prompt complexity by combining:
        - Instruction count
        - Nested clause depth
        - Technical word density
        Returns normalized 0-1 score where higher indicates simpler prompts.
        """
        text = prompt.lower()
        
        # 1. Instruction count (sentences containing directive verbs)
        directives = ['write', 'create', 'explain', 'analyze', 'describe', 'compare', 'provide', 'include']
        if sent_tokenize:
            sentences = sent_tokenize(prompt)
        else:
            sentences = re.split(r'[.!?]+', prompt)
        instr_count = sum(any(d in s.lower() for d in directives) for s in sentences)
        instr_score = min(1.0, instr_count / 3)  # 3+ instructions => full complexity

        # 2. Nested clause depth (parentheses + subordinators)
        paren_depth = max(prompt[:i].count('(') - prompt[:i].count(')') for i in range(len(prompt)+1))
        subordinators = ['because','since','although','while','where','if']
        sub_count = sum(text.count(s) for s in subordinators)
        nesting = paren_depth + min(sub_count, 3)
        nesting_score = min(1.0, nesting / 3)  # 3+ nested clauses => full complexity

        # 3. Technical word density (words >8 chars)
        if word_tokenize:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        tech_count = sum(1 for w in tokens if len(w) > 8)
        tech_score = min(1.0, tech_count / len(tokens)) if tokens else 0.0

        # Composite: higher means more complex → invert for simplicity
        comp = 0.4*instr_score + 0.3*nesting_score + 0.3*tech_score
        simplicity_score = 1.0 - comp
        return round(simplicity_score, 3)

    # Method 8: Clarity & Ambiguity Detection (Moderate-High)
    def clarity_score(self, prompt: str) -> float:
        """
        Evaluates prompt clarity by detecting ambiguous words,
        vague references, and unclear instruction patterns.
        Returns normalized 0-1 score where higher indicates clearer prompts.
        """
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

    # Method 9: Perplexity-Based Quality Estimation (High)
    def perplexity_score(self, prompt: str, model=None) -> float:
        """
        Calculates prompt perplexity using a language model's log-likelihood.
        Returns a normalized score from 0-1 where higher indicates lower perplexity (better prompts).
        Note: Requires a language model with tokenizer and compute_nll method.
        """
        if not model:
            return 0.5  # Default score if no model provided
        
        try:
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
        except Exception:
            return 0.5

    # Method 10: Semantic Similarity to Reference Prompts (High)
    def semantic_similarity_score(self, prompt: str, reference_prompts: List[str], model=None) -> float:
        """
        Computes maximum cosine similarity between the prompt and a set of reference prompts.
        Returns a normalized score 0-1 where higher indicates greater semantic closeness to exemplars.
        """
        if not reference_prompts:
            return 0.5
        
        # Use provided model or class instance model
        stm = model or self.similarity_model
        if not stm:
            return 0.5  # Default score if no model available
        
        try:
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
        except Exception:
            return 0.5

    # Composite evaluation method
    def evaluate_all(self, prompt: str, reference_prompts: List[str] = None, 
                     tokenizer=None, language_model=None) -> Dict[str, float]:
        """
        Run all evaluation methods and return a comprehensive score dictionary.
        
        Args:
            prompt: The prompt to evaluate
            reference_prompts: List of reference prompts for similarity scoring
            tokenizer: Optional tokenizer for length analysis
            language_model: Optional language model for perplexity scoring
            
        Returns:
            Dictionary with all individual scores and a weighted composite score
        """
        scores = {
            'length': self.prompt_length_score(prompt, tokenizer),
            'readability': self.readability_score(prompt),
            'lexical_diversity': self.lexical_diversity_score(prompt),
            'specificity': self.specificity_score(prompt),
            'completeness': self.completeness_score(prompt),
            'structure': self.structure_score(prompt),
            'simplicity': self.complexity_score(prompt),  # Note: inverted complexity
            'clarity': self.clarity_score(prompt),
            'perplexity': self.perplexity_score(prompt, language_model),
            'semantic_similarity': self.semantic_similarity_score(prompt, reference_prompts or [])
        }
        
        # Weighted composite score (equal weights, but can be customized)
        weights = {k: 0.1 for k in scores.keys()}  # Equal weights summing to 1.0
        composite = sum(scores[k] * weights[k] for k in scores.keys())
        scores['composite'] = round(composite, 3)
        
        return scores


# Example usage and testing functions
def demo_evaluation():
    """Demonstrate the evaluation methods with sample prompts."""
    evaluator = PreLLMEvaluator()
    
    # Test prompts
    prompts = [
        "Write something about stuff.",  # Poor quality
        "Write a detailed 500-word analysis of climate change impacts on agriculture for students.",  # Good quality
        "Create a comprehensive report analyzing the economic implications of renewable energy adoption. Include specific examples, data visualization requirements, and format as a professional document with bullet points and sections."  # High quality
    ]
    
    print("Pre-LLM Evaluation Demo")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print("-" * 30)
        
        # Individual method scores
        scores = evaluator.evaluate_all(prompt)
        
        for method, score in scores.items():
            if method != 'composite':
                print(f"{method.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"COMPOSITE SCORE: {scores['composite']:.3f}")


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_evaluation()