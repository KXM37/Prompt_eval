# Prompt Evolution System - Integration Guide

## 📁 Project Structure

```
prompt_evolution/
├── prompt_evolution_core.py    # Core engine (Artifact 1)
├── streamlit_app.py            # Streamlit UI (Artifact 2)
├── requirements.txt            # Dependencies
├── evolution_data/             # Auto-created for results
│   └── [run_name]/
│       ├── results.json
│       └── final_prompt.txt
└── README.md
```

## 🔧 Installation

### 1. Install Dependencies

```bash
pip install streamlit pandas plotly numpy dspy-ai textgrad
```

**requirements.txt:**
```
streamlit>=1.31.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
dspy-ai>=2.0.0
textgrad>=0.1.0
```

### 2. Integrate Your Internal API

Open `prompt_evolution_core.py` and replace the `TemplateAPIClient` class:

```python
class YourAPIClient:
    """Your internal LLM API integration"""
    
    def __init__(self, api_key, model_name="your-model"):
        # Initialize your API client
        self.api_key = api_key
        self.model_name = model_name
        # self.client = YourAPILibrary(api_key=api_key)
    
    def evaluate(self, evaluation_prompt: str) -> str:
        """Call your LLM for evaluation"""
        response = self.client.generate(
            prompt=evaluation_prompt,
            model=self.model_name,
            temperature=0.3,  # Lower temp for consistent scoring
            max_tokens=2000
        )
        return response.text
    
    def optimize(self, optimization_prompt: str) -> str:
        """Call your LLM for prompt rewriting"""
        response = self.client.generate(
            prompt=optimization_prompt,
            model=self.model_name,
            temperature=0.7,  # Higher temp for creative rewriting
            max_tokens=4000
        )
        return response.text
```

Then update the Streamlit app to use your client:

```python
# In streamlit_app.py, around line 300
api_client = YourAPIClient(
    api_key=st.secrets["api_key"],  # Use Streamlit secrets
    model_name="your-model-name"
)
```

## 🚀 Running the System

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

### Option 2: Programmatic Usage

```python
from prompt_evolution_core import PromptEvolutionEngine, YourAPIClient

# Initialize
api_client = YourAPIClient(api_key="your-key")
engine = PromptEvolutionEngine(
    api_client=api_client,
    top_n_worst=5,
    max_iterations=6,
    improvement_threshold=0.001
)

# Run evolution
results = engine.evolve(
    initial_prompt="Your prompt here...",
    run_name="my_optimization_run"
)

# Access results
print(f"Final Score: {results['summary']['final_score']}")
print(f"Improvement: {results['summary']['total_improvement_pct']:.2f}%")
print(f"Best Strategy: {results['summary']['best_strategy']}")
```

## 🎯 Key Features

### 1. **Metric Grouping**
35 metrics organized into 7 thematic groups:
- Structure & Clarity (5 metrics)
- Context & Information (5 metrics)
- Reasoning & Cognition (6 metrics)
- Safety & Alignment (6 metrics)
- Format & Style (5 metrics)
- Output Quality (6 metrics)
- Advanced Features (2 metrics)

### 2. **Optimization Strategies**
- **Aggressive**: Targets worst-performing metrics with bold changes
- **Balanced**: Holistic improvement while preserving strengths
- **Structural**: Focuses on formatting and organization

### 3. **Weighted Evaluation**
- Quadratic penalty on top-N worst metrics
- Formula: `weight = (1 - score/300)²` for worst metrics
- Ensures focus on biggest weaknesses

### 4. **Stopping Criteria**
- Max iterations (default: 6)
- Improvement threshold (default: 0.10%)
- Either condition triggers stop

## 📊 Output Format

### Results JSON Structure:
```json
{
  "run_name": "run_20241003_143022",
  "initial_prompt": "...",
  "final_prompt": "...",
  "total_iterations": 6,
  "summary": {
    "initial_score": 145.2,
    "final_score": 267.8,
    "total_improvement_pct": 84.5,
    "iterations_completed": 6,
    "best_strategy": "balanced"
  },
  "evolution_history": [...],
  "metric_trajectories": {...}
}
```

## 🔮 Future Enhancements (V2 Features)

### Template Support
```python
# Add to PromptEvolutionEngine
def evolve_template(self, template: str, variables: List[str]):
    """Optimize prompt templates with placeholders"""
    # Preserve {{variable}} syntax
    # Optimize around variables
    pass
```

### Constraint System
```python
@dataclass
class PromptConstraints:
    max_length: int = None
    required_keywords: List[str] = None
    forbidden_words: List[str] = None
    tone: str = None  # 'formal', 'casual', 'technical'
    preserve_sections: List[str] = None
```

### DSPy Integration (Advanced)
```python
import dspy

class PromptOptimizerSignature(dspy.Signature):
    """Optimize prompt based on metric feedback"""
    current_prompt = dspy.InputField()
    weak_metrics = dspy.InputField()
    metric_scores = dspy.InputField()
    optimized_prompt = dspy.OutputField()

# Use DSPy's optimization
optimizer = dspy.ChainOfThought(PromptOptimizerSignature)
```

## 🐛 Troubleshooting

### Issue: API Rate Limits
**Solution:** Add rate limiting in API client:
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=10):