<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# here is the current streamlit app:

Based on my analysis of your current Streamlit app, I can see you have a well-structured interface but it's missing the actual evaluation functions from your original `pre-llm-evaluation-methods.md` file. Here's my assessment and recommendations for improving your current app:

# Analysis of Your Current Streamlit App

## **Current Structure Assessment**

Your app has a **monolithic structure** with these key characteristics:

- **382 lines** in a single file
- **Split-button interface** for "Optimize" and "Evaluate" workflows
- **Good UI components**: Score display boxes, collapsible explanations, session state management
- **Missing core functionality**: The actual evaluation functions are not implemented


## **Key Issues Identified**

1. **❌ Missing Evaluation Functions**: Your pre-LLM evaluation methods aren't integrated
2. **❌ Hardcoded Scores**: Currently using placeholder/mock scores
3. **⚠️ Monolithic Design**: All code in one 382-line file makes maintenance difficult
4. **⚠️ Limited Extensibility**: Hard to add new evaluation metrics

## **Recommended Refactoring Plan**

Here's how to improve your current app by integrating it with the backend architecture I suggested earlier:

### **Step 1: Create the Modular Structure**

```
your-app/
├── backend/
│   ├── __init__.py
│   ├── evaluation_functions.py    # Your pre-LLM functions
│   └── evaluation_manager.py      # Business logic
├── frontend/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── ui_components.py       # Your existing UI helpers
│   │   ├── input_section.py       # Input components
│   │   └── results_display.py     # Results visualization
│   └── pages/
│       ├── __init__.py
│       └── main_page.py          # Your current main logic
├── utils/
│   ├── __init__.py
│   └── session_management.py     # Session state helpers
├── prompt_opteval_refactored.py  # New main entry point
└── requirements.txt
```


### **Step 2: Extract Your UI Components**

**`frontend/components/ui_components.py`**

```python
import streamlit as st

# Your existing color scheme and helper functions
COLORS = {
    'navy': '#2C3E50',
    'olive': '#7F8C8D', 
    'charcoal': '#34495E',
    'wine': '#8E44AD',
    'tan': '#D35400',
    'red': '#E74C3C',
    'yellow': '#F39C12',
    'green': '#27AE60'
}

def get_score_color(score, is_cumulative=False):
    """Your existing function"""
    if is_cumulative:
        if score <= 4:
            return COLORS['red']
        elif score <= 7:
            return COLORS['yellow']
        else:
            return COLORS['green']
    else:
        if score <= 1.75:
            return COLORS['red']
        elif score <= 3.50:
            return COLORS['yellow']
        else:
            return COLORS['green']

def score_display_box(title, score, is_cumulative=False):
    """Your existing score display function"""
    color = get_score_color(score, is_cumulative)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
    ">
        <h4 style="color: {color}; margin: 0;">{title}</h4>
        <h2 style="color: {color}; margin: 0.25rem 0;">{score:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

def collapsible_explanation(title, content):
    """Your existing collapsible explanation function"""
    with st.expander(title):
        st.markdown(content)
```


### **Step 3: Integrate Real Evaluation Functions**

**`backend/evaluation_functions.py`**

```python
"""
Pre-LLM evaluation functions from your original file
"""
import re
import textstat
import spacy
from typing import Dict, List, Optional, Any

def prompt_length_score(prompt, tokenizer=None):
    """Your original prompt_length_score function"""
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

def keyword_pattern_score(prompt, keywords=None, patterns=None):
    """Your original keyword_pattern_score function"""
    # ... (copy from your md file)
    pass

def readability_score(prompt):
    """Your original readability_score function"""
    # ... (copy from your md file)
    pass

def syntactic_complexity_score(prompt):
    """Your original syntactic_complexity_score function"""
    # ... (copy from your md file)
    pass

# Convert scores to your current 5-point scale for UI consistency
def convert_to_5point_scale(score_0_to_1: float) -> float:
    """Convert 0-1 score to 5-point scale for UI display"""
    return round(score_0_to_1 * 5.0, 2)

def evaluate_prompt_comprehensive(prompt: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation returning scores compatible with your UI
    """
    # Get raw scores (0-1 scale)
    length_raw = prompt_length_score(prompt)
    readability_raw = readability_score(prompt)
    complexity_raw = syntactic_complexity_score(prompt)
    keyword_results = keyword_pattern_score(prompt)
    
    # Convert to 5-point scale for your UI
    results = {
        'length': convert_to_5point_scale(length_raw),
        'readability': convert_to_5point_scale(readability_raw),
        'complexity': convert_to_5point_scale(complexity_raw),
        'keywords': convert_to_5point_scale(
            0.7 * keyword_results['keyword_score'] + 0.3 * keyword_results['pattern_score']
        ),
        'raw_scores': {
            'length': length_raw,
            'readability': readability_raw, 
            'complexity': complexity_raw,
            'keywords': keyword_results
        }
    }
    
    # Calculate cumulative score (compatible with your current UI)
    results['cumulative'] = sum([
        results['length'],
        results['readability'],
        results['complexity'], 
        results['keywords']
    ])
    
    return results
```


### **Step 4: Update Your Main App**

**`prompt_opteval_refactored.py`**

```python
import streamlit as st
import time
import json
from datetime import datetime

# Import your refactored modules
from backend.evaluation_functions import evaluate_prompt_comprehensive
from frontend.components.ui_components import (
    score_display_box, 
    collapsible_explanation,
    COLORS
)

# Page configuration
st.set_page_config(
    page_title="Prompt OptEval",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state (your existing logic)
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'pre_scores' not in st.session_state:
    st.session_state.pre_scores = None
if 'post_scores' not in st.session_state:
    st.session_state.post_scores = None
if 'rewrite_before' not in st.session_state:
    st.session_state.rewrite_before = ""

# Main UI (your existing layout)
st.title("Prompt OptEval")
st.markdown("---")

# Input Section (your existing code)
col1, col2 = st.columns([1, 3])
with col1:
    prompt_name = st.text_input("Input prompt name:")
with col2:
    prompt_text = st.text_area("Input Prompt:", height=120)

# Optional existing output (your existing code)
st.subheader("Optional: Do you have an existing output to evaluate?")
has_existing_output = st.radio("", [
    "No, generate new output", 
    "Yes, I have an output to evaluate"
], horizontal=True)

existing_output = ""
if has_existing_output == "Yes, I have an output to evaluate":
    existing_output = st.text_area("Paste your existing output here:", height=150)

# Split buttons - Optimize and Evaluate (your existing layout)
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Optimize", type="primary"):
        if prompt_text and prompt_name:
            # Your existing optimization logic
            with st.spinner("Optimizing prompt..."):
                time.sleep(2)  # Simulate optimization
                st.session_state.optimization_complete = True
                st.session_state.rewrite_before = f"OPTIMIZED: {prompt_text}"
        else:
            st.error("Please enter both prompt name and text.")

with col2:
    if st.button("📊 Evaluate", type="secondary"):
        if prompt_text and prompt_name:
            with st.spinner("Evaluating prompt..."):
                # REPLACE MOCK DATA WITH REAL EVALUATION
                evaluation_results = evaluate_prompt_comprehensive(prompt_text)
                
                # Store in session state using your existing structure
                st.session_state.pre_scores = {
                    'Length': evaluation_results['length'],
                    'Readability': evaluation_results['readability'], 
                    'Complexity': evaluation_results['complexity'],
                    'Keywords': evaluation_results['keywords'],
                    'Cumulative': evaluation_results['cumulative']
                }
                st.session_state.evaluation_complete = True
        else:
            st.error("Please enter both prompt name and text.")

# Results display (your existing code with real data)
if st.session_state.evaluation_complete:
    st.markdown("---")
    st.subheader("📊 Evaluation Results")
    
    # Display scores using your existing UI components
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        score_display_box("Length", st.session_state.pre_scores['Length'])
    with col2:
        score_display_box("Readability", st.session_state.pre_scores['Readability'])
    with col3:
        score_display_box("Complexity", st.session_state.pre_scores['Complexity'])
    with col4:
        score_display_box("Keywords", st.session_state.pre_scores['Keywords'])
    with col5:
        score_display_box("Cumulative", st.session_state.pre_scores['Cumulative'], is_cumulative=True)

# Continue with your existing optimization display and save functionality...
```


### **Step 5: Add Dependencies**

**`requirements.txt`**

```txt
streamlit>=1.28.0
textstat>=0.7.3
spacy>=3.7.0
pandas>=2.0.0
plotly>=5.15.0
```


### **Step 6: Quick Migration Script**

To help you migrate quickly, here's a script to extract your current functions:

**`migrate_existing_app.py`**

```python
"""
Quick migration helper to extract components from your current app
"""
import os

def create_directory_structure():
    """Create the new directory structure"""
    directories = [
        'backend',
        'frontend/components', 
        'frontend/pages',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files
        with open(f"{directory}/__init__.py", "w") as f:
            f.write("# Auto-generated __init__.py\n")

def extract_ui_components():
    """Extract UI components from current app"""
    # Read your current file
    with open("prompt_opteval_split_buttons.py", "r") as f:
        content = f.read()
    
    # Extract the UI helper functions
    lines = content.split('\n')
    
    # Find COLORS definition and helper functions
    colors_start = next(i for i, line in enumerate(lines) if 'COLORS = {' in line)
    colors_end = next(i for i, line in enumerate(lines[colors_start:], colors_start) if '}' in line) + 1
    
    colors_section = '\n'.join(lines[colors_start:colors_end])
    
    # Extract get_score_color function
    func_start = next(i for i, line in enumerate(lines) if 'def get_score_color' in line)
    func_end = next(i for i, line in enumerate(lines[func_start:], func_start) if line and not line.startswith((' ', '\t', 'def', 'if', 'elif', 'else', 'return'))) + 1
    
    print("Extracted UI components. Manual integration required.")

if __name__ == "__main__":
    create_directory_structure()
    extract_ui_components()
    print("Migration structure created!")
```


## **Benefits of This Refactoring**

1. **✅ Real Evaluation**: Integrates your actual pre-LLM evaluation functions
2. **✅ Maintainable**: Separates UI from business logic
3. **✅ Extensible**: Easy to add new evaluation metrics
4. **✅ Preserves UI**: Keeps your existing beautiful interface design
5. **✅ Testable**: Individual components can be unit tested

## **Next Steps**

1. **Create the directory structure** using the migration script
2. **Move your evaluation functions** from the .md file into `backend/evaluation_functions.py`
3. **Extract UI components** from your current file
4. **Test the integration** with real evaluation data
5. **Add new features** like batch evaluation, export functionality, etc.

Would you like me to help you with any specific part of this refactoring process?
<span style="display:none">[^1]</span>

<div style="text-align: center">⁂</div>

[^1]: prompt_opteval_split_buttons.py

