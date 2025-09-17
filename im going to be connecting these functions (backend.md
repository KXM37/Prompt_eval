<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# im going to be connecting these functions (backend) to a streamlit ui (frontend), where do i start?

Based on my research and your goal to connect your pre-LLM evaluation functions (backend) to a Streamlit UI (frontend), here's a comprehensive guide on where to start:

# Connecting Pre-LLM Evaluation Functions to Streamlit UI

## **Recommended Project Structure**

Start with this organized folder structure that separates backend logic from frontend UI:[^1][^2]

```
prompt-evaluation-app/
├── .streamlit/
│   ├── config.toml          # App configuration
│   └── secrets.toml         # Environment variables/secrets
├── backend/
│   ├── __init__.py
│   ├── evaluation_functions.py  # Your core evaluation functions
│   └── evaluation_manager.py   # Business logic layer
├── frontend/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── input_components.py   # Input widgets
│   │   ├── results_display.py   # Results visualization
│   │   └── sidebar.py           # Sidebar components
│   └── pages/
│       ├── __init__.py
│       ├── single_evaluation.py # Single prompt evaluation
│       ├── batch_evaluation.py  # Multiple prompts
│       └── comparison.py        # Compare prompts
├── utils/
│   ├── __init__.py
│   ├── data_processing.py       # Data handling utilities
│   └── session_management.py   # Session state helpers
├── assets/
│   └── styles.css              # Custom CSS (optional)
├── tests/
│   ├── test_evaluation.py
│   └── test_ui.py
├── requirements.txt
├── README.md
└── streamlit_app.py            # Main entry point
```


## **Step 1: Create Your Backend Layer**

### `backend/evaluation_functions.py`

Move your existing functions here with minimal modifications:

```python
"""
Core evaluation functions for prompt quality assessment
"""
import re
import textstat
import spacy
from typing import Dict, Optional, List

def prompt_length_score(prompt: str, tokenizer=None) -> float:
    """Your existing prompt_length_score function"""
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

def keyword_pattern_score(prompt: str, keywords: Optional[List[str]] = None, patterns: Optional[List[str]] = None) -> Dict[str, float]:
    """Your existing keyword_pattern_score function"""
    # ... (your existing implementation)
    pass

def readability_score(prompt: str) -> float:
    """Your existing readability_score function"""
    # ... (your existing implementation) 
    pass

def syntactic_complexity_score(prompt: str) -> float:
    """Your existing syntactic_complexity_score function"""
    # ... (your existing implementation)
    pass
```


### `backend/evaluation_manager.py`

Create a business logic layer that orchestrates your evaluation functions:

```python
"""
Business logic for prompt evaluation
"""
from typing import Dict, List, Optional, Any
from .evaluation_functions import (
    prompt_length_score, 
    keyword_pattern_score, 
    readability_score, 
    syntactic_complexity_score
)

class PromptEvaluationManager:
    """Manages prompt evaluation workflows"""
    
    def __init__(self):
        self.evaluation_functions = {
            'length': prompt_length_score,
            'keywords': keyword_pattern_score,
            'readability': readability_score,
            'complexity': syntactic_complexity_score
        }
    
    def evaluate_single_prompt(self, prompt: str, selected_metrics: List[str], 
                              thresholds: Optional[Dict[str, float]] = None,
                              keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a single prompt with selected metrics"""
        results = {
            'prompt': prompt,
            'scores': {},
            'passed': {},
            'recommendations': []
        }
        
        default_thresholds = {
            'length': 0.7,
            'keywords': 0.6, 
            'readability': 0.7,
            'complexity': 0.7
        }
        
        thresholds = thresholds or default_thresholds
        
        for metric in selected_metrics:
            if metric == 'keywords':
                score_result = keyword_pattern_score(prompt, keywords)
                # Combine keyword and pattern scores
                combined_score = 0.7 * score_result['keyword_score'] + 0.3 * score_result['pattern_score']
                results['scores'][metric] = combined_score
                results['scores'][f'{metric}_details'] = score_result
            else:
                score = self.evaluation_functions[metric](prompt)
                results['scores'][metric] = score
            
            # Determine if passed threshold
            score_value = results['scores'][metric]
            if isinstance(score_value, dict):
                score_value = 0.7 * score_value['keyword_score'] + 0.3 * score_value['pattern_score']
            
            results['passed'][metric] = score_value >= thresholds[metric]
            
            # Add recommendations
            if not results['passed'][metric]:
                results['recommendations'].append(self._get_recommendation(metric, score_value))
        
        return results
    
    def evaluate_batch_prompts(self, prompts: List[str], selected_metrics: List[str],
                              thresholds: Optional[Dict[str, float]] = None,
                              keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Evaluate multiple prompts"""
        return [
            self.evaluate_single_prompt(prompt, selected_metrics, thresholds, keywords) 
            for prompt in prompts
        ]
    
    def _get_recommendation(self, metric: str, score: float) -> str:
        """Generate recommendations based on metric performance"""
        recommendations = {
            'length': f"Prompt structure complexity ({score:.3f}) could be improved. Consider simplifying sentence structure and reducing nesting.",
            'keywords': f"Keyword coverage ({score:.3f}) is low. Include more professional prompt terms like 'context:', 'requirements:', 'format:'.",
            'readability': f"Readability ({score:.3f}) needs improvement. Simplify vocabulary and sentence structure.",
            'complexity': f"Syntactic complexity ({score:.3f}) is too high. Break down complex sentences and reduce clause nesting."
        }
        return recommendations.get(metric, f"Consider improving {metric} score: {score:.3f}")
```


## **Step 2: Create Frontend Components**

### `frontend/components/input_components.py`

```python
"""
Input components for the Streamlit UI
"""
import streamlit as st
from typing import List, Dict, Optional

def prompt_input_section() -> str:
    """Render prompt input section"""
    st.subheader("📝 Enter Your Prompt")
    
    input_method = st.radio(
        "Input method:",
        ["Text Area", "File Upload"],
        horizontal=True
    )
    
    if input_method == "Text Area":
        prompt = st.text_area(
            "Prompt text:",
            placeholder="Enter your prompt here...",
            height=200,
            key="prompt_input"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt'],
            key="prompt_file"
        )
        if uploaded_file:
            prompt = uploaded_file.read().decode('utf-8')
            st.text_area("File content:", value=prompt, height=150, disabled=True)
        else:
            prompt = ""
    
    return prompt

def batch_input_section() -> List[str]:
    """Render batch input section"""
    st.subheader("📚 Multiple Prompts")
    
    input_method = st.radio(
        "Batch input method:",
        ["Manual Entry", "CSV Upload"],
        horizontal=True,
        key="batch_method"
    )
    
    prompts = []
    
    if input_method == "Manual Entry":
        num_prompts = st.number_input("Number of prompts:", min_value=1, max_value=10, value=2)
        
        for i in range(num_prompts):
            prompt = st.text_area(
                f"Prompt {i+1}:",
                key=f"batch_prompt_{i}",
                height=100
            )
            if prompt.strip():
                prompts.append(prompt.strip())
    
    else:  # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with 'prompt' column",
            type=['csv'],
            key="batch_csv"
        )
        if uploaded_file:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            if 'prompt' in df.columns:
                prompts = df['prompt'].dropna().tolist()
                st.write(f"Loaded {len(prompts)} prompts from CSV")
            else:
                st.error("CSV must contain a 'prompt' column")
    
    return prompts

def evaluation_settings_sidebar() -> Dict:
    """Render evaluation settings in sidebar"""
    st.sidebar.subheader("⚙️ Evaluation Settings")
    
    # Metric selection
    available_metrics = {
        'length': 'Prompt Length Analysis',
        'keywords': 'Keyword & Pattern Matching', 
        'readability': 'Readability (Flesch-Kincaid)',
        'complexity': 'Syntactic Complexity'
    }
    
    selected_metrics = st.sidebar.multiselect(
        "Select evaluation metrics:",
        options=list(available_metrics.keys()),
        format_func=lambda x: available_metrics[x],
        default=list(available_metrics.keys()),
        key="selected_metrics"
    )
    
    # Threshold settings
    st.sidebar.subheader("🎯 Score Thresholds")
    thresholds = {}
    
    for metric in selected_metrics:
        thresholds[metric] = st.sidebar.slider(
            f"{available_metrics[metric]} threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key=f"threshold_{metric}"
        )
    
    # Keywords for keyword analysis
    keywords = []
    if 'keywords' in selected_metrics:
        st.sidebar.subheader("🔤 Custom Keywords")
        keyword_input = st.sidebar.text_area(
            "Enter keywords (one per line):",
            placeholder="context\nrequirements\nformat",
            key="custom_keywords"
        )
        if keyword_input:
            keywords = [kw.strip() for kw in keyword_input.split('\n') if kw.strip()]
    
    return {
        'selected_metrics': selected_metrics,
        'thresholds': thresholds,
        'keywords': keywords
    }
```


### `frontend/components/results_display.py`

```python
"""
Results display components
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any

def display_single_result(result: Dict[str, Any]):
    """Display results for a single prompt evaluation"""
    st.subheader("📊 Evaluation Results")
    
    # Overall summary
    total_metrics = len(result['scores'])
    passed_metrics = sum(result['passed'].values())
    overall_score = passed_metrics / total_metrics if total_metrics > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{overall_score:.1%}", delta=None)
    with col2:
        st.metric("Passed Metrics", f"{passed_metrics}/{total_metrics}")
    with col3:
        status = "✅ Good" if overall_score >= 0.7 else "⚠️ Needs Work" if overall_score >= 0.5 else "❌ Poor"
        st.metric("Status", status)
    
    # Detailed scores
    st.subheader("📈 Detailed Scores")
    
    scores_data = []
    for metric, score in result['scores'].items():
        if not metric.endswith('_details'):
            passed = result['passed'][metric]
            scores_data.append({
                'Metric': metric.title(),
                'Score': f"{score:.3f}" if isinstance(score, float) else str(score),
                'Status': "✅ Pass" if passed else "❌ Fail"
            })
    
    scores_df = pd.DataFrame(scores_data)
    st.dataframe(scores_df, use_container_width=True)
    
    # Radar chart
    if len(result['scores']) > 1:
        fig = create_radar_chart(result)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    if result['recommendations']:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.warning(f"{i}. {rec}")

def display_batch_results(results: List[Dict[str, Any]]):
    """Display results for batch evaluation"""
    st.subheader("📊 Batch Evaluation Results")
    
    # Summary statistics
    total_prompts = len(results)
    avg_scores = {}
    
    if results:
        # Calculate averages
        all_metrics = set()
        for result in results:
            all_metrics.update(k for k in result['scores'].keys() if not k.endswith('_details'))
        
        for metric in all_metrics:
            scores = [r['scores'][metric] for r in results if metric in r['scores']]
            avg_scores[metric] = sum(scores) / len(scores) if scores else 0
    
    # Display summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Prompts", total_prompts)
    with col2:
        overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        st.metric("Average Score", f"{overall_avg:.3f}")
    
    # Batch comparison chart
    if results:
        fig = create_batch_comparison_chart(results)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual results in expandable sections
    st.subheader("📋 Individual Results")
    for i, result in enumerate(results):
        with st.expander(f"Prompt {i+1} - {result['prompt'][:50]}..."):
            display_single_result(result)

def create_radar_chart(result: Dict[str, Any]) -> go.Figure:
    """Create a radar chart for metric scores"""
    metrics = []
    scores = []
    
    for metric, score in result['scores'].items():
        if not metric.endswith('_details'):
            metrics.append(metric.title())
            scores.append(score if isinstance(score, (int, float)) else 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=metrics,
        fill='toself',
        name='Scores',
        line_color='rgb(102, 153, 255)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Evaluation Metrics Radar Chart"
    )
    
    return fig

def create_batch_comparison_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """Create a comparison chart for batch results"""
    data = []
    
    for i, result in enumerate(results):
        for metric, score in result['scores'].items():
            if not metric.endswith('_details'):
                data.append({
                    'Prompt': f"Prompt {i+1}",
                    'Metric': metric.title(),
                    'Score': score if isinstance(score, (int, float)) else 0
                })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, 
        x='Prompt', 
        y='Score', 
        color='Metric',
        title="Batch Evaluation Comparison",
        barmode='group'
    )
    
    fig.update_layout(yaxis_range=[0, 1])
    return fig
```


## **Step 3: Create the Main App**

### `streamlit_app.py`

```python
"""
Main Streamlit application entry point
"""
import streamlit as st
from backend.evaluation_manager import PromptEvaluationManager
from frontend.components.input_components import (
    prompt_input_section, 
    batch_input_section, 
    evaluation_settings_sidebar
)
from frontend.components.results_display import (
    display_single_result, 
    display_batch_results
)

# Configure page
st.set_page_config(
    page_title="Prompt Quality Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'evaluation_manager' not in st.session_state:
    st.session_state.evaluation_manager = PromptEvaluationManager()

def main():
    st.title("🔍 Prompt Quality Evaluator")
    st.markdown("Analyze your prompts using pre-LLM evaluation methods")
    
    # Sidebar settings
    settings = evaluation_settings_sidebar()
    
    if not settings['selected_metrics']:
        st.warning("Please select at least one evaluation metric from the sidebar.")
        return
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🔍 Single Prompt", "📚 Batch Evaluation"])
    
    with tab1:
        prompt = prompt_input_section()
        
        if st.button("🚀 Evaluate Prompt", type="primary", key="eval_single"):
            if prompt.strip():
                with st.spinner("Evaluating prompt..."):
                    result = st.session_state.evaluation_manager.evaluate_single_prompt(
                        prompt=prompt,
                        selected_metrics=settings['selected_metrics'],
                        thresholds=settings['thresholds'],
                        keywords=settings['keywords']
                    )
                    display_single_result(result)
            else:
                st.error("Please enter a prompt to evaluate.")
    
    with tab2:
        prompts = batch_input_section()
        
        if st.button("🚀 Evaluate All Prompts", type="primary", key="eval_batch"):
            if prompts:
                with st.spinner(f"Evaluating {len(prompts)} prompts..."):
                    results = st.session_state.evaluation_manager.evaluate_batch_prompts(
                        prompts=prompts,
                        selected_metrics=settings['selected_metrics'],
                        thresholds=settings['thresholds'],
                        keywords=settings['keywords']
                    )
                    display_batch_results(results)
            else:
                st.error("Please enter at least one prompt to evaluate.")

if __name__ == "__main__":
    main()
```


## **Step 4: Set Up Dependencies**

### `requirements.txt`

```txt
streamlit>=1.28.0
textstat>=0.7.3
spacy>=3.7.0
pandas>=2.0.0
plotly>=5.15.0
```


### Install spaCy model:

```bash
python -m spacy download en_core_web_sm
```


## **Step 5: Session State Management**

### `utils/session_management.py`

```python
"""
Session state utilities for Streamlit
"""
import streamlit as st
from typing import Any, Dict, List

def initialize_session_state():
    """Initialize session state variables"""
    if 'evaluation_history' not in st.session_state:
        st.session_state.evaluation_history = []
    
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = None

def save_evaluation_to_history(result: Dict[str, Any]):
    """Save evaluation result to history"""
    if 'evaluation_history' not in st.session_state:
        st.session_state.evaluation_history = []
    
    st.session_state.evaluation_history.append(result)
    st.session_state.last_evaluation = result

def get_evaluation_history() -> List[Dict[str, Any]]:
    """Get evaluation history"""
    return st.session_state.get('evaluation_history', [])
```


## **Step 6: Run the Application**

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the app
streamlit run streamlit_app.py
```


## **Next Steps for Enhancement**

1. **Add caching**: Use `@st.cache_data` for expensive operations[^3][^4]
2. **Export functionality**: Add buttons to download results as CSV/PDF
3. **Visualization enhancements**: Add more charts and metrics displays
4. **Error handling**: Implement robust error handling and user feedback
5. **Testing**: Add unit tests for both backend and frontend components
6. **Configuration**: Use `.streamlit/config.toml` for app settings[^1]

This structure provides clean separation between your backend evaluation logic and the Streamlit UI, making it maintainable and scalable. The modular approach allows you to easily add new evaluation metrics or UI components without affecting existing functionality.[^5][^6][^2][^1]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^7][^8][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: https://blog.streamlit.io/best-practices-for-building-genai-apps-with-streamlit/

[^2]: https://www.andymcdonald.scot/how-to-structure-and-organise-a-streamlit-app

[^3]: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

[^4]: https://articles.aceso.no/the-importance-of-session-state-when-creating-a-streamlit-application/

[^5]: https://pybit.es/articles/from-backend-to-frontend-connecting-fastapi-and-streamlit/

[^6]: https://towardsdatascience.com/how-to-build-an-interconnected-multi-page-streamlit-app-3114c313f88f/

[^7]: https://docs.streamlit.io/develop/concepts/architecture/architecture

[^8]: https://www.youtube.com/watch?v=MdjMC0PLJ2s

[^9]: https://auth0.com/blog/introduction-to-streamlit-and-streamlit-components/

[^10]: https://discuss.streamlit.io/t/project-structure-for-medium-and-large-apps-full-example-ui-and-logic-splitted/59967

[^11]: https://blog.streamlit.io/session-state-for-streamlit/

[^12]: https://discuss.streamlit.io/t/what-is-the-pattern-to-connect-streamlit-ui-to-backend-database/44378

[^13]: https://discuss.streamlit.io/t/streamlit-project-folder-structure-for-medium-sized-apps/5272

[^14]: https://docs.streamlit.io/develop/concepts/architecture/session-state

[^15]: https://discuss.streamlit.io/t/clean-architecture-with-streamlit/15262

[^16]: https://www.reddit.com/r/Streamlit/comments/1exogi4/how_to_develop_a_structured_streamlit_app/

[^17]: https://docs.kanaries.net/topics/Streamlit/streamlit-session-state

[^18]: https://dev.to/aws-builders/how-to-build-and-deploy-an-api-driven-streamlitpython-microservice-on-aws-3bkj

[^19]: https://discuss.streamlit.io/t/streamlit-best-practices/57921

[^20]: https://www.youtube.com/watch?v=5l9COMQ3acc

[^21]: https://discuss.streamlit.io/t/high-level-streamlit-system-design-questions/61163

[^22]: https://www.linkedin.com/posts/annafilippova_how-to-structure-and-organise-a-streamlit-activity-7169493533211418624-hhlX

[^23]: https://www.reddit.com/r/Streamlit/comments/1fkobij/streamlit_folder_structure/

[^24]: https://www.youtube.com/watch?v=YKJCkll7WZs

[^25]: https://dev-kit.io/blog/python/streamlit-real-time-design-patterns-creating-interactive-and-dynamic-data-visualizations

[^26]: https://discuss.streamlit.io/t/multipage-app-directory-structure-and-imports/61344

[^27]: https://stackoverflow.com/questions/76318245/util-files-within-streamlit-application

[^28]: https://docs.streamlit.io/develop/concepts

[^29]: https://discuss.streamlit.io/t/question-about-project-structure/28867

[^30]: https://docs.streamlit.io/develop/concepts/design/custom-classes

[^31]: https://discuss.streamlit.io/t/watching-custom-folders/1507

[^32]: https://docs.streamlit.io/develop/concepts/design

[^33]: https://discuss.streamlit.io/t/backend-workings-of-streamlit/11834

[^34]: https://python.plainenglish.io/creating-and-deploying-streamlit-applications-essential-tips-aef69039fc05

