import streamlit as st
import time
import json
from datetime import datetime

# Color scheme - Earth tones
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
    """Get color based on score thresholds"""
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
    """Create a colored score display box"""
    color = get_score_color(score, is_cumulative)
    st.markdown(f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px 0;
    ">
        <h4 style="margin: 0; color: white;">{title}</h4>
        <h2 style="margin: 5px 0; color: white;">{score:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

def collapsible_explanation(title, content):
    """Create a collapsible explanation section"""
    with st.expander(f"📋 {title} - Reasoning & Details"):
        st.write(content)

# Initialize session state
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'original_output' not in st.session_state:
    st.session_state.original_output = ""
if 'rewrite_before' not in st.session_state:
    st.session_state.rewrite_before = ""
if 'rewrite_after' not in st.session_state:
    st.session_state.rewrite_after = ""
if 'pre_scores' not in st.session_state:
    st.session_state.pre_scores = {}
if 'post_scores' not in st.session_state:
    st.session_state.post_scores = {}
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}

# Main UI
st.title("Prompt OptEval")
st.markdown("---")

# Input Section
col1, col2 = st.columns([1, 3])
with col1:
    prompt_name = st.text_input("Input prompt name:")
with col2:
    prompt_text = st.text_area("Input Prompt:", height=120)

# Optional existing output
st.subheader("Optional: Do you have an existing output to evaluate?")
has_existing_output = st.radio("", ["No, generate new output", "Yes, I have an output to evaluate"], horizontal=True)

existing_output = ""
if has_existing_output == "Yes, I have an output to evaluate":
    existing_output = st.text_area("Paste your existing output here:", height=150)

# Split buttons - Optimize and Evaluate
col1, col2 = st.columns(2)

with col1:
    if st.button("Optimize", type="primary", use_container_width=True):
        if prompt_text and prompt_name:
            # Loading sequence for optimization
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Evaluating prompt
            status_text.text("Evaluating prompt...")
            progress_bar.progress(50)
            time.sleep(2)
            
            # Mock pre-execution scores (replace with actual evaluation)
            st.session_state.pre_scores = {
                'clarity': 3.2,
                'specificity': 4.1,
                'completeness': 3.8,
                'structure': 3.5,
                'complexity': 4.0
            }
            
            # Step 2: Optimizing
            status_text.text("Optimizing...")
            progress_bar.progress(100)
            time.sleep(2)
            
            # Mock optimization output (replace with actual optimization)
            st.session_state.rewrite_before = "This is a mock optimized prompt based on pre-execution feedback. The actual implementation would use the pre-execution scores to improve the prompt clarity, specificity, completeness, structure, and complexity."
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.optimization_complete = True
            st.success("Optimization complete!")
            st.rerun()

with col2:
    if st.button("Evaluate", type="primary", use_container_width=True):
        if prompt_text and prompt_name:
            # Loading sequence for evaluation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Evaluating prompt (if not already done)
            if not st.session_state.pre_scores:
                status_text.text("Evaluating prompt...")
                progress_bar.progress(33)
                time.sleep(2)
                
                # Mock pre-execution scores (replace with actual evaluation)
                st.session_state.pre_scores = {
                    'clarity': 3.2,
                    'specificity': 4.1,
                    'completeness': 3.8,
                    'structure': 3.5,
                    'complexity': 4.0
                }
            else:
                progress_bar.progress(33)
            
            # Step 2: Running prompt
            status_text.text("Running Prompt...")
            progress_bar.progress(66)
            time.sleep(2)
            
            if existing_output:
                st.session_state.original_output = existing_output
            else:
                # Mock output generation (replace with actual LLM call)
                st.session_state.original_output = "This is a mock generated output based on your prompt. In the actual implementation, this would be the LLM response."
            
            # Mock post-execution scores (replace with actual evaluation)
            st.session_state.post_scores = {
                'relevance': 3.9,
                'accuracy': 4.2,
                'completeness': 3.7,
                'coherence': 4.0,
                'format': 3.5,
                'safety': 4.8
            }
            
            # Step 3: Final evaluation
            status_text.text("Finalizing evaluation...")
            progress_bar.progress(100)
            time.sleep(1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.evaluation_complete = True
            st.success("Evaluation complete!")
            st.rerun()

# Display results if optimization is complete
if st.session_state.optimization_complete and st.session_state.pre_scores:
    st.markdown("---")
    st.subheader("Optimization Results")
    
    # Calculate pre-execution cumulative score
    pre_cumulative = sum(st.session_state.pre_scores.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Initial Prompt Score (Pre-execution)**")
        score_display_box("Pre-execution Score", pre_cumulative, is_cumulative=True)
        
        # Individual pre-execution scores
        st.markdown("**Individual Scores:**")
        for dimension, score in st.session_state.pre_scores.items():
            st.markdown(f"**{dimension.title()}**: {score:.2f}")
    
    with col2:
        st.markdown("**Optimized Prompt**")
        st.markdown(f"""
        <div style="
            background-color: #F8F9FA;
            border: 2px solid {COLORS['olive']};
            padding: 15px;
            border-radius: 8px;
            height: 200px;
            overflow-y: scroll;
        ">
            {st.session_state.rewrite_before}
        </div>
        """, unsafe_allow_html=True)
        
        collapsible_explanation("Optimization Reasoning", "This optimization was generated based on the pre-execution evaluation scores. The AI identified areas for improvement in prompt clarity, specificity, completeness, structure, and complexity.")

# Display results if evaluation is complete
if st.session_state.evaluation_complete:
    st.markdown("---")
    st.subheader("Evaluation Results")
    
    # Calculate cumulative scores
    pre_cumulative = sum(st.session_state.pre_scores.values())
    post_cumulative = sum(st.session_state.post_scores.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Initial Prompt Score (Pre-execution)**")
        score_display_box("Pre-execution Score", pre_cumulative, is_cumulative=True)
        
        # Individual pre-execution scores
        st.markdown("**Individual Scores:**")
        for dimension, score in st.session_state.pre_scores.items():
            st.markdown(f"**{dimension.title()}**: {score:.2f}")
    
    with col2:
        st.markdown("**Prompt Output Grade (Post-execution)**")
        score_display_box("Post-execution Score", post_cumulative, is_cumulative=True)
        
        # Individual post-execution scores
        st.markdown("**Individual Scores:**")
        for dimension, score in st.session_state.post_scores.items():
            st.markdown(f"**{dimension.title()}**: {score:.2f}")
    
    st.markdown("---")
    
    # User Feedback Section
    st.subheader("Take User Feedback")
    st.markdown("Do you agree with the grade?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Higher", use_container_width=True):
            st.session_state.user_feedback['rating'] = 'higher'
            st.success("Feedback recorded: Higher")
    
    with col2:
        if st.button("Lower", use_container_width=True):
            st.session_state.user_feedback['rating'] = 'lower'
            st.success("Feedback recorded: Lower")
    
    with col3:
        if st.button("Neutral", use_container_width=True):
            st.session_state.user_feedback['rating'] = 'neutral'
            st.success("Feedback recorded: Neutral")
    
    # Optional text feedback
    feedback_text = st.text_area("Additional feedback (optional):", height=80)
    if feedback_text:
        st.session_state.user_feedback['text'] = feedback_text
    
    # Generate rewrite after score based on feedback
    if st.session_state.user_feedback and not st.session_state.rewrite_after:
        st.session_state.rewrite_after = "This is a mock rewrite based on post-execution feedback and user input. The actual implementation would incorporate user feedback to address issues in relevance, accuracy, completeness, coherence, format adherence, and safety."
    
    st.markdown("---")
    
    # Choose Preferred Output Section
    st.subheader("Choose Preferred Output")
    
    # Three column layout for outputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original**")
        st.markdown(f"""
        <div style="
            background-color: #F8F9FA;
            border: 2px solid {COLORS['charcoal']};
            padding: 15px;
            border-radius: 8px;
            height: 300px;
            overflow-y: scroll;
        ">
            {st.session_state.original_output}
        </div>
        """, unsafe_allow_html=True)
        
        collapsible_explanation("Original Output", "This is the reasoning for the original output evaluation. In the actual implementation, this would contain detailed analysis of why the output received its scores.")
        
        if st.button("Select Original", key="select_original", use_container_width=True):
            st.session_state.preferred_output = 'original'
            st.success("Original output selected")
    
    with col2:
        st.markdown("**AI Rewrite Before Score**")
        rewrite_before_display = st.session_state.rewrite_before if st.session_state.rewrite_before else "Click 'Optimize' to generate optimized prompt"
        st.markdown(f"""
        <div style="
            background-color: #F8F9FA;
            border: 2px solid {COLORS['olive']};
            padding: 15px;
            border-radius: 8px;
            height: 300px;
            overflow-y: scroll;
        ">
            {rewrite_before_display}
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.rewrite_before:
            collapsible_explanation("AI Rewrite (Pre-score)", "This rewrite was generated based on the pre-execution evaluation scores. The AI identified areas for improvement in prompt clarity, specificity, completeness, structure, and complexity.")
            
            if st.button("Select Rewrite Before", key="select_before", use_container_width=True):
                st.session_state.preferred_output = 'rewrite_before'
                st.success("Rewrite (before score) selected")
    
    with col3:
        st.markdown("**AI Rewrite After Score**")
        rewrite_after_display = st.session_state.rewrite_after if st.session_state.rewrite_after else "Provide user feedback to generate this rewrite"
        st.markdown(f"""
        <div style="
            background-color: #F8F9FA;
            border: 2px solid {COLORS['wine']};
            padding: 15px;
            border-radius: 8px;
            height: 300px;
            overflow-y: scroll;
        ">
            {rewrite_after_display}
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.rewrite_after:
            collapsible_explanation("AI Rewrite (Post-score)", "This rewrite incorporates both the post-execution evaluation and your user feedback. It addresses issues in relevance, accuracy, completeness, coherence, format adherence, and safety based on your input.")
            
            if st.button("Select Rewrite After", key="select_after", use_container_width=True):
                st.session_state.preferred_output = 'rewrite_after'
                st.success("Rewrite (after score) selected")
    
    # Save evaluation button
    st.markdown("---")
    if st.button("Save Evaluation", type="primary"):
        # Mock save functionality - in actual implementation, save to database/file
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            'prompt_name': prompt_name,
            'prompt_text': prompt_text,
            'pre_scores': st.session_state.pre_scores,
            'post_scores': st.session_state.post_scores,
            'user_feedback': st.session_state.user_feedback,
            'preferred_output': st.session_state.get('preferred_output', ''),
            'original_output': st.session_state.original_output,
            'rewrite_before': st.session_state.rewrite_before,
            'rewrite_after': st.session_state.rewrite_after
        }
        
        st.success("Evaluation saved successfully!")
        st.json(evaluation_data)  # For demo - remove in production