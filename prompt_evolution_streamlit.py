"""
Prompt Evolution System - Streamlit UI
Interactive interface for prompt optimization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path
import sys

# Import core engine (assumes prompt_evolution_core.py is in same directory)
from prompt_evolution_core import (
    PromptEvolutionEngine, 
    TemplateAPIClient,
    METRIC_GROUPS
)

# Page config
st.set_page_config(
    page_title="Prompt Evolution System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evolution_running' not in st.session_state:
    st.session_state.evolution_running = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'engine' not in st.session_state:
    st.session_state.engine = None

def init_engine(api_client, top_n, max_iter, threshold):
    """Initialize evolution engine with settings"""
    return PromptEvolutionEngine(
        api_client=api_client,
        top_n_worst=top_n,
        max_iterations=max_iter,
        improvement_threshold=threshold / 100  # Convert from percentage
    )

def plot_evolution_trajectory(results):
    """Plot metric evolution over iterations"""
    trajectories = results['metric_trajectories']
    
    fig = go.Figure()
    
    # Plot each metric
    for metric, data in trajectories.items():
        iterations = [d['iteration'] for d in data]
        scores = [d['score'] for d in data]
        
        fig.add_trace(go.Scatter(
            x=iterations,
            y=scores,
            mode='lines+markers',
            name=metric,
            hovertemplate=f'<b>{metric}</b><br>' +
                         'Iteration: %{x}<br>' +
                         'Score: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Metric Evolution Across Iterations",
        xaxis_title="Iteration",
        yaxis_title="Score (0-300)",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_strategy_performance(results):
    """Plot which strategy performed best"""
    strategy_counts = {"aggressive": 0, "balanced": 0, "structural": 0, "initial": 0}
    
    for iteration in results['evolution_history']:
        best_idx = iteration['best_candidate_idx']
        strategy = iteration['candidates'][best_idx]['strategy']
        strategy_counts[strategy] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(strategy_counts.keys()),
            y=list(strategy_counts.values()),
            marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe']
        )
    ])
    
    fig.update_layout(
        title="Strategy Win Rate",
        xaxis_title="Strategy",
        yaxis_title="Times Selected as Best",
        height=400
    )
    
    return fig

def plot_score_comparison(results):
    """Plot candidate scores for each iteration"""
    data = []
    
    for iteration in results['evolution_history']:
        for idx, (candidate, evaluation) in enumerate(zip(
            iteration['candidates'], 
            iteration['evaluations']
        )):
            data.append({
                'Iteration': iteration['iteration'],
                'Candidate': f"{candidate['strategy']}",
                'Score': evaluation['overall_score'],
                'Selected': idx == iteration['best_candidate_idx']
            })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='Iteration',
        y='Score',
        color='Candidate',
        pattern_shape='Selected',
        barmode='group',
        title="Candidate Scores by Iteration",
        height=400
    )
    
    return fig

def display_metric_heatmap(results):
    """Display heatmap of metric improvements"""
    # Get all unique metrics
    all_metrics = []
    for group in METRIC_GROUPS.values():
        all_metrics.extend(group['metrics'])
    
    # Build score matrix
    iterations = []
    scores_matrix = []
    
    for iteration in results['evolution_history']:
        best_idx = iteration['best_candidate_idx']
        eval_scores = iteration['evaluations'][best_idx]['scores']
        iterations.append(f"Iter {iteration['iteration']}")
        scores_matrix.append([eval_scores.get(m, 0) for m in all_metrics])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=scores_matrix,
        x=all_metrics,
        y=iterations,
        colorscale='Viridis',
        hovertemplate='Metric: %{x}<br>Iteration: %{y}<br>Score: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Metric Score Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Iterations",
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig

# ==================== UI LAYOUT ====================

# Header
st.markdown('<div class="main-header">🧬 Prompt Evolution System</div>', unsafe_allow_html=True)
st.markdown("**Meta-optimization framework for iteratively improving prompts using LLM-as-Judge**")
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Evolution Settings")
    max_iterations = st.slider("Max Iterations", 1, 20, 6)
    improvement_threshold = st.slider("Stop Threshold (%)", 0.01, 1.0, 0.10, 0.01)
    top_n_worst = st.slider("Top N Worst Metrics", 3, 15, 5)
    
    st.subheader("API Configuration")
    st.info("⚠️ Configure your internal API in `TemplateAPIClient`")
    
    # Option to load previous runs
    st.subheader("📁 Load Previous Run")
    storage_dir = Path("evolution_data")
    if storage_dir.exists():
        previous_runs = [d.name for d in storage_dir.iterdir() if d.is_dir()]
        if previous_runs:
            selected_run = st.selectbox("Select Run", [""] + previous_runs)
            if selected_run and st.button("Load Results"):
                results_path = storage_dir / selected_run / "results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        st.session_state.results = json.load(f)
                    st.success(f"Loaded: {selected_run}")
                    st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Run Evolution", 
    "📊 Results Dashboard", 
    "📈 Metric Analysis",
    "💾 Export & History"
])

# Tab 1: Run Evolution
with tab1:
    st.header("Run Prompt Evolution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Initial Prompt")
        initial_prompt = st.text_area(
            "Enter your prompt to optimize:",
            height=300,
            placeholder="Enter the prompt you want to optimize...",
            help="This is the starting point for the evolution process"
        )
        
        run_name = st.text_input(
            "Run Name (optional)",
            placeholder=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Metrics", "35")
        st.metric("Metric Groups", "7")
        st.metric("Candidates/Iteration", "3")
        
        st.info("""
        **Optimization Strategies:**
        - 🎯 Aggressive: Focus on worst metrics
        - ⚖️ Balanced: Holistic improvement
        - 🏗️ Structural: Format & organization
        """)
    
    # Run button
    if st.button("▶️ Start Evolution", type="primary", disabled=not initial_prompt):
        st.session_state.evolution_running = True
        
        # Initialize engine (replace with your API client)
        try:
            # TODO: Replace TemplateAPIClient with your actual API integration
            api_client = TemplateAPIClient()
            
            engine = init_engine(
                api_client=api_client,
                top_n=top_n_worst,
                max_iter=max_iterations,
                threshold=improvement_threshold
            )
            
            # Run evolution with progress tracking
            with st.spinner("🔄 Evolution in progress..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run evolution
                results = engine.evolve(
                    initial_prompt=initial_prompt,
                    run_name=run_name if run_name else None
                )
                
                progress_bar.progress(100)
                status_text.success("✅ Evolution complete!")
                
                st.session_state.results = results
                st.session_state.evolution_running = False
                
                st.balloons()
                st.rerun()
                
        except NotImplementedError:
            st.error("""
            ⚠️ **API Not Configured**
            
            Please implement your internal API in the `TemplateAPIClient` class:
            1. Replace `evaluate()` method with your LLM API call
            2. Replace `optimize()` method with your LLM API call
            
            See `prompt_evolution_core.py` for details.
            """)
            st.session_state.evolution_running = False

# Tab 2: Results Dashboard
with tab2:
    if st.session_state.results:
        results = st.session_state.results
        summary = results['summary']
        
        st.header("📊 Evolution Results Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Initial Score",
                f"{summary['initial_score']:.2f}",
                help="Starting overall score"
            )
        
        with col2:
            st.metric(
                "Final Score",
                f"{summary['final_score']:.2f}",
                delta=f"+{summary['final_score'] - summary['initial_score']:.2f}"
            )
        
        with col3:
            st.metric(
                "Total Improvement",
                f"{summary['total_improvement_pct']:.2f}%",
                delta=f"{summary['total_improvement_pct']:.2f}%"
            )
        
        with col4:
            st.metric(
                "Iterations",
                summary['iterations_completed'],
                help="Number of evolution iterations"
            )
        
        st.markdown("---")
        
        # Best strategy
        st.subheader("🏆 Best Performing Strategy")
        st.info(f"**{summary['best_strategy'].title()}** strategy won most often")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_strategy_performance(results),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_score_comparison(results),
                use_container_width=True
            )
        
        # Prompts comparison
        st.markdown("---")
        st.subheader("📝 Prompt Evolution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Initial Prompt:**")
            st.code(results['initial_prompt'], language=None)
        
        with col2:
            st.markdown("**Final Optimized Prompt:**")
            st.code(results['final_prompt'], language=None)
            
            if st.button("📋 Copy Final Prompt"):
                st.code(results['final_prompt'])
                st.success("Prompt ready to copy!")
    
    else:
        st.info("👈 Run an evolution or load previous results to view dashboard")

# Tab 3: Metric Analysis
with tab3:
    if st.session_state.results:
        results = st.session_state.results
        
        st.header("📈 Detailed Metric Analysis")
        
        # Evolution trajectory
        st.subheader("Metric Evolution Over Time")
        st.plotly_chart(
            plot_evolution_trajectory(results),
            use_container_width=True
        )
        
        # Heatmap
        st.subheader("Score Heatmap")
        st.plotly_chart(
            display_metric_heatmap(results),
            use_container_width=True
        )
        
        # Detailed iteration breakdown
        st.markdown("---")
        st.subheader("Iteration-by-Iteration Breakdown")
        
        for iteration in results['evolution_history']:
            with st.expander(f"Iteration {iteration['iteration']} - {iteration['timestamp'][:10]}"):
                
                # Show all candidates
                for idx, (candidate, evaluation) in enumerate(zip(
                    iteration['candidates'],
                    iteration['evaluations']
                )):
                    is_best = idx == iteration['best_candidate_idx']
                    
                    st.markdown(f"**Candidate {idx+1}: {candidate['strategy'].title()}** {'🏆' if is_best else ''}")
                    st.caption(f"_{candidate['generation_rationale']}_")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Overall Score", f"{evaluation['overall_score']:.2f}/300")
                    
                    with col2:
                        # Top 5 and bottom 5 metrics
                        scores = evaluation['scores']
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
                        
                        st.caption("**Lowest Scores:**")
                        for metric, score in sorted_scores[:3]:
                            st.text(f"• {metric}: {score:.1f}")
                    
                    if st.checkbox(f"Show full prompt (Candidate {idx+1}, Iter {iteration['iteration']})", key=f"prompt_{iteration['iteration']}_{idx}"):
                        st.code(candidate['prompt'], language=None)
                    
                    st.markdown("---")
                
                if iteration['iteration'] > 1:
                    st.info(f"📊 Improvement: {iteration['improvement']*100:.3f}%")
    
    else:
        st.info("👈 Run an evolution or load previous results to view analysis")

# Tab 4: Export & History
with tab4:
    st.header("💾 Export & History")
    
    if st.session_state.results:
        results = st.session_state.results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Results")
            
            # Export as JSON
            if st.button("📥 Download Full Results (JSON)"):
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"evolution_results_{results['run_name']}.json",
                    mime="application/json"
                )
            
            # Export final prompt
            if st.button("📥 Download Final Prompt (TXT)"):
                st.download_button(
                    label="Download Prompt",
                    data=results['final_prompt'],
                    file_name=f"optimized_prompt_{results['run_name']}.txt",
                    mime="text/plain"
                )
            
            # Export metrics CSV
            if st.button("📥 Download Metrics (CSV)"):
                # Build dataframe
                data = []
                for iteration in results['evolution_history']:
                    best_idx = iteration['best_candidate_idx']
                    eval_data = iteration['evaluations'][best_idx]
                    
                    row = {
                        'iteration': iteration['iteration'],
                        'overall_score': eval_data['overall_score'],
                        **eval_data['scores']
                    }
                    data.append(row)
                
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"metrics_{results['run_name']}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("Run Information")
            st.json({
                "run_name": results['run_name'],
                "total_iterations": results['total_iterations'],
                "initial_score": results['summary']['initial_score'],
                "final_score": results['summary']['final_score'],
                "improvement_pct": results['summary']['total_improvement_pct']
            })
    
    # Previous runs
    st.markdown("---")
    st.subheader("📚 Previous Runs")
    
    storage_dir = Path("evolution_data")
    if storage_dir.exists():
        runs = sorted(
            [d for d in storage_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if runs:
            for run_dir in runs:
                results_file = run_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        run_data = json.load(f)
                    
                    with st.expander(f"📁 {run_dir.name}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Final Score", f"{run_data['summary']['final_score']:.2f}")
                        with col2:
                            st.metric("Improvement", f"{run_data['summary']['total_improvement_pct']:.2f}%")
                        with col3:
                            st.metric("Iterations", run_data['total_iterations'])
                        
                        if st.button("Load This Run", key=f"load_{run_dir.name}"):
                            st.session_state.results = run_data
                            st.rerun()
        else:
            st.info("No previous runs found")
    else:
        st.info("No evolution data directory found")

# Footer
st.markdown("---")
st.caption("🧬 Prompt Evolution System v1.0 | Built with Streamlit")
