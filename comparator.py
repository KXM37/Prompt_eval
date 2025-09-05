import subprocess
import json
import yaml
from datetime import datetime
from pathlib import Path

def enhanced_compare_prompts(registry, client, version_a, input_text, prompt_engineering_strategy="improve_clarity"):
    """
    Enhanced prompt comparator that uses promptfoo for automatic prompt engineering
    and comparison workflow.
    """
    # Get original prompt
    prompt_a = registry.get_prompt(version_a)["text"]
    
    # Generate engineered prompt using promptfoo dynamic generation
    engineered_prompt = generate_engineered_prompt(
        prompt_a, 
        input_text, 
        strategy=prompt_engineering_strategy
    )
    
    # Create temporary promptfoo configuration
    config = create_promptfoo_config(prompt_a, engineered_prompt, input_text)
    
    # Run promptfoo evaluation
    results = run_promptfoo_evaluation(config)
    
    # Display results and get user choice
    winner = display_results_and_get_choice(results, prompt_a, engineered_prompt)
    
    # Save feedback with promptfoo integration
    feedback = save_enhanced_feedback(
        version_a, prompt_a, engineered_prompt, 
        results, winner, input_text
    )
    
    return feedback
# prompt_engineer.py
  
def generate_improved_prompt(context):
    """Dynamic prompt generation for engineering improvements"""
    vars_data = context['vars']
    original_prompt = vars_data['original_prompt']
    strategy = vars_data.get('strategy', 'improve_clarity')
    
    if strategy == "improve_clarity":
        return f"""
{original_prompt}

Please be more specific and clear in your response. 
Structure your answer with clear steps and examples.
"""
    elif strategy == "add_context":
        return f"""
Context: You are an expert assistant helping users understand complex topics.

{original_prompt}

Provide comprehensive context and background information in your response.
"""
    elif strategy == "increase_specificity":
        return f"""
{original_prompt}

Be very specific in your response. Include:
- Concrete examples
- Step-by-step explanations  
- Relevant details and context
"""
    
    return original_prompt

  
def create_promptfoo_config(original_prompt, engineered_prompt, input_text):
    """Create promptfoo configuration for comparison"""
    config = {
        "description": "Prompt comparison and engineering evaluation",
        "prompts": [
            {
                "id": "original",
                "label": "Original Prompt",
                "prompt": original_prompt
            },
            {
                "id": "engineered", 
                "label": "Engineered Prompt",
                "prompt": engineered_prompt
            }
        ],
        "providers": [
            "openai:gpt-4"  # or your preferred provider
        ],
        "tests": [
            {
                "vars": {
                    "input": input_text
                },
                "assert": [
                    {
                        "type": "llm-rubric",
                        "value": "Response is helpful, accurate, and well-structured"
                    }
                ]
            }
        ],
        "outputPath": "evaluation_results.json"
    }
    
    # Write config to temporary file
    config_path = Path("temp_promptfoo_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

          
def run_promptfoo_evaluation(config_path):
    """Run promptfoo evaluation and return results"""
    try:
        # Run promptfoo evaluation
        result = subprocess.run([
            "npx", "promptfoo", "eval", 
            "--config", str(config_path),
            "--output", "evaluation_results.json"
        ], capture_output=True, text=True, check=True)
        
        # Load and return results
        with open("evaluation_results.json", 'r') as f:
            return json.load(f)
            
    except subprocess.CalledProcessError as e:
        print(f"Promptfoo evaluation failed: {e}")
        return None

  
def display_results_and_get_choice(results, original_prompt, engineered_prompt):
    """Display comparison results and get user choice"""
    if not results:
        print("Evaluation failed. Falling back to basic comparison.")
        return compare_basic(original_prompt, engineered_prompt)
    
    print("\n=== PROMPT COMPARISON RESULTS ===")
    
    # Display original vs engineered outputs
    for i, result in enumerate(results.get('results', [])):
        prompt_label = "Original" if i == 0 else "Engineered"
        output = result.get('response', {}).get('output', '')
        score = result.get('score', 'N/A')
        
        print(f"\n[ {prompt_label} ] (Score: {score})")
        print(f"Output: {output}")
        
        # Show any assertion results
        if 'gradingResult' in result:
            grading = result['gradingResult']
            print(f"Quality Assessment: {grading.get('reason', 'N/A')}")
    
    # Get user choice
    choice = input("\nWhich prompt performed better? (Original/Engineered/Tie): ").strip().lower()
    
    return choice

  
def save_enhanced_feedback(version_a, original_prompt, engineered_prompt, results, winner, input_text):
    """Save feedback with promptfoo evaluation data"""
    feedback = {
        'timestamp': datetime.now().isoformat(),
        'input': input_text,
        'original_prompt_version': version_a,
        'original_prompt': original_prompt,
        'engineered_prompt': engineered_prompt,
        'winner': winner,
        'promptfoo_results': results,
        'evaluation_scores': extract_scores(results) if results else None
    }
    
    # Save to feedback file
    with open("../data/enhanced_feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback) + "\n")
    
    print("Enhanced feedback saved with evaluation metrics!")
    return feedback

      
def extract_scores(results):
    """Extract scoring metrics from promptfoo results"""
    if not results or 'results' not in results:
        return None
        
    scores = {}
    for i, result in enumerate(results['results']):
        label = "original" if i == 0 else "engineered" 
        scores[label] = {
            'overall_score': result.get('score', 0),
            'assertion_results': result.get('gradingResult', {}),
            'token_usage': result.get('response', {}).get('tokenUsage', {}),
            'cost': result.get('response', {}).get('cost', 0)
        }
    
    return scores

      
def generate_engineered_prompt(original_prompt, input_text, strategy="improve_clarity"):
    """Generate an engineered version of the prompt using various strategies"""
    
    # Create a temporary promptfoo config for dynamic generation
    dynamic_config = {
        "prompts": ["file://prompt_engineer.py:generate_improved_prompt"],
        "providers": ["openai:gpt-4"],
        "tests": [{
            "vars": {
                "original_prompt": original_prompt,
                "input_text": input_text,
                "strategy": strategy
            }
        }]
    }
    
    # This would use promptfoo's dynamic generation capabilities
    # For simplicity, we'll implement basic strategies here
    
    strategies = {
        "improve_clarity": lambda p: f"{p}\n\nPlease provide a clear, step-by-step response with specific examples.",
        "add_context": lambda p: f"You are an expert assistant. {p}\n\nProvide comprehensive background and context.",
        "increase_specificity": lambda p: f"{p}\n\nBe very specific and include concrete details, examples, and actionable information."
    }
    
    return strategies.get(strategy, lambda p: p)(original_prompt)

      
# Updated version of your original function
def compare_prompts_with_engineering(registry, client, version_a, input_text, 
                                   auto_engineer=True, engineering_strategy="improve_clarity"):
    """
    Enhanced version that can optionally auto-engineer prompts or compare existing versions
    """
    
    if auto_engineer:
        # Use the enhanced workflow with automatic prompt engineering
        return enhanced_compare_prompts(registry, client, version_a, input_text, engineering_strategy)
    else:
        # Fall back to original two-version comparison
        version_b = input("Enter second prompt version ID: ").strip()
        return compare_prompts(registry, client, version_a, version_b, input_text)
