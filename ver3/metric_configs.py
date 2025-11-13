# Complete Metric Configuration for Prompt Evaluation System
# Total Possible Score: 150 points

METRIC_CONFIG = {
    # ========================================
    # STRUCTURE & CLARITY (25 points)
    # ========================================
    "Clarity & Specificity": {
        "index": 1,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,  # 70%
        "group": "structure_clarity",
        "description": "Is the prompt clear and specific about what is being requested?"
    },
    "Explicit Task Definition": {
        "index": 3,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "structure_clarity",
        "description": "Does the prompt explicitly define the task to be performed?"
    },
    "Avoiding Ambiguity or Contradictions": {
        "index": 5,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "structure_clarity",
        "description": "Is the prompt free from ambiguous or contradictory instructions?"
    },
    "Structured / Numbered Instructions": {
        "index": 10,
        "type": "custom_function",
        "max_points": 5,
        "threshold": 3.5,
        "group": "structure_clarity",
        "description": "Does the prompt use structured formatting (bullets, numbers, sections)?"
    },
    "Brevity vs. Detail Balance": {
        "index": 11,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "structure_clarity",
        "description": "Does the prompt balance conciseness with necessary detail?"
    },
    
    # ========================================
    # CONTEXT & INFORMATION (25 points)
    # ========================================
    "Context / Background Provided": {
        "index": 2,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "context_information",
        "description": "Does the prompt provide relevant context or background information?"
    },
    "Desired Output Format / Style": {
        "index": 7,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "context_information",
        "description": "Does the prompt specify the desired output format or style?"
    },
    "Examples or Demonstrations": {
        "index": 13,
        "type": "custom_function",
        "max_points": 5,
        "threshold": 3.5,
        "group": "context_information",
        "description": "Does the prompt include examples or demonstrations?"
    },
    "Knowledge Boundary Awareness": {
        "index": 16,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "context_information",
        "description": "Does the prompt acknowledge what the model may or may not know?"
    },
    "Limitations Disclosure": {
        "index": 30,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "context_information",
        "description": "Does the prompt disclose limitations or constraints?"
    },
    
    # ========================================
    # REASONING & COGNITION (30 points)
    # ========================================
    "Step-by-Step Reasoning Encouraged": {
        "index": 9,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "reasoning_cognition",
        "description": "Does the prompt encourage step-by-step reasoning?"
    },
    "Iteration / Refinement Potential": {
        "index": 12,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "reasoning_cognition",
        "description": "Does the prompt allow for iteration or refinement?"
    },
    "Meta-Cognition Triggers": {
        "index": 20,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "reasoning_cognition",
        "description": "Does the prompt trigger meta-cognitive processes (thinking about thinking)?"
    },
    "Divergent vs. Convergent Thinking Management": {
        "index": 21,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "reasoning_cognition",
        "description": "Does the prompt manage divergent (creative) vs convergent (focused) thinking?"
    },
    "Hypothetical Frame Switching": {
        "index": 22,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "reasoning_cognition",
        "description": "Does the prompt encourage considering different perspectives or scenarios?"
    },
    "Progressive Complexity": {
        "index": 24,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "reasoning_cognition",
        "description": "Does the prompt build complexity progressively?"
    },
    
    # ========================================
    # SAFETY & ALIGNMENT (30 points)
    # ========================================
    "Handling Uncertainty / Gaps": {
        "index": 14,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "safety_alignment",
        "description": "Does the prompt guide how to handle uncertainty or knowledge gaps?"
    },
    "Hallucination Minimization": {
        "index": 15,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "safety_alignment",
        "description": "Does the prompt include mechanisms to minimize hallucinations?"
    },
    "Safe Failure Mode": {
        "index": 23,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "safety_alignment",
        "description": "Does the prompt define safe behavior when the task cannot be completed?"
    },
    "Alignment with Evaluation Metrics": {
        "index": 25,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "safety_alignment",
        "description": "Is the prompt aligned with how success will be evaluated?"
    },
    "Ethical Alignment or Bias Mitigation": {
        "index": 29,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "safety_alignment",
        "description": "Does the prompt address ethical considerations or bias mitigation?"
    },
    "Compression / Summarization Ability": {
        "index": 31,
        "type": "llm_judge",
        "max_points": 5,
        "threshold": 3.5,
        "group": "safety_alignment",
        "description": "Does the prompt leverage or request compression/summarization appropriately?"
    },
    
    # ========================================
    # FORMAT & STYLE (15 points)
    # ========================================
    "Use of Role or Persona": {
        "index": 8,
        "type": "custom_function",
        "max_points": 3,
        "threshold": 2.1,  # 70%
        "group": "format_style",
        "description": "Does the prompt assign a role or persona to the model?"
    },
    "Audience Specification": {
        "index": 17,
        "type": "custom_function",
        "max_points": 3,
        "threshold": 2.1,
        "group": "format_style",
        "description": "Does the prompt specify the target audience?"
    },
    "Style Emulation or Imitation": {
        "index": 18,
        "type": "llm_judge",
        "max_points": 3,
        "threshold": 2.1,
        "group": "format_style",
        "description": "Does the prompt request a specific style to emulate?"
    },
    "Emotional Resonance Calibration": {
        "index": 32,
        "type": "llm_judge",
        "max_points": 3,
        "threshold": 2.1,
        "group": "format_style",
        "description": "Does the prompt calibrate emotional tone appropriately?"
    },
    "Cross-Disciplinary Bridging": {
        "index": 33,
        "type": "llm_judge",
        "max_points": 3,
        "threshold": 2.1,
        "group": "format_style",
        "description": "Does the prompt bridge multiple disciplines or domains?"
    },
    
    # ========================================
    # OUTPUT QUALITY (20 points)
    # ========================================
    "Feasibility within Model Constraints": {
        "index": 4,
        "type": "llm_judge",
        "max_points": 3.33,
        "threshold": 2.33,  # 70%
        "group": "output_quality",
        "description": "Is the request feasible within typical model constraints?"
    },
    "Model Fit / Scenario Appropriateness": {
        "index": 6,
        "type": "llm_judge",
        "max_points": 3.33,
        "threshold": 2.33,
        "group": "output_quality",
        "description": "Is the task appropriate for an LLM to handle?"
    },
    "Output Validation Hooks": {
        "index": 27,
        "type": "custom_function",
        "max_points": 3.33,
        "threshold": 2.33,
        "group": "output_quality",
        "description": "Does the prompt include validation or verification requests?"
    },
    "Time/Effort Estimation Request": {
        "index": 28,
        "type": "custom_function",
        "max_points": 3.33,
        "threshold": 2.33,
        "group": "output_quality",
        "description": "Does the prompt request time or effort estimation?"
    },
    "Output Risk Categorization": {
        "index": 34,
        "type": "llm_judge",
        "max_points": 3.33,
        "threshold": 2.33,
        "group": "output_quality",
        "description": "Does the prompt categorize or acknowledge output risks?"
    },
    "Self-Repair Loops": {
        "index": 35,
        "type": "custom_function",
        "max_points": 3.33,
        "threshold": 2.33,
        "group": "output_quality",
        "description": "Does the prompt enable self-correction or refinement loops?"
    },
    
    # ========================================
    # ADVANCED FEATURES (5 points)
    # ========================================
    "Memory Anchoring": {
        "index": 19,
        "type": "custom_function",
        "max_points": 1.67,
        "threshold": 1.17,  # 70%
        "group": "advanced_features",
        "description": "Does the prompt anchor to previous context or memory?"
    },
    "Calibration Requests": {
        "index": 26,
        "type": "custom_function",
        "max_points": 1.67,
        "threshold": 1.17,
        "group": "advanced_features",
        "description": "Does the prompt request calibration or confidence levels?"
    },
    "Comparison Requests": {
        "index": None,  # Not specified in original
        "type": "custom_function",
        "max_points": 1.66,
        "threshold": 1.16,
        "group": "advanced_features",
        "description": "Does the prompt request comparisons between options?"
    }
}

# Group Summary
GROUP_TOTALS = {
    "structure_clarity": 25,
    "context_information": 25,
    "reasoning_cognition": 30,
    "safety_alignment": 30,
    "format_style": 15,
    "output_quality": 20,
    "advanced_features": 5
}

# System Configuration
CONFIG = {
    "max_score": 150,
    "max_iterations": 5,
    "target_score": 120,  # 80% of 150
    "min_improvement": 3,  # Minimum point improvement to continue
    "metric_threshold_pct": 0.70,  # 70% threshold for each metric
    
    # LLM Judge settings
    "llm_batch_size": 5,  # Metrics per API call
    "first_run_all_metrics": True,
    "reeval_only_below_threshold": True,
}

print("✅ Metric Configuration Complete!")
print(f"Total Metrics: {len(METRIC_CONFIG)}")
print(f"Maximum Score: {CONFIG['max_score']}")
print(f"Target Score: {CONFIG['target_score']} ({CONFIG['target_score']/CONFIG['max_score']*100:.0f}%)")
print("\nGroup Distribution:")
for group, points in GROUP_TOTALS.items():
    count = sum(1 for m in METRIC_CONFIG.values() if m['group'] == group)
    print(f"  {group}: {points} points ({count} metrics)")