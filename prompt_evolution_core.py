"""
Prompt Evolution System - Core Engine
A meta-optimization system for iteratively improving prompts using LLM-as-Judge evaluation
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import copy

# Metric groupings for batched evaluation (35 metrics → 6 groups)
METRIC_GROUPS = {
    "structure_clarity": {
        "metrics": [
            "Clarity & Specificity",
            "Explicit Task Definition", 
            "Avoiding Ambiguity or Contradictions",
            "Structured / Numbered Instructions",
            "Brevity vs. Detail Balance"
        ],
        "indices": [1, 3, 5, 10, 11]
    },
    "context_information": {
        "metrics": [
            "Context / Background Provided",
            "Desired Output Format / Style",
            "Examples or Demonstrations",
            "Knowledge Boundary Awareness",
            "Limitations Disclosure"
        ],
        "indices": [2, 7, 13, 16, 30]
    },
    "reasoning_cognition": {
        "metrics": [
            "Step-by-Step Reasoning Encouraged",
            "Iteration / Refinement Potential",
            "Meta-Cognition Triggers",
            "Divergent vs. Convergent Thinking Management",
            "Hypothetical Frame Switching",
            "Progressive Complexity"
        ],
        "indices": [9, 12, 20, 21, 22, 24]
    },
    "safety_alignment": {
        "metrics": [
            "Handling Uncertainty / Gaps",
            "Hallucination Minimization",
            "Safe Failure Mode",
            "Alignment with Evaluation Metrics",
            "Ethical Alignment or Bias Mitigation",
            "Compression / Summarization Ability"
        ],
        "indices": [14, 15, 23, 25, 29, 31]
    },
    "format_style": {
        "metrics": [
            "Use of Role or Persona",
            "Audience Specification",
            "Style Emulation or Imitation",
            "Emotional Resonance Calibration",
            "Cross-Disciplinary Bridging"
        ],
        "indices": [8, 17, 18, 32, 33]
    },
    "output_quality": {
        "metrics": [
            "Feasibility within Model Constraints",
            "Model Fit / Scenario Appropriateness",
            "Output Validation Hooks",
            "Time/Effort Estimation Request",
            "Output Risk Categorization",
            "Self-Repair Loops"
        ],
        "indices": [4, 6, 27, 28, 34, 35]
    },
    "advanced_features": {
        "metrics": [
            "Memory Anchoring (in Multi-Turn Systems)",
            "Calibration Requests",
            "Comparison requests"
        ],
        "indices": [19, 26]
    }
}

@dataclass
class EvaluationResult:
    """Results from a single evaluation"""
    scores: Dict[str, float]  # metric_name -> score (0-300)
    overall_score: float
    timestamp: str
    prompt_version: str

@dataclass
class CandidatePrompt:
    """A candidate prompt with its metadata"""
    prompt: str
    strategy: str  # 'aggressive', 'balanced', 'structural'
    generation_rationale: str
    parent_version: str

@dataclass
class IterationResult:
    """Results from one iteration of the evolution"""
    iteration: int
    candidates: List[CandidatePrompt]
    evaluations: List[EvaluationResult]
    best_candidate_idx: int
    improvement: float
    timestamp: str

class PromptEvolutionEngine:
    """Core engine for prompt evolution"""
    
    def __init__(self, api_client=None, top_n_worst=5, max_iterations=6, 
                 improvement_threshold=0.001):  # 0.10%
        self.api_client = api_client or TemplateAPIClient()
        self.top_n_worst = top_n_worst
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.evolution_history = []
        
        # Storage
        self.storage_dir = Path("evolution_data")
        self.storage_dir.mkdir(exist_ok=True)
        
    def evaluate_prompt(self, prompt: str, version: str) -> EvaluationResult:
        """Evaluate a prompt across all metric groups"""
        all_scores = {}
        
        for group_name, group_data in METRIC_GROUPS.items():
            group_scores = self._evaluate_group(prompt, group_name, group_data)
            all_scores.update(group_scores)
        
        overall_score = np.mean(list(all_scores.values()))
        
        return EvaluationResult(
            scores=all_scores,
            overall_score=overall_score,
            timestamp=datetime.now().isoformat(),
            prompt_version=version
        )
    
    def _evaluate_group(self, prompt: str, group_name: str, 
                       group_data: Dict) -> Dict[str, float]:
        """Evaluate a group of metrics using LLM-as-Judge"""
        evaluation_prompt = self._build_evaluation_prompt(
            prompt, group_name, group_data['metrics']
        )
        
        # Call API (templated for user's internal API)
        response = self.api_client.evaluate(evaluation_prompt)
        
        # Parse response into scores
        scores = self._parse_evaluation_response(response, group_data['metrics'])
        return scores
    
    def _build_evaluation_prompt(self, prompt: str, group_name: str, 
                                 metrics: List[str]) -> str:
        """Build the evaluation prompt for LLM-as-Judge"""
        return f"""You are an expert prompt engineer evaluating prompt quality.

Evaluate the following prompt across these {len(metrics)} metrics related to {group_name.replace('_', ' ').title()}:

{chr(10).join(f"{i+1}. {metric}" for i, metric in enumerate(metrics))}

PROMPT TO EVALUATE:
'''
{prompt}
'''

For each metric, provide a score from 0-300 where:
- 0-100: Poor/Missing
- 101-200: Adequate/Present
- 201-300: Excellent/Optimal

Respond ONLY with a JSON object in this exact format:
{{
    "Metric Name 1": score,
    "Metric Name 2": score,
    ...
    "reasoning": "Brief explanation of key strengths and weaknesses"
}}"""

    def _parse_evaluation_response(self, response: str, 
                                   metrics: List[str]) -> Dict[str, float]:
        """Parse LLM evaluation response into scores"""
        try:
            data = json.loads(response)
            scores = {metric: float(data.get(metric, 150)) for metric in metrics}
            return scores
        except:
            # Fallback: return median scores
            return {metric: 150.0 for metric in metrics}
    
    def calculate_metric_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate quadratic penalty weights for worst N metrics"""
        # Sort metrics by score (ascending)
        sorted_metrics = sorted(scores.items(), key=lambda x: x[1])
        
        weights = {}
        for i, (metric, score) in enumerate(sorted_metrics):
            if i < self.top_n_worst:
                # Quadratic penalty: lower score = higher weight
                # Normalize score to 0-1, invert, then square
                normalized_score = score / 300.0
                inverted = 1.0 - normalized_score
                weights[metric] = inverted ** 2
            else:
                # Linear weight for others
                weights[metric] = (300 - score) / 300.0
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def generate_candidates(self, current_prompt: str, evaluation: EvaluationResult,
                           iteration: int) -> List[CandidatePrompt]:
        """Generate 3 candidate prompts with different strategies"""
        weights = self.calculate_metric_weights(evaluation.scores)
        
        # Identify worst metrics
        worst_metrics = sorted(evaluation.scores.items(), key=lambda x: x[1])[:self.top_n_worst]
        
        candidates = []
        
        # Strategy 1: Aggressive - Focus heavily on worst metrics
        candidates.append(self._generate_aggressive_candidate(
            current_prompt, worst_metrics, weights, iteration
        ))
        
        # Strategy 2: Balanced - Holistic improvement
        candidates.append(self._generate_balanced_candidate(
            current_prompt, evaluation.scores, weights, iteration
        ))
        
        # Strategy 3: Structural - Format and organization
        candidates.append(self._generate_structural_candidate(
            current_prompt, evaluation.scores, iteration
        ))
        
        return candidates
    
    def _generate_aggressive_candidate(self, prompt: str, worst_metrics: List[Tuple],
                                      weights: Dict, iteration: int) -> CandidatePrompt:
        """Generate candidate focusing aggressively on worst metrics"""
        optimization_prompt = f"""You are an expert prompt engineer. Improve this prompt by AGGRESSIVELY addressing these worst-performing metrics:

{chr(10).join(f"- {metric} (score: {score}/300)" for metric, score in worst_metrics)}

CURRENT PROMPT:
'''
{prompt}
'''

Create an improved version that dramatically improves these specific weaknesses. Be bold with changes.

Respond with ONLY the improved prompt text, no explanations."""

        improved = self.api_client.optimize(optimization_prompt)
        
        return CandidatePrompt(
            prompt=improved,
            strategy="aggressive",
            generation_rationale=f"Targeted fix for: {', '.join(m[0] for m in worst_metrics[:3])}",
            parent_version=f"v{iteration}"
        )
    
    def _generate_balanced_candidate(self, prompt: str, all_scores: Dict,
                                    weights: Dict, iteration: int) -> CandidatePrompt:
        """Generate candidate with holistic, balanced improvements"""
        # Find both strengths and weaknesses
        strengths = [k for k, v in sorted(all_scores.items(), key=lambda x: -x[1])[:5]]
        weaknesses = [k for k, v in sorted(all_scores.items(), key=lambda x: x[1])[:5]]
        
        optimization_prompt = f"""You are an expert prompt engineer. Create a balanced improvement to this prompt.

CURRENT STRENGTHS (preserve these):
{chr(10).join(f"- {s}" for s in strengths)}

AREAS TO IMPROVE:
{chr(10).join(f"- {w} (score: {all_scores[w]}/300)" for w in weaknesses)}

CURRENT PROMPT:
'''
{prompt}
'''

Create an improved version that maintains strengths while addressing weaknesses holistically.

Respond with ONLY the improved prompt text, no explanations."""

        improved = self.api_client.optimize(optimization_prompt)
        
        return CandidatePrompt(
            prompt=improved,
            strategy="balanced",
            generation_rationale="Holistic improvement maintaining strengths",
            parent_version=f"v{iteration}"
        )
    
    def _generate_structural_candidate(self, prompt: str, all_scores: Dict,
                                      iteration: int) -> CandidatePrompt:
        """Generate candidate with structural/formatting improvements"""
        optimization_prompt = f"""You are an expert prompt engineer. Improve this prompt's STRUCTURE and FORMAT:

Focus on:
- Clear section organization
- Numbered/structured instructions
- Better formatting and readability
- Logical flow and hierarchy

CURRENT PROMPT:
'''
{prompt}
'''

Restructure and reformat for maximum clarity and organization. Preserve core content but improve structure.

Respond with ONLY the improved prompt text, no explanations."""

        improved = self.api_client.optimize(optimization_prompt)
        
        return CandidatePrompt(
            prompt=improved,
            strategy="structural",
            generation_rationale="Enhanced structure, formatting, and organization",
            parent_version=f"v{iteration}"
        )
    
    def select_best_candidate(self, candidates: List[CandidatePrompt],
                             evaluations: List[EvaluationResult]) -> Tuple[int, float]:
        """Select best candidate and calculate improvement"""
        best_idx = np.argmax([e.overall_score for e in evaluations])
        best_score = evaluations[best_idx].overall_score
        
        # Calculate improvement from previous iteration
        if len(self.evolution_history) > 0:
            prev_best = self.evolution_history[-1].evaluations[
                self.evolution_history[-1].best_candidate_idx
            ].overall_score
            improvement = (best_score - prev_best) / prev_best if prev_best > 0 else 0
        else:
            improvement = 0.0
        
        return best_idx, improvement
    
    def evolve(self, initial_prompt: str, run_name: str = None) -> Dict:
        """Main evolution loop"""
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        current_prompt = initial_prompt
        self.evolution_history = []
        
        print(f"🚀 Starting prompt evolution: {run_name}")
        print(f"📊 Evaluating across {sum(len(g['metrics']) for g in METRIC_GROUPS.values())} metrics")
        print(f"🔄 Max iterations: {self.max_iterations}")
        print(f"📉 Stop threshold: {self.improvement_threshold*100:.2f}%\n")
        
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Generate candidates (3 strategies)
            if iteration == 0:
                # First iteration: evaluate initial prompt
                candidates = [CandidatePrompt(
                    prompt=current_prompt,
                    strategy="initial",
                    generation_rationale="Starting prompt",
                    parent_version="v0"
                )]
            else:
                # Evaluate current best
                current_eval = self.evaluate_prompt(current_prompt, f"v{iteration}")
                print(f"\n📊 Current Score: {current_eval.overall_score:.2f}/300")
                
                # Generate new candidates
                print("🔄 Generating 3 candidate prompts...")
                candidates = self.generate_candidates(current_prompt, current_eval, iteration)
            
            # Evaluate all candidates
            print(f"⚖️  Evaluating {len(candidates)} candidates...")
            evaluations = []
            for i, candidate in enumerate(candidates):
                eval_result = self.evaluate_prompt(candidate.prompt, f"v{iteration}_{i}")
                evaluations.append(eval_result)
                print(f"  Candidate {i+1} ({candidate.strategy}): {eval_result.overall_score:.2f}/300")
            
            # Select best candidate
            best_idx, improvement = self.select_best_candidate(candidates, evaluations)
            best_candidate = candidates[best_idx]
            current_prompt = best_candidate.prompt
            
            print(f"\n✅ Best: Candidate {best_idx+1} ({best_candidate.strategy})")
            print(f"📈 Improvement: {improvement*100:.3f}%")
            
            # Store iteration results
            iteration_result = IterationResult(
                iteration=iteration + 1,
                candidates=candidates,
                evaluations=evaluations,
                best_candidate_idx=best_idx,
                improvement=improvement,
                timestamp=datetime.now().isoformat()
            )
            self.evolution_history.append(iteration_result)
            
            # Check stopping criteria
            if iteration > 0 and improvement < self.improvement_threshold:
                print(f"\n🛑 Stopping: Improvement ({improvement*100:.3f}%) below threshold ({self.improvement_threshold*100:.2f}%)")
                break
        
        # Save results
        final_results = self._compile_results(run_name, initial_prompt, current_prompt)
        self._save_results(run_name, final_results)
        
        print(f"\n{'='*60}")
        print("✨ EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Results saved to: {self.storage_dir / run_name}")
        
        return final_results
    
    def _compile_results(self, run_name: str, initial_prompt: str, 
                        final_prompt: str) -> Dict:
        """Compile all results into final output"""
        return {
            "run_name": run_name,
            "initial_prompt": initial_prompt,
            "final_prompt": final_prompt,
            "total_iterations": len(self.evolution_history),
            "evolution_history": [
                {
                    "iteration": iter_result.iteration,
                    "candidates": [asdict(c) for c in iter_result.candidates],
                    "evaluations": [asdict(e) for e in iter_result.evaluations],
                    "best_candidate_idx": iter_result.best_candidate_idx,
                    "improvement": iter_result.improvement,
                    "timestamp": iter_result.timestamp
                }
                for iter_result in self.evolution_history
            ],
            "metric_trajectories": self._compute_trajectories(),
            "summary": self._generate_summary()
        }
    
    def _compute_trajectories(self) -> Dict:
        """Compute metric score trajectories across iterations"""
        trajectories = {}
        
        for iter_result in self.evolution_history:
            best_eval = iter_result.evaluations[iter_result.best_candidate_idx]
            iteration = iter_result.iteration
            
            for metric, score in best_eval.scores.items():
                if metric not in trajectories:
                    trajectories[metric] = []
                trajectories[metric].append({
                    "iteration": iteration,
                    "score": score
                })
        
        return trajectories
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.evolution_history:
            return {}
        
        initial_score = self.evolution_history[0].evaluations[
            self.evolution_history[0].best_candidate_idx
        ].overall_score
        
        final_score = self.evolution_history[-1].evaluations[
            self.evolution_history[-1].best_candidate_idx
        ].overall_score
        
        total_improvement = (final_score - initial_score) / initial_score if initial_score > 0 else 0
        
        return {
            "initial_score": initial_score,
            "final_score": final_score,
            "total_improvement_pct": total_improvement * 100,
            "iterations_completed": len(self.evolution_history),
            "best_strategy": self._find_best_strategy()
        }
    
    def _find_best_strategy(self) -> str:
        """Find which strategy performed best"""
        strategy_wins = {"aggressive": 0, "balanced": 0, "structural": 0, "initial": 0}
        
        for iter_result in self.evolution_history:
            best_strategy = iter_result.candidates[iter_result.best_candidate_idx].strategy
            strategy_wins[best_strategy] += 1
        
        return max(strategy_wins.items(), key=lambda x: x[1])[0]
    
    def _save_results(self, run_name: str, results: Dict):
        """Save results to JSON"""
        run_dir = self.storage_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        # Save complete results
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save final prompt separately for easy access
        with open(run_dir / "final_prompt.txt", "w") as f:
            f.write(results["final_prompt"])
        
        print(f"💾 Saved results to {run_dir}")


class TemplateAPIClient:
    """
    Template API client - REPLACE THIS with your internal API integration
    """
    
    def evaluate(self, evaluation_prompt: str) -> str:
        """
        Call your internal LLM API for evaluation
        
        Args:
            evaluation_prompt: The evaluation prompt
            
        Returns:
            JSON string with scores
            
        TODO: Replace with actual API call like:
            response = your_api.call(
                prompt=evaluation_prompt,
                model="your-model",
                temperature=0.3
            )
            return response.content
        """
        # PLACEHOLDER - Replace with actual API call
        raise NotImplementedError(
            "Replace this method with your internal API integration"
        )
    
    def optimize(self, optimization_prompt: str) -> str:
        """
        Call your internal LLM API for prompt optimization
        
        Args:
            optimization_prompt: The optimization instruction
            
        Returns:
            Improved prompt text
            
        TODO: Replace with actual API call
        """
        # PLACEHOLDER - Replace with actual API call
        raise NotImplementedError(
            "Replace this method with your internal API integration"
        )


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = PromptEvolutionEngine(
        api_client=None,  # Will use TemplateAPIClient
        top_n_worst=5,
        max_iterations=6,
        improvement_threshold=0.001  # 0.10%
    )
    
    # Example prompt to optimize
    initial_prompt = """
    You are a helpful assistant. Answer the user's question clearly and concisely.
    """
    
    # Run evolution
    results = engine.evolve(
        initial_prompt=initial_prompt,
        run_name="example_run"
    )
    
    print("\n📊 Final Summary:")
    print(f"Initial Score: {results['summary']['initial_score']:.2f}")
    print(f"Final Score: {results['summary']['final_score']:.2f}")
    print(f"Total Improvement: {results['summary']['total_improvement_pct']:.2f}%")
    print(f"Best Strategy: {results['summary']['best_strategy']}")
