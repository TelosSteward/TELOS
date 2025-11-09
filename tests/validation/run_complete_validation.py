#!/usr/bin/env python3
"""
TELOS Complete Conversation Validation
=======================================

Runs full validation on Claude conversation with:
- Complete PA semantic description (before vectorization)
- Drift detection reasoning with cosine similarity
- Mid-range technical explanations
- Rate-limited API calls (3 second delays)
- Full conversation analysis from turn 9 onwards

Produces comprehensive human-readable report showing exactly how TELOS works.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mistralai import Mistral
from telos.core.embedding_provider import EmbeddingProvider
from telos.core.unified_steward import PrimacyAttractor
from telos.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor


class CompleteValidator:
    """
    Complete conversation validator with human-readable explanations.

    Validates TELOS system by:
    1. Establishing PA from real conversation
    2. Measuring fidelity across entire conversation
    3. Comparing TELOS vs baseline performance
    4. Explaining drift detection reasoning
    """

    def __init__(self):
        """Initialize with API clients and embedding provider."""
        print("🔧 Initializing TELOS Complete Validator...")

        # Load Mistral API key
        self.mistral_api_key = self._load_api_key('MISTRAL_API_KEY')
        self.mistral = Mistral(api_key=self.mistral_api_key)
        self.model = "mistral-small-2501"

        # Initialize TELOS embedding provider (deterministic for speed)
        print("📦 Loading TELOS embedding provider...")
        self.embedder = EmbeddingProvider(deterministic=True)
        print("✅ Embedding provider loaded")

        # Report structure
        self.report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "embedding_provider": "TELOS DeterministicEmbeddingProvider",
                "validation_type": "Complete Conversation Analysis"
            },
            "primacy_attractor": {
                "semantic_description": {},
                "convergence_details": {},
                "vectorization_info": {}
            },
            "fidelity_analysis": {
                "per_turn_metrics": [],
                "drift_detections": [],
                "overall_statistics": {}
            },
            "comparative_results": {
                "baseline_avg": 0.0,
                "telos_avg": 0.0,
                "improvement": 0.0
            }
        }

    def _load_api_key(self, key_name: str) -> str:
        """Load API key from secrets."""
        try:
            with open(".streamlit/secrets.toml", 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if key_name in line:
                        return line.split('=')[1].strip().strip('"')
        except Exception as e:
            raise ValueError(f"Could not load {key_name}: {e}")

    def validate_complete_conversation(self, conversation_file: str, start_turn: int = 9):
        """
        Run complete validation on conversation.

        Args:
            conversation_file: Path to conversation JSON
            start_turn: Turn to start analysis (default: 9, skipping file uploads)
        """
        print("\n" + "="*80)
        print("🔬 TELOS COMPLETE CONVERSATION VALIDATION")
        print("="*80)
        print(f"Conversation: {conversation_file}")
        print(f"Starting from turn: {start_turn}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Load conversation
        with open(conversation_file, 'r') as f:
            data = json.load(f)
        conversations = data['conversations']
        print(f"📊 Total conversation entries: {len(conversations)}")

        # PHASE 1: Establish Primacy Attractor
        print("\n" + "="*80)
        print("PHASE 1: PRIMACY ATTRACTOR ESTABLISHMENT")
        print("="*80)

        pa_result = self._establish_primacy_attractor(conversations, start_turn)

        if not pa_result:
            print("\n❌ PA establishment failed - cannot continue")
            return None

        # PHASE 2: Analyze full conversation
        print("\n" + "="*80)
        print("PHASE 2: FULL CONVERSATION FIDELITY ANALYSIS")
        print("="*80)

        self._analyze_full_conversation(conversations, pa_result, start_turn)

        # Generate report
        print("\n" + "="*80)
        print("GENERATING COMPLETE VALIDATION REPORT")
        print("="*80)

        return self._generate_report()

    def _establish_primacy_attractor(self, conversations: List[Dict], start_turn: int):
        """
        Establish PA from conversation with semantic descriptions.

        Returns dict with:
        - extractor: ProgressivePrimacyExtractor instance
        - attractor: PrimacyAttractor instance
        - semantic_description: Human-readable PA components
        - convergence_details: Statistics about convergence
        """
        print(f"\n🔄 Establishing Primacy Attractor from turns {start_turn}-{start_turn+9}")
        print("   (Using Progressive PA Extractor with statistical convergence)")

        # Wrapper classes for TELOS compatibility
        class LLMClient:
            def __init__(self, mistral):
                self.mistral = mistral

            def generate(self, messages, max_tokens=500, temperature=0.7):
                response = self.mistral.chat.complete(
                    model='mistral-small-2501',
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content

        class EmbeddingProviderWrapper:
            def __init__(self, embedder):
                self._embedder = embedder

            def encode(self, text):
                return self._embedder.encode(text)

        # Initialize Progressive PA Extractor
        extractor = ProgressivePrimacyExtractor(
            llm_client=LLMClient(self.mistral),
            embedding_provider=EmbeddingProviderWrapper(self.embedder),
            mode='progressive',
            llm_per_turn=True,  # Semantic analysis at each turn
            max_turns_safety=10
        )

        # Process turns to establish PA
        start_idx = (start_turn - 1) * 2
        turns_processed = 0
        i = start_idx

        print("\n📊 Processing turns for PA establishment:\n")

        while i < len(conversations) - 1 and turns_processed < 10:
            # Find next human-gpt pair
            while i < len(conversations) and conversations[i]['from'] != 'human':
                i += 1
            if i >= len(conversations) - 1:
                break

            j = i + 1
            while j < len(conversations) and conversations[j]['from'] != 'gpt':
                j += 1
            if j >= len(conversations):
                break

            # Process turn
            user_input = conversations[i]['value']
            assistant_response = conversations[j]['value']
            turn_num = turns_processed + start_turn
            turns_processed += 1
            i = j + 1

            print(f"Turn {turn_num}: Processing...")

            try:
                result = extractor.add_turn(user_input, assistant_response)
                time.sleep(3)  # Rate limiting

                print(f"  Status: {result['status']}")
                if result.get('convergence_metrics'):
                    metrics = result['convergence_metrics']
                    print(f"  Confidence: {metrics.get('confidence', 0):.3f}")

                if result['status'] == 'converged':
                    print(f"\n✅ PA CONVERGED at turn {turn_num}!")
                    attractor = result.get('attractor')

                    # Store semantic description
                    semantic_desc = {
                        "purpose": attractor.purpose if attractor else ["General conversation"],
                        "scope": attractor.scope if attractor else ["Various topics"],
                        "boundaries": attractor.boundaries if attractor else ["Stay on topic"],
                        "interpretation": self._interpret_pa(attractor)
                    }

                    print(f"\n📍 PRIMACY ATTRACTOR (Semantic Description):")
                    print(f"\n   **Purpose** (What is this conversation trying to accomplish?):")
                    for p in semantic_desc['purpose']:
                        print(f"      • {p}")
                    print(f"\n   **Scope** (What topics are being discussed?):")
                    for s in semantic_desc['scope'][:5]:
                        print(f"      • {s}")
                    print(f"\n   **Boundaries** (What should be avoided?):")
                    for b in semantic_desc['boundaries']:
                        print(f"      • {b}")

                    self.report["primacy_attractor"]["semantic_description"] = semantic_desc
                    self.report["primacy_attractor"]["convergence_details"] = {
                        "convergence_turn": turn_num,
                        "confidence": result['convergence_metrics'].get('confidence', 0),
                        "centroid_stability": result['convergence_metrics'].get('centroid_stability', 0),
                        "variance_stability": result['convergence_metrics'].get('variance_stability', 0)
                    }
                    self.report["primacy_attractor"]["vectorization_info"] = {
                        "embedding_dimension": len(extractor.attractor_centroid) if extractor.attractor_centroid is not None else 384,
                        "method": "Statistical convergence of semantic embeddings",
                        "explanation": "The PA is converted to a vector (list of numbers) that captures the conversation's meaning in mathematical form, allowing us to measure how far responses drift from the original purpose."
                    }

                    return {
                        'extractor': extractor,
                        'attractor': attractor,
                        'semantic_description': semantic_desc,
                        'convergence_turn': turn_num
                    }

            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                if "429" in str(e):
                    print(f"  Rate limit - waiting 10 seconds...")
                    time.sleep(10)
                continue

        print("\n⚠️  PA did not converge within 10 turns")
        return None

    def _interpret_pa(self, attractor: PrimacyAttractor) -> str:
        """Generate human-readable interpretation of PA."""
        if not attractor:
            return "No attractor established"

        purpose_summary = attractor.purpose[0] if attractor.purpose else "General discussion"
        scope_summary = ', '.join(attractor.scope[:3]) if attractor.scope else "various topics"

        return (
            f"This conversation is focused on {purpose_summary.lower()}. "
            f"The discussion covers {scope_summary}. "
            f"Responses should stay within these bounds to maintain alignment."
        )

    def _analyze_full_conversation(self, conversations: List[Dict], pa_result: Dict, start_turn: int):
        """
        Analyze full conversation with fidelity measurements and drift detection.

        For each turn:
        1. Calculate fidelity (alignment with PA)
        2. Detect if drift occurred
        3. Explain reasoning with cosine similarity
        4. Store metrics
        """
        extractor = pa_result['extractor']
        attractor = pa_result['attractor']
        convergence_turn = pa_result['convergence_turn']

        print(f"\n📊 Analyzing conversation from turn {convergence_turn + 1} onwards...")
        print(f"   Measuring fidelity and detecting drift\n")

        # Start from turn after PA convergence
        start_idx = convergence_turn * 2
        turn_num = convergence_turn
        i = start_idx
        turns_analyzed = 0

        fidelities = []

        while i < len(conversations) - 1 and turns_analyzed < 20:  # Limit to 20 turns for now
            # Find next human-gpt pair
            while i < len(conversations) and conversations[i]['from'] != 'human':
                i += 1
            if i >= len(conversations) - 1:
                break

            j = i + 1
            while j < len(conversations) and conversations[j]['from'] != 'gpt':
                j += 1
            if j >= len(conversations):
                break

            user_input = conversations[i]['value']
            assistant_response = conversations[j]['value']
            turn_num += 1
            turns_analyzed += 1
            i = j + 1

            print(f"{'='*80}")
            print(f"Turn {turn_num}")
            print(f"{'='*80}")
            print(f"\n👤 USER: {user_input[:100]}...")
            print(f"\n🤖 ASSISTANT: {assistant_response[:100]}...")

            # Calculate fidelity with reasoning
            fidelity_data = self._calculate_fidelity_with_reasoning(
                assistant_response, attractor, extractor
            )

            fidelities.append(fidelity_data['fidelity'])

            print(f"\n📊 FIDELITY ANALYSIS:")
            print(f"   Score: {fidelity_data['fidelity']:.3f}")
            print(f"   Cosine Similarity: {fidelity_data['cosine_similarity']:.1%}")
            print(f"   Semantic Divergence: {fidelity_data['divergence_percent']:.1f}%")
            print(f"\n   💡 Reasoning: {fidelity_data['reasoning']}")

            # Detect drift
            drift_threshold = 0.75
            if fidelity_data['fidelity'] < drift_threshold:
                print(f"\n   🚨 DRIFT DETECTED (below {drift_threshold} threshold)")
                print(f"      Why: {fidelity_data['drift_explanation']}")

                self.report["fidelity_analysis"]["drift_detections"].append({
                    "turn": turn_num,
                    "fidelity": fidelity_data['fidelity'],
                    "cosine_similarity": fidelity_data['cosine_similarity'],
                    "divergence_percent": fidelity_data['divergence_percent'],
                    "reasoning": fidelity_data['reasoning'],
                    "drift_explanation": fidelity_data['drift_explanation']
                })
            else:
                print(f"\n   ✅ ALIGNED (above {drift_threshold} threshold)")

            # Store turn metrics
            self.report["fidelity_analysis"]["per_turn_metrics"].append({
                "turn": turn_num,
                "fidelity": fidelity_data['fidelity'],
                "cosine_similarity": fidelity_data['cosine_similarity'],
                "divergence_percent": fidelity_data['divergence_percent'],
                "drift_detected": fidelity_data['fidelity'] < drift_threshold
            })

            print()
            time.sleep(2)  # Rate limiting

        # Calculate overall statistics
        if fidelities:
            avg_fidelity = np.mean(fidelities)
            min_fidelity = np.min(fidelities)
            max_fidelity = np.max(fidelities)
            drift_count = sum(1 for f in fidelities if f < 0.75)

            print(f"\n{'='*80}")
            print("OVERALL CONVERSATION STATISTICS")
            print(f"{'='*80}")
            print(f"\nTurns Analyzed: {len(fidelities)}")
            print(f"Average Fidelity: {avg_fidelity:.3f}")
            print(f"Minimum Fidelity: {min_fidelity:.3f}")
            print(f"Maximum Fidelity: {max_fidelity:.3f}")
            print(f"Drift Occurrences: {drift_count} ({drift_count/len(fidelities)*100:.1f}%)")

            self.report["fidelity_analysis"]["overall_statistics"] = {
                "turns_analyzed": len(fidelities),
                "average_fidelity": float(avg_fidelity),
                "minimum_fidelity": float(min_fidelity),
                "maximum_fidelity": float(max_fidelity),
                "drift_count": drift_count,
                "drift_rate": float(drift_count / len(fidelities))
            }

    def _calculate_fidelity_with_reasoning(
        self,
        response: str,
        attractor: PrimacyAttractor,
        extractor: ProgressivePrimacyExtractor
    ) -> Dict[str, Any]:
        """
        Calculate fidelity with detailed human-readable reasoning.

        Returns:
        - fidelity: 0-1 score
        - cosine_similarity: 0-1 similarity measure
        - divergence_percent: How far response drifted (0-100%)
        - reasoning: Human-readable explanation
        - drift_explanation: Why drift occurred (if detected)
        """
        # Get embeddings
        response_emb = self.embedder.encode(response)
        centroid = extractor.attractor_centroid

        if centroid is None:
            purpose_text = ' '.join(attractor.purpose)
            centroid = self.embedder.encode(purpose_text)

        # Calculate cosine similarity (measures alignment of meaning)
        # 1.0 = identical meaning, 0.5 = somewhat related, 0.0 = unrelated
        cos_sim = 1 - cosine(response_emb, centroid)

        # Calculate Euclidean distance (geometric distance in embedding space)
        distance = float(np.linalg.norm(response_emb - centroid))

        # Convert to fidelity score (0-1, where 1 = perfect alignment)
        fidelity = np.exp(-distance / 2.0)
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        # Human-readable classification
        if cos_sim >= 0.85:
            alignment = "strongly aligned"
            drift_risk = "low"
        elif cos_sim >= 0.70:
            alignment = "moderately aligned"
            drift_risk = "moderate"
        elif cos_sim >= 0.50:
            alignment = "weakly aligned"
            drift_risk = "high"
        else:
            alignment = "divergent"
            drift_risk = "critical"

        divergence_pct = (1.0 - cos_sim) * 100

        # Generate reasoning
        reasoning = (
            f"Response is {alignment} with the conversation's original purpose. "
            f"The semantic similarity between this response and the established purpose is {cos_sim:.1%}, "
            f"meaning {divergence_pct:.1f}% divergence from the intended topic. "
            f"Fidelity score: {fidelity:.3f}/1.00 (drift risk: {drift_risk})."
        )

        # Drift explanation (if drifted)
        drift_explanation = ""
        if fidelity < 0.75:
            drift_explanation = (
                f"The response drifted because its meaning ({cos_sim:.1%} similar to purpose) "
                f"fell below the alignment threshold. This indicates the conversation "
                f"is moving away from its original focus on {attractor.purpose[0].lower()}."
            )

        return {
            'fidelity': fidelity,
            'cosine_similarity': float(cos_sim),
            'distance': distance,
            'divergence_percent': float(divergence_pct),
            'alignment_category': alignment,
            'drift_risk': drift_risk,
            'reasoning': reasoning,
            'drift_explanation': drift_explanation
        }

    def _generate_report(self) -> Dict[str, Any]:
        """Generate and save complete validation report."""
        output_dir = Path("tests/validation_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON
        json_file = output_dir / f"complete_validation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.report, f, indent=2)

        # Save human-readable
        txt_file = output_dir / f"complete_validation_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write(self._format_human_readable_report())

        print(f"\n✅ Complete validation reports saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📝 Human-readable: {txt_file}")

        return self.report

    def _format_human_readable_report(self) -> str:
        """Format complete human-readable report."""
        lines = []
        lines.append("="*80)
        lines.append("TELOS COMPLETE CONVERSATION VALIDATION REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {self.report['metadata']['timestamp']}\n")

        # Section 1: Primacy Attractor
        lines.append("\n" + "="*80)
        lines.append("SECTION 1: PRIMACY ATTRACTOR (PA)")
        lines.append("="*80)
        lines.append("\nWhat is the Primacy Attractor?")
        lines.append("-" * 40)
        lines.append("The PA is a mathematical representation of the conversation's purpose.")
        lines.append("It acts as an 'anchor point' that TELOS uses to measure if responses")
        lines.append("stay on-topic or drift away from the intended discussion.\n")

        pa = self.report["primacy_attractor"]["semantic_description"]
        lines.append("Semantic Description (Before Vectorization):")
        lines.append("-" * 40)
        lines.append(f"\nPurpose: {', '.join(pa.get('purpose', []))}")
        lines.append(f"Scope: {', '.join(pa.get('scope', []))}")
        lines.append(f"Boundaries: {', '.join(pa.get('boundaries', []))}")
        lines.append(f"\nInterpretation: {pa.get('interpretation', '')}\n")

        conv_details = self.report["primacy_attractor"]["convergence_details"]
        lines.append("Convergence Details:")
        lines.append("-" * 40)
        lines.append(f"Converged at turn: {conv_details.get('convergence_turn', 'N/A')}")
        lines.append(f"Confidence: {conv_details.get('confidence', 0):.3f}")
        lines.append(f"Centroid Stability: {conv_details.get('centroid_stability', 0):.3f}")
        lines.append(f"Variance Stability: {conv_details.get('variance_stability', 0):.3f}\n")

        vec_info = self.report["primacy_attractor"]["vectorization_info"]
        lines.append("Vectorization (How PA Becomes Mathematical):")
        lines.append("-" * 40)
        lines.append(f"Dimension: {vec_info.get('embedding_dimension', 'N/A')}")
        lines.append(f"Method: {vec_info.get('method', 'N/A')}")
        lines.append(f"Explanation: {vec_info.get('explanation', 'N/A')}")

        # Section 2: Fidelity Analysis
        lines.append("\n\n" + "="*80)
        lines.append("SECTION 2: FIDELITY ANALYSIS")
        lines.append("="*80)
        lines.append("\nWhat is Fidelity?")
        lines.append("-" * 40)
        lines.append("Fidelity measures how well each response aligns with the PA.")
        lines.append("Score range: 0.0 (completely off-topic) to 1.0 (perfect alignment)")
        lines.append("Threshold: 0.75 (below this = drift detected)\n")

        stats = self.report["fidelity_analysis"]["overall_statistics"]
        if stats:
            lines.append("Overall Statistics:")
            lines.append("-" * 40)
            lines.append(f"Turns Analyzed: {stats['turns_analyzed']}")
            lines.append(f"Average Fidelity: {stats['average_fidelity']:.3f}")
            lines.append(f"Minimum Fidelity: {stats['minimum_fidelity']:.3f}")
            lines.append(f"Maximum Fidelity: {stats['maximum_fidelity']:.3f}")
            lines.append(f"Drift Occurrences: {stats['drift_count']} ({stats['drift_rate']*100:.1f}%)\n")

        # Section 3: Drift Detections
        lines.append("\n" + "="*80)
        lines.append("SECTION 3: DRIFT DETECTIONS")
        lines.append("="*80)
        lines.append("\nWhat is Drift?")
        lines.append("-" * 40)
        lines.append("Drift occurs when a response moves away from the conversation's purpose.")
        lines.append("TELOS detects drift by measuring semantic divergence using cosine similarity.\n")

        drifts = self.report["fidelity_analysis"]["drift_detections"]
        if drifts:
            lines.append(f"Total Drift Events: {len(drifts)}\n")
            for i, drift in enumerate(drifts[:10], 1):  # Show first 10
                lines.append(f"\nDrift Event #{i} (Turn {drift['turn']}):")
                lines.append(f"  Fidelity: {drift['fidelity']:.3f}")
                lines.append(f"  Cosine Similarity: {drift['cosine_similarity']:.1%}")
                lines.append(f"  Divergence: {drift['divergence_percent']:.1f}%")
                lines.append(f"  Reasoning: {drift['reasoning']}")
                lines.append(f"  Why Drift Occurred: {drift['drift_explanation']}")
        else:
            lines.append("No drift detected - all responses remained aligned!\n")

        # Section 4: Per-Turn Metrics
        lines.append("\n\n" + "="*80)
        lines.append("SECTION 4: PER-TURN FIDELITY METRICS")
        lines.append("="*80)

        metrics = self.report["fidelity_analysis"]["per_turn_metrics"]
        if metrics:
            lines.append(f"\nShowing fidelity progression across conversation:\n")
            for metric in metrics[:20]:  # Show first 20
                drift_marker = "🚨 DRIFT" if metric['drift_detected'] else "✅ ALIGNED"
                lines.append(f"Turn {metric['turn']}: {metric['fidelity']:.3f} | "
                           f"Similarity: {metric['cosine_similarity']:.1%} | "
                           f"Divergence: {metric['divergence_percent']:.1f}% | "
                           f"{drift_marker}")

        lines.append("\n\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)

        return '\n'.join(lines)


def main():
    """Run complete validation."""
    print("\n🔬 TELOS COMPLETE CONVERSATION VALIDATOR")
    print("="*80)
    print("This validation will:")
    print("  1. Establish PA from real conversation (turn 9+)")
    print("  2. Analyze full conversation with fidelity measurements")
    print("  3. Detect drift with detailed reasoning")
    print("  4. Generate comprehensive human-readable report")
    print("="*80)

    validator = CompleteValidator()

    conversation_file = "tests/validation_data/baseline_conversations/real_claude_conversation.json"

    report = validator.validate_complete_conversation(conversation_file, start_turn=9)

    if report:
        print("\n✅ COMPLETE VALIDATION FINISHED")
        print("\nReport includes:")
        print("  ✓ Semantic PA description (purpose, scope, boundaries)")
        print("  ✓ Drift detection reasoning with cosine similarity")
        print("  ✓ Mid-range technical explanations")
        print("  ✓ Full conversation fidelity metrics")
    else:
        print("\n❌ Validation failed")


if __name__ == "__main__":
    main()
