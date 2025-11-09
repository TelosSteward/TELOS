#!/usr/bin/env python3
"""
TELOS Forensic Validation Runner
=================================

Runs COMPLETE forensic validation with human-readable reporting showing:
1. Progressive PA Extractor establishment timeline (first 10 turns)
2. Counterfactual branch contamination prevention
3. Every intervention branch point with actual API calls
4. Conversational DNA of both baseline and TELOS branches
5. Comparative metrics at every decision point

This produces the FULL forensic report for grant applications and research validation.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import is_dataclass, asdict
import numpy as np
from scipy.spatial.distance import cosine

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mistralai import Mistral
from telos.core.embedding_provider import EmbeddingProvider
from telos.core.unified_steward import PrimacyAttractor, UnifiedGovernanceSteward


def convert_bools_to_strings(obj):
    """Recursively convert all booleans and numpy types in a data structure to JSON-safe types."""
    # Check bool FIRST before other types (bool is a subclass of int!)
    # Use type() to avoid subclass issues
    # Also check for numpy bool types
    if type(obj) is bool or isinstance(obj, (np.bool_, np.bool)):
        return "yes" if obj else "no"
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_bools_to_strings(item) for item in obj.tolist()]
    elif is_dataclass(obj):
        # Convert dataclass to dict first, then recursively process
        return convert_bools_to_strings(asdict(obj))
    elif isinstance(obj, dict):
        return {k: convert_bools_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_bools_to_strings(item) for item in obj]
    else:
        return obj
from telos.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor


class ForensicValidator:
    """
    Complete forensic validation with granular human-readable reporting.

    Captures:
    - PA establishment timeline (turn-by-turn convergence)
    - Counterfactual branching (isolated baseline vs TELOS paths)
    - Intervention decisions (every API call, every choice point)
    - Conversational DNA (actual text at every branch)
    - Comparative metrics (fidelity scores throughout)
    """

    def __init__(self):
        """Initialize with local SentenceTransformer embeddings and Mistral LLM."""
        # Load API key for LLM only
        self.mistral_api_key = self._load_api_key('MISTRAL_API_KEY')
        self.mistral = Mistral(api_key=self.mistral_api_key)
        self.model = "mistral-small-2501"  # For LLM calls only

        # Initialize local SentenceTransformer embeddings (no API calls!)
        print("📦 Loading SentenceTransformer embeddings (local, no API)...")
        from telos.core.embedding_provider import EmbeddingProvider
        self.embedder = EmbeddingProvider(deterministic=False)  # Real semantic embeddings, runs locally
        print(f"✅ SentenceTransformer loaded: {self.embedder.model_name}")
        print(f"   Dimension: {self.embedder.dimension}")

        # Test Mistral LLM API connection (still needed for PA extraction)
        print("\n🧪 Testing Mistral LLM API connection...")
        try:
            test_response = self.mistral.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": "respond with: test successful"}],
                max_tokens=10
            )
            print(f"✅ Mistral LLM working: '{test_response.choices[0].message.content}'")
        except Exception as e:
            print(f"❌ Mistral LLM test FAILED: {e}")
            print("   Check API key and model access in .streamlit/secrets.toml")
            raise
        print()

        # Forensic report structure
        self.report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "embedding_provider": "TELOS DeterministicEmbeddingProvider"
            },
            "pa_establishment": {
                "turn_by_turn_convergence": [],
                "convergence_summary": {}
            },
            "counterfactual_branches": [],
            "intervention_analysis": {
                "decision_points": [],
                "api_call_log": []
            },
            "conversational_dna": {
                "baseline_branches": [],
                "telos_branches": []
            },
            "comparative_metrics": {
                "per_turn_fidelity": [],
                "branch_comparisons": []
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

    def _create_mistral_embedder(self):
        """Create wrapper for Mistral embeddings API with rate limiting and caching."""
        import time

        class MistralEmbedder:
            def __init__(self, mistral_client, model_name):
                self.mistral = mistral_client
                self.model = model_name
                self.expected_dim = 1024  # mistral-embed dimension
                self.cache = {}  # Cache embeddings by text hash
                self.last_call_time = 0
                self.min_delay = 5.0  # Minimum 5 seconds between calls

            def encode(self, text):
                """Encode text using Mistral embeddings API with rate limiting and caching."""
                if isinstance(text, list):
                    text = text[0] if text else ""

                # Check cache first
                cache_key = hash(text)
                if cache_key in self.cache:
                    return self.cache[cache_key]

                # Rate limiting - wait if needed
                time_since_last_call = time.time() - self.last_call_time
                if time_since_last_call < self.min_delay:
                    sleep_time = self.min_delay - time_since_last_call
                    time.sleep(sleep_time)

                # Try with exponential backoff
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.mistral.embeddings.create(
                            model=self.model,
                            inputs=[text]
                        )
                        embedding = np.array(response.data[0].embedding)
                        self.last_call_time = time.time()

                        # Cache the result
                        self.cache[cache_key] = embedding
                        return embedding

                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "capacity exceeded" in error_str.lower():
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                                print(f"⚠️  Rate limited, waiting {wait_time}s before retry {attempt+2}/{max_retries}")
                                time.sleep(wait_time)
                            else:
                                print(f"❌ Rate limit retry exhausted: {e}")
                                return np.zeros(self.expected_dim)
                        else:
                            print(f"⚠️  Mistral embeddings API error: {e}")
                            return np.zeros(self.expected_dim)

                return np.zeros(self.expected_dim)

        return MistralEmbedder(self.mistral, self.embedding_model)

    def run_forensic_validation(self, conversation_file: str, start_turn: int = 1):
        """
        Run complete forensic validation.

        Args:
            conversation_file: Path to conversation JSON
            start_turn: Turn number to start PA extraction (default: 1)
                       Useful for skipping file uploads or preamble
        """
        print("\n" + "="*80)
        print("🔬 TELOS FORENSIC VALIDATION")
        print("="*80)
        print(f"Target: {conversation_file}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Load conversation
        with open(conversation_file, 'r') as f:
            data = json.load(f)

        conversations = data['conversations']
        print(f"📊 Total conversation turns: {len(conversations) // 2}")

        if start_turn > 1:
            print(f"⏩ Starting PA extraction from turn {start_turn} (skipping preamble)")
            print(f"   Reason: First {start_turn-1} turns are file uploads/setup, not substantive conversation")

        # PHASE 1: Progressive PA Extraction (First ~10-20 turns for convergence)
        print("\n" + "="*80)
        print("PHASE 1: PROGRESSIVE PRIMACY ATTRACTOR ESTABLISHMENT")
        print(f"Starting from turn {start_turn}, will converge within ~10-20 turns")
        print("="*80)

        pa_extractor = self._run_pa_establishment(conversations, start_turn)

        if pa_extractor is None:
            print("❌ PA extraction failed - cannot continue")
            return None

        # PHASE 2: Run counterfactual validation
        print("\n" + "="*80)
        print("PHASE 2: COUNTERFACTUAL VALIDATION")
        print("="*80)

        self._run_counterfactual_validation(conversations, pa_extractor)

        # Generate final report
        print("\n" + "="*80)
        print("GENERATING FORENSIC REPORT")
        print("="*80)

        return self._generate_final_report()

    def _run_pa_establishment(self, conversations: List[Dict], start_turn: int = 1) -> ProgressivePrimacyExtractor:
        """
        Phase 1: Run Progressive PA Extractor and log every turn.

        Returns:
            ProgressivePrimacyExtractor instance or None if failed
        """
        # Wrapper classes for API compatibility
        class LLMClient:
            def __init__(self, mistral_client, model_name):
                self.mistral = mistral_client
                self.model = model_name

            def generate(self, messages, max_tokens=500, temperature=0.7):
                print(f"\n🔧 DEBUG: LLMClient.generate() called")
                print(f"   Model: {self.model}")
                print(f"   Messages: {len(messages)} messages")
                print(f"   Max tokens: {max_tokens}, Temperature: {temperature}")

                try:
                    response = self.mistral.chat.complete(
                        model=self.model,  # Use the actual model we configured
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    result = response.choices[0].message.content
                    print(f"✅ DEBUG: Mistral API call successful, response length: {len(result)} chars")
                    return result
                except Exception as e:
                    print(f"❌ DEBUG: Mistral API call FAILED: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise so caller can handle it

        class EmbeddingProvider:
            def __init__(self, embedder):
                self.embedder = embedder

            def encode(self, text):
                if isinstance(text, list):
                    return self.embedder.encode(text[0])
                return self.embedder.encode(text)

        # Initialize Progressive PA Extractor with STATISTICAL convergence mode
        # LLM called ONCE at convergence, not per-turn
        extractor = ProgressivePrimacyExtractor(
            llm_client=LLMClient(self.mistral, self.model),  # Pass the working model
            embedding_provider=EmbeddingProvider(self.embedder),
            mode='progressive',
            llm_per_turn=False,  # Statistical convergence first, then ONE LLM call
            max_turns_safety=20,  # PA should converge within ~10-20 turns
            consecutive_stable_turns=2,  # Require 2 stable turns before convergence
            confidence_threshold=0.80  # 80% confidence needed
        )

        print("\n🔄 Progressive PA Extraction (Statistical Convergence → Single LLM Call)")
        print("-" * 80)

        # Process turns until PA converges (or safety limit reached)
        pa_established = False
        turn_num = start_turn - 1
        start_idx = (start_turn - 1) * 2

        print(f"DEBUG: Starting from conversation index {start_idx}")
        print(f"DEBUG: Total conversations: {len(conversations)}")

        turns_processed = 0

        # Find next user-assistant pair starting from start_idx
        i = start_idx
        while i < len(conversations) - 1:  # Will stop when PA converges
            # Find next human message
            while i < len(conversations) and conversations[i]['from'] != 'human':
                i += 1

            if i >= len(conversations) - 1:
                break

            # Find next gpt message after this human message
            j = i + 1
            while j < len(conversations) and conversations[j]['from'] != 'gpt':
                j += 1

            if j >= len(conversations):
                break

            # Found a valid pair!
            user_turn = conversations[i]
            assistant_turn = conversations[j]
            turn_num = turns_processed + start_turn
            turns_processed += 1

            # Move to next position
            i = j + 1

            user_input = user_turn['value']
            assistant_response = assistant_turn['value']

            print(f"\n{'='*80}")
            print(f"TURN {turn_num}")
            print(f"{'='*80}")
            print(f"\n👤 USER:")
            print(f"{user_input[:200]}..." if len(user_input) > 200 else user_input)
            print(f"\n🤖 ASSISTANT:")
            print(f"{assistant_response[:200]}..." if len(assistant_response) > 200 else assistant_response)

            # Process turn through Progressive PA Extractor with retry logic
            max_retries = 10
            retry_delay = 20  # Start with 20 seconds
            retry_count = 0
            result = None

            while retry_count < max_retries:
                try:
                    result = extractor.add_turn(user_input, assistant_response)
                    # Success! Add small delay to avoid rate limiting
                    time.sleep(3)
                    break  # Exit retry loop on success

                except Exception as e:
                    if "429" in str(e):
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"\n⚠️  Rate limit hit (attempt {retry_count}/{max_retries})")
                            print(f"    Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            # Exponential backoff, but cap at 60 seconds
                            retry_delay = min(retry_delay * 1.5, 60)
                            continue
                        else:
                            print(f"\n❌ Max retries reached after {max_retries} attempts")
                            result = {
                                'status': 'error',
                                'message': f'Rate limit error after {max_retries} retries',
                                'baseline_established': False,
                                'turn_count': turn_num
                            }
                            break
                    else:
                        # Non-rate-limit error - log and continue
                        print(f"\n⚠️  Error processing turn: {e}")
                        result = {
                            'status': 'error',
                            'message': f'Error: {e}',
                            'baseline_established': False,
                            'turn_count': turn_num
                        }
                        break

            if result is None:
                # Should never happen, but safety check
                continue

            # Log API call
            self.report["intervention_analysis"]["api_call_log"].append({
                "turn": turn_num,
                "operation": "progressive_pa_extraction",
                "llm_called": "yes" if extractor.llm_per_turn else "no",
                "status": result['status']
            })

            # Log convergence progress
            pa_turn_data = {
                "turn": turn_num,
                "status": result['status'],
                "message": result['message'],
                "baseline_established": "yes" if result.get('baseline_established', False) else "no",
                "convergence_metrics": result.get('convergence_metrics', {})
            }

            self.report["pa_establishment"]["turn_by_turn_convergence"].append(pa_turn_data)

            print(f"\n📊 PA EXTRACTOR STATUS: {result['status']}")
            print(f"    {result['message']}")

            if result.get('convergence_metrics'):
                metrics = result['convergence_metrics']
                print(f"    Confidence: {metrics.get('confidence', 0):.3f}")
                print(f"    Centroid Stability: {metrics.get('centroid_stability', 0):.3f}")
                print(f"    Variance Stability: {metrics.get('variance_stability', 0):.3f}")

            # Check if converged
            if result['status'] == 'converged':
                pa_established = True
                convergence_turn = turn_num
                attractor = result.get('attractor')

                print(f"\n✅ PA CONVERGED AT TURN {convergence_turn}!")
                print(f"\n📍 PRIMACY ATTRACTOR ESTABLISHED:")
                if attractor:
                    print(f"   Purpose: {', '.join(attractor.purpose[:2])}")
                    print(f"   Scope: {', '.join(attractor.scope[:3])}")
                    print(f"   Boundaries: {', '.join(attractor.boundaries[:2])}")

                # Store summary
                self.report["pa_establishment"]["convergence_summary"] = {
                    "converged": "yes",
                    "convergence_turn": convergence_turn,
                    "total_turns_analyzed": turn_num,
                    "attractor": {
                        "purpose": attractor.purpose if attractor else [],
                        "scope": attractor.scope if attractor else [],
                        "boundaries": attractor.boundaries if attractor else []
                    },
                    "convergence_metrics": result.get('convergence_metrics', {}),
                    "llm_analyses": result.get('llm_analyses', [])
                }

                break

        if not pa_established:
            print(f"\n⚠️ PA did not converge within 10 turns")
            print(f"DEBUG: Turns processed: {turns_processed}")
            print(f"DEBUG: Final turn_num: {turn_num}")
            self.report["pa_establishment"]["convergence_summary"] = {
                "converged": "no",
                "turns_analyzed": turn_num,
                "turns_processed": turns_processed,
                "reason": "max_turns_reached"
            }
            return None

        return extractor

    def _run_counterfactual_validation(self, conversations: List[Dict], pa_extractor: ProgressivePrimacyExtractor):
        """
        Phase 2: Run counterfactual validation showing baseline vs TELOS branches.

        For each intervention point:
        1. Fork pristine state
        2. Generate baseline branch (no intervention)
        3. Generate TELOS branch (with intervention)
        4. Compare metrics
        """
        print("\n🔀 Counterfactual Branch Analysis")
        print("-" * 80)

        attractor = pa_extractor.get_attractor()
        print(f"🔧 DEBUG: Attractor available: {attractor is not None}")
        if not attractor:
            print("⚠️ No attractor available - skipping counterfactual analysis")
            return

        # Get convergence turn
        convergence_turn = pa_extractor.get_convergence_turn()
        print(f"🔧 DEBUG: Convergence turn: {convergence_turn}")

        # Find where PA establishment ended - need to search for actual conversation index
        # The PA extractor processed turns, but conversations don't strictly alternate
        # So we need to find the conversation index where turn 10 ended
        i = 0
        turns_seen = 0
        while i < len(conversations) - 1 and turns_seen < convergence_turn:
            if conversations[i]['from'] == 'human':
                # Find matching assistant response
                j = i + 1
                while j < len(conversations) and conversations[j]['from'] != 'gpt':
                    j += 1
                if j < len(conversations):
                    turns_seen += 1
                    i = j + 1  # Move past this turn
                else:
                    break
            else:
                i += 1

        start_idx = i
        print(f"🔧 DEBUG: After {convergence_turn} turns, starting counterfactual at index {start_idx}")

        print(f"\n📊 Starting counterfactual analysis from turn {convergence_turn + 1}")
        remaining_conversations = len(conversations) - start_idx
        print(f"    Remaining conversation entries: {remaining_conversations}")

        # Initialize steward for TELOS path
        steward = self._create_steward(attractor)

        # Track conversation history for both paths
        baseline_history = []
        telos_history = []

        # Process ALL remaining turns after PA establishment
        turn_num = convergence_turn
        turns_processed = 0

        i = start_idx
        while i < len(conversations) - 1:  # Process all remaining turns
            # Find next human message
            while i < len(conversations) and conversations[i]['from'] != 'human':
                i += 1

            if i >= len(conversations):
                break

            # Find matching assistant response
            j = i + 1
            while j < len(conversations) and conversations[j]['from'] != 'gpt':
                j += 1

            if j >= len(conversations):
                break

            turn_num += 1
            turns_processed += 1
            user_turn = conversations[i]
            assistant_turn = conversations[j]

            user_input = user_turn['value']
            original_response = assistant_turn['value']

            print(f"\n{'='*80}")
            print(f"COUNTERFACTUAL BRANCH - TURN {turn_num}")
            print(f"{'='*80}")
            print(f"\n👤 USER INPUT:")
            print(f"{user_input[:150]}...")

            # FORK POINT: Create two independent branches
            print(f"\n🔀 FORK POINT: Creating baseline and TELOS branches...")

            # === BASELINE BRANCH (No Intervention) ===
            print(f"\n📊 BASELINE BRANCH (No Governance):")
            baseline_history_copy = baseline_history.copy()
            baseline_history_copy.append({"role": "user", "content": user_input})

            baseline_response = self._generate_response(baseline_history_copy, "BASELINE")
            baseline_history.append({"role": "user", "content": user_input})
            baseline_history.append({"role": "assistant", "content": baseline_response})

            baseline_fidelity_result = self._calculate_fidelity(baseline_response, attractor, pa_extractor)
            baseline_fidelity = baseline_fidelity_result['fidelity']

            print(f"   Response: {baseline_response[:150]}...")
            print(f"   Fidelity: {baseline_fidelity:.3f}")

            # === TELOS BRANCH (With Intervention) ===
            print(f"\n🔧 TELOS BRANCH (Full Governance):")
            telos_history_copy = telos_history.copy()
            telos_history_copy.append({"role": "user", "content": user_input})

            telos_response = self._generate_response(telos_history_copy, "TELOS")

            # Check if intervention needed
            telos_fidelity_result = self._calculate_fidelity(telos_response, attractor, pa_extractor)
            telos_fidelity = telos_fidelity_result['fidelity']
            intervention_needed = telos_fidelity < 0.75

            if intervention_needed:
                print(f"   🚨 DRIFT DETECTED! Fidelity: {telos_fidelity:.3f}")
                print(f"   🔧 APPLYING INTERVENTION...")

                # Apply intervention
                telos_response_corrected = self._apply_intervention(
                    telos_response, attractor, telos_history_copy
                )
                telos_final_fidelity_result = self._calculate_fidelity(telos_response_corrected, attractor, pa_extractor)
                telos_final_fidelity = telos_final_fidelity_result['fidelity']

                print(f"   Response (corrected): {telos_response_corrected[:150]}...")
                print(f"   Fidelity (post-intervention): {telos_final_fidelity:.3f}")
                print(f"   Improvement: {telos_final_fidelity - telos_fidelity:+.3f}")

                telos_history.append({"role": "user", "content": user_input})
                telos_history.append({"role": "assistant", "content": telos_response_corrected})

                # Log intervention decision
                self.report["intervention_analysis"]["decision_points"].append({
                    "turn": turn_num,
                    "drift_detected": "yes",
                    "pre_intervention_fidelity": telos_fidelity,
                    "post_intervention_fidelity": telos_final_fidelity,
                    "improvement": telos_final_fidelity - telos_fidelity,
                    "intervention_type": "boundary_correction"
                })

            else:
                print(f"   ✅ NO INTERVENTION NEEDED - Fidelity: {telos_fidelity:.3f}")
                telos_final_fidelity = telos_fidelity
                telos_history.append({"role": "user", "content": user_input})
                telos_history.append({"role": "assistant", "content": telos_response})

                self.report["intervention_analysis"]["decision_points"].append({
                    "turn": turn_num,
                    "drift_detected": "no",
                    "fidelity": telos_fidelity,
                    "intervention_type": None
                })

            # Store conversational DNA
            self.report["conversational_dna"]["baseline_branches"].append({
                "turn": turn_num,
                "user_input": user_input,
                "assistant_response": baseline_response,
                "fidelity": baseline_fidelity
            })

            self.report["conversational_dna"]["telos_branches"].append({
                "turn": turn_num,
                "user_input": user_input,
                "assistant_response": telos_response if not intervention_needed else telos_response_corrected,
                "fidelity": telos_final_fidelity,
                "intervention_applied": "yes" if intervention_needed else "no"
            })

            # Comparative metrics
            delta_f = telos_final_fidelity - baseline_fidelity
            print(f"\n📊 COMPARATIVE ANALYSIS:")
            print(f"   Baseline Fidelity: {baseline_fidelity:.3f}")
            print(f"   TELOS Fidelity: {telos_final_fidelity:.3f}")
            print(f"   ΔF (TELOS - Baseline): {delta_f:+.3f}")

            self.report["comparative_metrics"]["branch_comparisons"].append({
                "turn": turn_num,
                "baseline_fidelity": baseline_fidelity,
                "telos_fidelity": telos_final_fidelity,
                "delta_f": delta_f,
                "intervention_applied": "yes" if intervention_needed else "no"
            })

            # Move to next turn
            i = j + 1

    def _create_steward(self, attractor: PrimacyAttractor):
        """Create unified governance steward."""
        # Wrapper classes
        class LLMClient:
            def __init__(self, mistral, model_name):
                self.mistral = mistral
                self.model = model_name

            def generate(self, messages, max_tokens=500):
                response = self.mistral.chat.complete(
                    model=self.model,  # Use the actual working model
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

        class EmbeddingProviderWrapper:
            def __init__(self, embedder):
                self._embedder = embedder

            def encode(self, text):
                return self._embedder.encode(text)

        return UnifiedGovernanceSteward(
            attractor=attractor,
            llm_client=LLMClient(self.mistral, self.model),  # Pass working model
            embedding_provider=EmbeddingProviderWrapper(self.embedder),
            enable_interventions=True
        )

    def _generate_response(self, history: List[Dict], branch_type: str) -> str:
        """Generate response via API with retry logic."""
        max_retries = 10
        retry_delay = 20

        for retry_count in range(max_retries):
            try:
                response = self.mistral.chat.complete(
                    model=self.model,
                    messages=history,
                    max_tokens=300,
                    temperature=0.7
                )

                # Log API call
                self.report["intervention_analysis"]["api_call_log"].append({
                    "operation": "generate_response",
                    "branch_type": branch_type,
                    "model": self.model,
                    "success": "yes",
                    "retries": retry_count
                })

                return response.choices[0].message.content

            except Exception as e:
                if "429" in str(e) and retry_count < max_retries - 1:
                    print(f"\n⚠️  Rate limit in _generate_response (attempt {retry_count + 1}/{max_retries})")
                    print(f"    Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 60)
                    continue
                else:
                    # Final failure
                    self.report["intervention_analysis"]["api_call_log"].append({
                        "operation": "generate_response",
                        "branch_type": branch_type,
                        "model": self.model,
                        "success": "no",
                        "error": str(e),
                        "retries": retry_count
                    })
                    return f"[Generation failed: {e}]"

    def _apply_intervention(self, response: str, attractor: PrimacyAttractor, history: List[Dict]) -> str:
        """Apply governance intervention with retry logic."""
        correction_prompt = f"""The following response drifted from the intended purpose.

Purpose: {', '.join(attractor.purpose)}
Scope: {', '.join(attractor.scope)}

Original response: {response}

Please revise the response to better align with the stated purpose and scope while maintaining conversational quality."""

        max_retries = 10
        retry_delay = 20

        for retry_count in range(max_retries):
            try:
                corrected = self.mistral.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": correction_prompt}],
                    max_tokens=300,
                    temperature=0.5
                )

                # Log API call
                self.report["intervention_analysis"]["api_call_log"].append({
                    "operation": "apply_intervention",
                    "model": self.model,
                    "success": "yes",
                    "retries": retry_count
                })

                return corrected.choices[0].message.content

            except Exception as e:
                if "429" in str(e) and retry_count < max_retries - 1:
                    print(f"\n⚠️  Rate limit in _apply_intervention (attempt {retry_count + 1}/{max_retries})")
                    print(f"    Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 60)
                    continue
                else:
                    # Final failure
                    self.report["intervention_analysis"]["api_call_log"].append({
                        "operation": "apply_intervention",
                        "model": self.model,
                        "success": "no",
                        "error": str(e),
                        "retries": retry_count
                    })
                    return response  # Return original if correction fails

    def _calculate_fidelity(self, response: str, attractor: PrimacyAttractor, pa_extractor: ProgressivePrimacyExtractor) -> Dict[str, Any]:
        """
        Calculate fidelity score with detailed reasoning.

        Returns dict with:
        - fidelity: float score 0-1
        - cosine_similarity: float 0-1
        - distance: float (Euclidean distance)
        - reasoning: human-readable explanation
        """
        # Get embedding using TELOS provider
        response_emb = self.embedder.encode(response)

        # Get attractor centroid
        centroid = pa_extractor.attractor_centroid

        if centroid is None:
            # Fallback: use purpose embedding
            purpose_text = ' '.join(attractor.purpose)
            centroid = self.embedder.encode(purpose_text)

        # Calculate cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
        cos_sim = 1 - cosine(response_emb, centroid)

        # Calculate Euclidean distance
        distance = float(np.linalg.norm(response_emb - centroid))

        # Convert to fidelity (exponential decay)
        fidelity = np.exp(-distance / 2.0)
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        # Generate human-readable reasoning
        if cos_sim >= 0.85:
            alignment = "strongly aligned"
        elif cos_sim >= 0.70:
            alignment = "moderately aligned"
        elif cos_sim >= 0.50:
            alignment = "weakly aligned"
        else:
            alignment = "divergent"

        divergence_pct = (1.0 - cos_sim) * 100

        reasoning = (
            f"Response is {alignment} with conversation purpose. "
            f"Semantic similarity: {cos_sim:.1%} (divergence: {divergence_pct:.1f}%). "
            f"Fidelity score: {fidelity:.3f}"
        )

        return {
            'fidelity': fidelity,
            'cosine_similarity': float(cos_sim),
            'distance': distance,
            'divergence_percent': float(divergence_pct),
            'reasoning': reasoning
        }

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final forensic report."""
        output_dir = Path("tests/validation_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON report
        json_file = output_dir / f"forensic_report_{timestamp}.json"

        # Debug: Save raw report to inspect
        import pickle
        debug_file = output_dir / f"forensic_report_{timestamp}_debug.pkl"
        with open(debug_file, 'wb') as f:
            pickle.dump(self.report, f)
        print(f"DEBUG: Saved raw report to {debug_file}")

        # Convert and save
        print("DEBUG: Converting booleans...")
        json_safe_report = convert_bools_to_strings(self.report)
        print(f"DEBUG: Converted, saving JSON...")

        # Try saving with better error handling
        try:
            with open(json_file, 'w') as f:
                json.dump(json_safe_report, f, indent=2)
        except TypeError as e:
            print(f"❌ JSON serialization error: {e}")
            print("DEBUG: Attempting to find the problematic value...")

            # Save as pickle as fallback
            import json as json_module

            # Use custom encoder that converts remaining booleans
            class BoolEncoder(json_module.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, bool):
                        return "yes" if obj else "no"
                    if isinstance(obj, (np.bool_, np.bool)):
                        return "yes" if obj else "no"
                    return super().default(obj)

            with open(json_file, 'w') as f:
                json_module.dump(json_safe_report, f, indent=2, cls=BoolEncoder)

        # Generate human-readable report
        txt_file = output_dir / f"forensic_report_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write(self._format_human_readable_report())

        print(f"\n✅ Forensic reports saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📝 Human-readable: {txt_file}")

        return self.report

    def _format_human_readable_report(self) -> str:
        """Format human-readable report."""
        lines = []
        lines.append("="*80)
        lines.append("TELOS FORENSIC VALIDATION REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {self.report['metadata']['timestamp']}")
        lines.append("")

        # PA Establishment Section
        lines.append("\n" + "="*80)
        lines.append("SECTION 1: PROGRESSIVE PRIMACY ATTRACTOR ESTABLISHMENT")
        lines.append("="*80)

        convergence = self.report['pa_establishment']['convergence_summary']
        if convergence.get('converged'):
            lines.append(f"\n✅ PA CONVERGED at turn {convergence['convergence_turn']}")
            lines.append(f"\nAttractor Configuration:")
            attractor = convergence.get('attractor', {})
            lines.append(f"  Purpose: {', '.join(attractor.get('purpose', []))}")
            lines.append(f"  Scope: {', '.join(attractor.get('scope', []))}")
            lines.append(f"  Boundaries: {', '.join(attractor.get('boundaries', []))}")

            lines.append(f"\nConvergence Metrics:")
            metrics = convergence.get('convergence_metrics', {})
            lines.append(f"  Confidence: {metrics.get('confidence', 0):.3f}")
            lines.append(f"  Centroid Stability: {metrics.get('centroid_stability', 0):.3f}")
            lines.append(f"  Variance Stability: {metrics.get('variance_stability', 0):.3f}")
        else:
            lines.append("\n⚠️ PA did not converge")

        lines.append(f"\nTurn-by-Turn Convergence Log:")
        for turn_data in self.report['pa_establishment']['turn_by_turn_convergence']:
            lines.append(f"\n  Turn {turn_data['turn']}: {turn_data['status']}")
            lines.append(f"    {turn_data['message']}")

        # Intervention Analysis Section
        lines.append("\n\n" + "="*80)
        lines.append("SECTION 2: INTERVENTION ANALYSIS")
        lines.append("="*80)

        lines.append(f"\nTotal Decision Points: {len(self.report['intervention_analysis']['decision_points'])}")

        interventions = [d for d in self.report['intervention_analysis']['decision_points'] if d.get('drift_detected')]
        lines.append(f"Interventions Applied: {len(interventions)}")

        for decision in self.report['intervention_analysis']['decision_points']:
            lines.append(f"\n  Turn {decision['turn']}:")
            if decision.get('drift_detected') == 'yes':
                lines.append(f"    🚨 DRIFT DETECTED")
                lines.append(f"    Pre-intervention fidelity: {decision['pre_intervention_fidelity']:.3f}")
                lines.append(f"    Post-intervention fidelity: {decision['post_intervention_fidelity']:.3f}")
                lines.append(f"    Improvement: {decision['improvement']:+.3f}")
            else:
                lines.append(f"    ✅ No intervention needed (F={decision.get('fidelity', 0.0):.3f})")

        # Conversational DNA Section
        lines.append("\n\n" + "="*80)
        lines.append("SECTION 3: CONVERSATIONAL DNA")
        lines.append("="*80)

        lines.append("\nBASELINE BRANCH (No Governance):")
        for dna in self.report['conversational_dna']['baseline_branches'][:5]:  # First 5
            lines.append(f"\n  Turn {dna['turn']}:")
            lines.append(f"    User: {dna['user_input'][:100]}...")
            lines.append(f"    Assistant: {dna['assistant_response'][:100]}...")
            lines.append(f"    Fidelity: {dna['fidelity']:.3f}")

        lines.append("\n\nTELOS BRANCH (Full Governance):")
        for dna in self.report['conversational_dna']['telos_branches'][:5]:  # First 5
            lines.append(f"\n  Turn {dna['turn']}:")
            lines.append(f"    User: {dna['user_input'][:100]}...")
            lines.append(f"    Assistant: {dna['assistant_response'][:100]}...")
            lines.append(f"    Fidelity: {dna['fidelity']:.3f}")
            lines.append(f"    Intervention: {'YES' if dna.get('intervention_applied') else 'NO'}")

        # Comparative Metrics Section
        lines.append("\n\n" + "="*80)
        lines.append("SECTION 4: COMPARATIVE METRICS")
        lines.append("="*80)

        if self.report['comparative_metrics']['branch_comparisons']:
            avg_baseline = np.mean([c['baseline_fidelity'] for c in self.report['comparative_metrics']['branch_comparisons']])
            avg_telos = np.mean([c['telos_fidelity'] for c in self.report['comparative_metrics']['branch_comparisons']])
            avg_delta = np.mean([c['delta_f'] for c in self.report['comparative_metrics']['branch_comparisons']])

            lines.append(f"\nAggregate Results:")
            lines.append(f"  Average Baseline Fidelity: {avg_baseline:.3f}")
            lines.append(f"  Average TELOS Fidelity: {avg_telos:.3f}")
            lines.append(f"  Average ΔF (TELOS improvement): {avg_delta:+.3f}")
            lines.append(f"  Percentage Improvement: {(avg_delta / avg_baseline * 100):+.1f}%")

            lines.append(f"\nPer-Turn Comparison:")
            for comp in self.report['comparative_metrics']['branch_comparisons']:
                lines.append(f"\n  Turn {comp['turn']}:")
                lines.append(f"    Baseline: {comp['baseline_fidelity']:.3f}")
                lines.append(f"    TELOS: {comp['telos_fidelity']:.3f}")
                lines.append(f"    ΔF: {comp['delta_f']:+.3f}")
                lines.append(f"    Intervention: {'YES' if comp.get('intervention_applied') else 'NO'}")

        # API Call Log Section
        lines.append("\n\n" + "="*80)
        lines.append("SECTION 5: API CALL LOG")
        lines.append("="*80)

        lines.append(f"\nTotal API Calls: {len(self.report['intervention_analysis']['api_call_log'])}")

        call_types = {}
        for call in self.report['intervention_analysis']['api_call_log']:
            op = call['operation']
            call_types[op] = call_types.get(op, 0) + 1

        lines.append(f"\nBreakdown by Operation:")
        for op, count in call_types.items():
            lines.append(f"  {op}: {count} calls")

        lines.append("\n" + "="*80)
        lines.append("END OF FORENSIC REPORT")
        lines.append("="*80)

        return '\n'.join(lines)


def main():
    """Run forensic validation."""
    print("\n🔬 TELOS FORENSIC VALIDATION RUNNER")
    print("="*80)

    validator = ForensicValidator()

    # Target conversation
    conversation_file = "tests/validation_data/baseline_conversations/real_claude_conversation.json"

    # Run validation starting from turn 9 (skip file upload preamble)
    report = validator.run_forensic_validation(conversation_file, start_turn=9)

    if report:
        print("\n✅ FORENSIC VALIDATION COMPLETE")
        print("\nThis report contains:")
        print("  ✓ PA establishment timeline (turn-by-turn)")
        print("  ✓ Counterfactual branch isolation proof")
        print("  ✓ Every intervention decision point")
        print("  ✓ Conversational DNA (actual text)")
        print("  ✓ Comparative metrics (baseline vs TELOS)")
        print("  ✓ Complete API call log")
    else:
        print("\n❌ Forensic validation failed")


if __name__ == "__main__":
    main()
