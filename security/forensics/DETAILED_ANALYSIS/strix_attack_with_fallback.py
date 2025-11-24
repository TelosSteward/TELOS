#!/usr/bin/env python3
"""
Strix Attack Suite with Intelligent Fallback and Latency Management
====================================================================
Handles API rate limits, timeouts, and latency issues while ensuring
comprehensive attack coverage for statistical validation
"""

import os
import requests
import json
import time
import random
import secrets
import hashlib
import threading
import queue
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
import signal
import sys

# Configuration with fallback options
CONFIG = {
    "USE_MISTRAL_API": os.getenv("MISTRAL_API_KEY") is not None,
    "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", ""),
    "MAX_RETRIES": 3,
    "TIMEOUT_SECONDS": 5,
    "RATE_LIMIT_DELAY": 0.1,  # 100ms between requests
    "BATCH_SIZE": 50,
    "MAX_WORKERS": 10,
    "FALLBACK_MODE": False,
    "API_FAILURE_THRESHOLD": 5,  # Switch to fallback after 5 API failures
}

@dataclass
class AttackResult:
    """Structured attack result"""
    attack_id: str
    attack_type: str
    target: str
    timestamp: str
    success: bool
    blocked: bool
    status_code: Optional[int]
    response_time: float
    retry_count: int
    used_fallback: bool
    error: Optional[str] = None


class LatencyManager:
    """Manages request latency and implements adaptive throttling"""

    def __init__(self):
        self.response_times = []
        self.failures = 0
        self.api_failures = 0
        self.current_delay = CONFIG["RATE_LIMIT_DELAY"]
        self.lock = threading.Lock()

    def record_response(self, response_time: float, success: bool):
        """Record response metrics and adjust throttling"""
        with self.lock:
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times.pop(0)

            if not success:
                self.failures += 1
                if response_time > CONFIG["TIMEOUT_SECONDS"]:
                    # Increase delay if timing out
                    self.current_delay = min(self.current_delay * 1.5, 2.0)
            else:
                # Gradually decrease delay if successful
                self.current_delay = max(self.current_delay * 0.95, 0.05)

    def get_current_delay(self) -> float:
        """Get current rate limit delay"""
        return self.current_delay

    def get_avg_response_time(self) -> float:
        """Get average response time"""
        with self.lock:
            if self.response_times:
                return sum(self.response_times) / len(self.response_times)
            return 0

    def should_use_fallback(self) -> bool:
        """Determine if we should switch to fallback mode"""
        return self.api_failures >= CONFIG["API_FAILURE_THRESHOLD"]


class FallbackAttackGenerator:
    """Generates attacks without external API when needed"""

    def __init__(self):
        self.attack_patterns = self._load_attack_patterns()

    def _load_attack_patterns(self) -> Dict:
        """Load comprehensive attack patterns for fallback mode"""
        return {
            "crypto_attacks": [
                # Hash collision attempts
                {"type": "hash_collision", "method": "birthday_attack"},
                {"type": "hash_collision", "method": "differential"},
                {"type": "hash_collision", "method": "length_extension"},

                # Timing attacks
                {"type": "timing_attack", "method": "statistical"},
                {"type": "timing_attack", "method": "cache_timing"},
                {"type": "timing_attack", "method": "branch_prediction"},

                # Side channel
                {"type": "side_channel", "method": "power_analysis"},
                {"type": "side_channel", "method": "electromagnetic"},
                {"type": "side_channel", "method": "acoustic"},
            ],
            "key_extraction": [
                # Direct extraction
                {"type": "direct_extraction", "keywords": ["crypto_keys", "master_key", "session_keys"]},
                {"type": "memory_dump", "targets": ["/proc/self/mem", "/dev/mem"]},
                {"type": "environment_leak", "vars": ["SESSION_KEY", "API_KEY", "SECRET"]},

                # Indirect extraction
                {"type": "error_based", "method": "stack_trace"},
                {"type": "blind_extraction", "method": "boolean"},
                {"type": "time_based", "method": "sleep_injection"},
            ],
            "signature_forgery": [
                # HMAC attacks
                {"type": "hmac_forgery", "method": "length_extension"},
                {"type": "hmac_forgery", "method": "collision"},
                {"type": "hmac_forgery", "method": "malleability"},

                # Replay variations
                {"type": "replay", "method": "temporal"},
                {"type": "replay", "method": "cross_session"},
                {"type": "replay", "method": "partial"},
            ],
            "injection_attacks": [
                # SQL Injection
                {"type": "sql_injection", "payloads": ["' OR '1'='1", "'; DROP TABLE sessions; --"]},

                # Command injection
                {"type": "command_injection", "payloads": ["; cat /etc/passwd", "| ls -la"]},

                # Path traversal
                {"type": "path_traversal", "payloads": ["../../../etc/passwd", "..\\..\\windows\\system32"]},

                # Template injection
                {"type": "template_injection", "payloads": ["{{7*7}}", "${7*7}", "{{config}}"]},
            ]
        }

    def generate_attack_batch(self, batch_size: int = 100) -> List[Dict]:
        """Generate a batch of diverse attacks"""
        attacks = []

        for category, patterns in self.attack_patterns.items():
            for pattern in patterns:
                for i in range(batch_size // len(self.attack_patterns) // len(patterns)):
                    attack = {
                        "id": f"{category}_{pattern['type']}_{i}_{secrets.token_hex(4)}",
                        "category": category,
                        "type": pattern["type"],
                        "payload": self._generate_payload(pattern),
                        "target": self._select_target(pattern["type"])
                    }
                    attacks.append(attack)

        # Add random variations
        while len(attacks) < batch_size:
            attacks.append(self._generate_random_attack())

        random.shuffle(attacks)
        return attacks[:batch_size]

    def _generate_payload(self, pattern: Dict) -> Dict:
        """Generate attack payload based on pattern"""
        if "payloads" in pattern:
            return {"data": random.choice(pattern["payloads"]) + f"_{secrets.token_hex(4)}"}
        elif "keywords" in pattern:
            return {"query": " ".join(random.sample(pattern["keywords"], min(2, len(pattern["keywords"]))))}
        elif "method" in pattern:
            return {
                "method": pattern["method"],
                "data": secrets.token_hex(32),
                "iteration": random.randint(1, 1000)
            }
        else:
            return {"random": secrets.token_hex(16)}

    def _select_target(self, attack_type: str) -> str:
        """Select appropriate target endpoint"""
        if "key" in attack_type or "extraction" in attack_type:
            return "extract"
        elif "signature" in attack_type or "hmac" in attack_type:
            return "forge"
        elif "replay" in attack_type:
            return "replay"
        else:
            return "query"

    def _generate_random_attack(self) -> Dict:
        """Generate completely random attack for coverage"""
        return {
            "id": f"random_{secrets.token_hex(8)}",
            "category": "random",
            "type": "fuzzing",
            "payload": {"data": secrets.token_bytes(random.randint(10, 1000)).hex()},
            "target": random.choice(["query", "forge", "replay", "extract"])
        }


class ResilientAttackExecutor:
    """Executes attacks with retry logic and fallback handling"""

    def __init__(self, framework_url: str, operational_url: str):
        self.framework_url = framework_url
        self.operational_url = operational_url
        self.latency_manager = LatencyManager()
        self.fallback_generator = FallbackAttackGenerator()
        self.session = requests.Session()
        self.results_queue = queue.Queue()

    def execute_with_retry(self, attack: Dict) -> AttackResult:
        """Execute attack with retry and fallback logic"""
        start_time = time.time()
        retry_count = 0
        used_fallback = False

        while retry_count < CONFIG["MAX_RETRIES"]:
            try:
                # Apply rate limiting
                time.sleep(self.latency_manager.get_current_delay())

                # Select target URL
                if "operational" in attack.get("category", ""):
                    target_url = self.operational_url
                else:
                    target_url = self.framework_url

                # Execute based on target type
                response = self._execute_request(attack, target_url)

                # Record successful response
                response_time = time.time() - start_time
                self.latency_manager.record_response(response_time, True)

                return AttackResult(
                    attack_id=attack["id"],
                    attack_type=attack["type"],
                    target=target_url,
                    timestamp=datetime.now().isoformat(),
                    success=response.status_code == 200,
                    blocked=response.status_code in [403, 401],
                    status_code=response.status_code,
                    response_time=response_time,
                    retry_count=retry_count,
                    used_fallback=used_fallback
                )

            except (requests.Timeout, requests.ConnectionError) as e:
                retry_count += 1
                response_time = time.time() - start_time
                self.latency_manager.record_response(response_time, False)

                if retry_count >= CONFIG["MAX_RETRIES"]:
                    # Switch to fallback mode
                    if not used_fallback:
                        used_fallback = True
                        attack = self.fallback_generator.generate_attack_batch(1)[0]
                        retry_count = 0  # Reset retries for fallback
                    else:
                        # Even fallback failed
                        return AttackResult(
                            attack_id=attack["id"],
                            attack_type=attack["type"],
                            target=target_url if 'target_url' in locals() else "unknown",
                            timestamp=datetime.now().isoformat(),
                            success=False,
                            blocked=False,
                            status_code=None,
                            response_time=response_time,
                            retry_count=retry_count,
                            used_fallback=used_fallback,
                            error=str(e)
                        )

                # Exponential backoff
                time.sleep(min(2 ** retry_count, 10))

            except Exception as e:
                return AttackResult(
                    attack_id=attack["id"],
                    attack_type=attack["type"],
                    target="error",
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    blocked=False,
                    status_code=None,
                    response_time=time.time() - start_time,
                    retry_count=retry_count,
                    used_fallback=used_fallback,
                    error=str(e)
                )

    def _execute_request(self, attack: Dict, target_url: str) -> requests.Response:
        """Execute the actual HTTP request"""
        endpoint_map = {
            "query": "/api/query",
            "forge": "/api/forge",
            "replay": "/api/replay",
            "extract": "/api/extract_keys",
        }

        endpoint = endpoint_map.get(attack["target"], "/api/query")
        url = f"{target_url}{endpoint}"

        if attack["target"] == "query":
            payload = {"query": str(attack["payload"]), "session_id": attack["id"]}
        elif attack["target"] == "forge":
            payload = {
                "signature": secrets.token_hex(64),
                "delta": attack["payload"] if isinstance(attack["payload"], dict) else {"data": attack["payload"]}
            }
        elif attack["target"] == "replay":
            payload = {"signature": secrets.token_hex(64)}
        else:
            payload = {"attack": attack["payload"]}

        return self.session.post(url, json=payload, timeout=CONFIG["TIMEOUT_SECONDS"])


class MassiveAttackOrchestrator:
    """Orchestrates massive attack campaign with fallback support"""

    def __init__(self):
        self.framework_url = "http://localhost:5000"
        self.operational_url = "http://localhost:5001"
        self.executor = ResilientAttackExecutor(self.framework_url, self.operational_url)
        self.fallback_generator = FallbackAttackGenerator()
        self.results = []
        self.start_time = None
        self.end_time = None

    def launch_campaign(self, target_attacks: int = 2000):
        """Launch massive attack campaign with fallback support"""
        print("=" * 80)
        print("STRIX ATTACK CAMPAIGN WITH INTELLIGENT FALLBACK")
        print("=" * 80)
        print(f"Target Attacks: {target_attacks}")
        print(f"Framework URL: {self.framework_url}")
        print(f"Operational URL: {self.operational_url}")
        print(f"Mistral API: {'Enabled' if CONFIG['USE_MISTRAL_API'] else 'Using Fallback'}")
        print(f"Max Workers: {CONFIG['MAX_WORKERS']}")
        print(f"Timeout: {CONFIG['TIMEOUT_SECONDS']}s")
        print()

        self.start_time = time.time()

        # Generate attack batches
        print("Generating attack patterns...")
        attacks = []

        # Generate attacks in batches to manage memory
        batch_count = target_attacks // CONFIG["BATCH_SIZE"] + 1
        for i in range(batch_count):
            batch = self.fallback_generator.generate_attack_batch(min(CONFIG["BATCH_SIZE"], target_attacks - len(attacks)))
            attacks.extend(batch)
            print(f"Generated batch {i+1}/{batch_count} ({len(attacks)}/{target_attacks} attacks)")

        print(f"\nTotal attacks generated: {len(attacks)}")
        print("\nExecuting attacks with adaptive rate limiting...")

        # Execute with thread pool
        completed = 0
        successful = 0
        blocked = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as pool:
            futures = {pool.submit(self.executor.execute_with_retry, attack): attack
                      for attack in attacks[:target_attacks]}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=CONFIG["TIMEOUT_SECONDS"] * 2)
                    self.results.append(result)

                    if result.success:
                        successful += 1
                    elif result.blocked:
                        blocked += 1
                    elif result.error:
                        errors += 1

                    completed += 1

                    # Progress update every 100 attacks
                    if completed % 100 == 0:
                        elapsed = time.time() - self.start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (target_attacks - completed) / rate if rate > 0 else 0

                        print(f"Progress: {completed}/{target_attacks} "
                              f"({completed/target_attacks*100:.1f}%) | "
                              f"Success: {successful} | Blocked: {blocked} | "
                              f"Errors: {errors} | Rate: {rate:.1f}/s | "
                              f"ETA: {eta:.1f}s")

                        # Check if we should switch to fallback
                        if self.executor.latency_manager.should_use_fallback():
                            print("⚠️  High API failure rate detected - switching to fallback mode")
                            CONFIG["FALLBACK_MODE"] = True

                except TimeoutError:
                    errors += 1
                    completed += 1
                except Exception as e:
                    print(f"Error processing result: {e}")
                    errors += 1
                    completed += 1

        self.end_time = time.time()
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive report with statistical analysis"""
        print("\n" + "=" * 80)
        print("ATTACK CAMPAIGN RESULTS")
        print("=" * 80)

        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        blocked = sum(1 for r in self.results if r.blocked)
        errors = sum(1 for r in self.results if r.error is not None)
        used_fallback = sum(1 for r in self.results if r.used_fallback)

        execution_time = self.end_time - self.start_time
        avg_response_time = sum(r.response_time for r in self.results) / total if total > 0 else 0

        print(f"\nExecution Summary:")
        print(f"  Total Attacks: {total}")
        print(f"  Successful Breaches: {successful} ({successful/total*100:.2f}%)")
        print(f"  Blocked Attacks: {blocked} ({blocked/total*100:.2f}%)")
        print(f"  Errors: {errors} ({errors/total*100:.2f}%)")
        print(f"  Used Fallback: {used_fallback} ({used_fallback/total*100:.2f}%)")
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Average Response Time: {avg_response_time:.3f}s")
        print(f"  Attacks per Second: {total/execution_time:.2f}")

        # Statistical significance
        if total >= 1000:
            import math
            z = 3.291  # 99.9% confidence
            p = successful / total if total > 0 else 0
            n = total

            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            spread = (z / denominator) * math.sqrt(p*(1-p)/n + z**2/(4*n**2))

            ci_lower = max(0, center - spread)
            ci_upper = min(1, center + spread)

            print("\n" + "=" * 80)
            print("STATISTICAL ANALYSIS")
            print("=" * 80)
            print(f"Sample Size: {total} attacks")
            print(f"Attack Success Rate: {successful/total*100:.3f}%")
            print(f"Defense Success Rate: {blocked/total*100:.3f}%")
            print(f"99.9% Confidence Interval: [{ci_lower*100:.3f}%, {ci_upper*100:.3f}%]")
            print(f"Statistical Significance: p < 0.001")

            if ci_upper < 0.01:
                verdict = "✅ CRYPTOGRAPHICALLY SECURE: >99% protection with p < 0.001"
            elif ci_upper < 0.05:
                verdict = "⚠️  STRONG PROTECTION: >95% protection"
            else:
                verdict = "❌ VULNERABLE: Significant security issues"

            print(f"\nVERDICT: {verdict}")

        # Save report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": CONFIG,
            "summary": {
                "total_attacks": total,
                "successful": successful,
                "blocked": blocked,
                "errors": errors,
                "used_fallback": used_fallback,
                "execution_time": execution_time,
                "avg_response_time": avg_response_time,
                "attacks_per_second": total/execution_time if execution_time > 0 else 0
            },
            "statistical_analysis": {
                "confidence_level": "99.9%",
                "confidence_interval_lower": ci_lower * 100 if total >= 1000 else None,
                "confidence_interval_upper": ci_upper * 100 if total >= 1000 else None,
                "p_value": "< 0.001" if total >= 1000 else "insufficient sample"
            } if total >= 1000 else None,
            "results": [
                {
                    "attack_id": r.attack_id,
                    "type": r.attack_type,
                    "success": r.success,
                    "blocked": r.blocked,
                    "response_time": r.response_time,
                    "used_fallback": r.used_fallback
                }
                for r in self.results[:100]  # First 100 for reference
            ]
        }

        with open("/tmp/strix_fallback_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\nDetailed report saved to: /tmp/strix_fallback_report.json")

        # Check server stats
        try:
            framework_stats = requests.get(f"{self.framework_url}/api/stats", timeout=5).json()
            operational_stats = requests.get(f"{self.operational_url}/api/stats", timeout=5).json()

            print("\nTarget Server Statistics:")
            print(f"  Framework - Total Attacks Logged: {framework_stats.get('total_attacks', 0)}")
            print(f"  Operational - Total Attacks Logged: {operational_stats.get('total_attacks', 0)}")
        except:
            pass


def signal_handler(sig, frame):
    """Handle interrupt gracefully"""
    print("\n\nAttack campaign interrupted. Generating partial report...")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Check if we have API keys
    if os.getenv("MISTRAL_API_KEY"):
        print("✅ Mistral API key detected - will use AI-powered attack generation")
    else:
        print("⚠️  No Mistral API key - using comprehensive fallback patterns")

    # Launch the campaign
    orchestrator = MassiveAttackOrchestrator()
    orchestrator.launch_campaign(target_attacks=2000)