# TELOS Academic Paper - Chunk 1: Abstract and Introduction

## Abstract

We present TELOS, a runtime AI governance system that achieves 0% observed Attack Success Rate (ASR) across 1,300 adversarial attacks (95% CI: [0%, 0.28%])—unprecedented in AI safety literature. While current state-of-the-art systems accept violation rates of 3.7% to 43.9% as inevitable, TELOS demonstrates that mathematical enforcement of constitutional boundaries can achieve near-perfect defense through a novel three-tier architecture combining embedding-space mathematics, authoritative policy retrieval, and human expert escalation.

Our key innovation applies industrial quality control methodologies (Lean Six Sigma DMAIC/SPC) to AI governance, treating constitutional enforcement as a statistical process control problem rather than a prompt engineering challenge. This cross-domain insight, implemented through Primacy Attractor (PA) mathematics with Lyapunov-stable basin dynamics, creates mathematically grounded governance against tested attack vectors.

We validate our approach across 1,300 attacks (400 from HarmBench general-purpose, 900 from MedSafetyBench healthcare-specific) spanning multiple harm categories from direct violations to sophisticated jailbreaks. TELOS-governed models achieve 0% ASR on both small and large language models, while baseline approaches using system prompts show 3.7-11.1% ASR and raw models exhibit 30.8-43.9% ASR.

Beyond the core governance system, we introduce TELOSCOPE, a research instrument for making AI governance observable and measurable through counterfactual analysis and forensic decision tracing. All results are fully reproducible with provided code and attack libraries.

Keywords: AI safety, constitutional AI, adversarial robustness, embedding space, Lyapunov stability, governance verification

## 1. Introduction

The deployment of Large Language Models (LLMs) in regulated sectors—healthcare, finance, education—presents a fundamental tension: these systems offer transformative capabilities but lack reliable mechanisms to enforce regulatory boundaries. This tension has become legally urgent: the European Union's AI Act mandates runtime monitoring and continuous compliance for high-risk AI systems by August 2026, while California's SB 243 represents the first state legislation specifically targeting AI chatbot safety for minors, effective January 2026. These regulations explicitly require mechanisms that current governance approaches cannot provide.

Current approaches to AI governance, whether through fine-tuning, prompt engineering, or post-hoc filtering, consistently fail against adversarial attacks. The HarmBench benchmark established that leading AI systems exhibit 4.4-90% attack success rates across 400 standardized attacks, while MedSafetyBench demonstrated similar vulnerabilities in healthcare contexts with 900 domain-specific attacks. State-of-the-art guardrail systems, including NVIDIA NeMo Guardrails and Llama Guard, accept violation rates between 3.7% and 43.9% as unavoidable—a failure rate incompatible with emerging regulatory requirements.

We challenge this accepted failure rate. Through a novel cross-domain insight applying industrial quality control methodologies to AI governance, we demonstrate that constitutional violations are not inevitable—they are a choice to accept imperfect governance.

### 1.1 The Governance Problem

Consider a healthcare AI assistant that must never disclose Protected Health Information (PHI) per HIPAA regulations. Current approaches fail in predictable ways:

1. Prompt Engineering: System prompts saying "never disclose PHI" are easily bypassed through social engineering or prompt injection
2. Fine-tuning: RLHF/DPO approaches bake constraints into model weights but remain vulnerable to jailbreaks
3. Output Filtering: Post-generation filtering catches obvious violations but misses semantic equivalents

The core issue: all current approaches treat governance as a linguistic problem (what the model says) rather than a geometric problem (where the query lives in semantic space).

### 1.2 Our Approach: Governance as Geometric Control

TELOS reconceptualizes AI governance through three key insights:

1. Fixed Reference Points: Instead of using the model's shifting attention mechanism for self-governance, we establish immutable reference points (Primacy Attractors) in embedding space
2. Mathematical Enforcement: Cosine similarity in embedding space provides deterministic, non-bypassable measurement of constitutional alignment
3. Three-Tier Defense: Mathematical (PA) to Authoritative (RAG) to Human (Expert) escalation ensures all three layers must fail simultaneously for violation

### 1.3 Contributions

This paper makes four primary contributions:

1. Theoretical: We prove that external reference points in embedding space enable Lyapunov-stable governance with characterized basin geometry (r = 2/ρ)
2. Empirical: We demonstrate 0% ASR across 1,300 adversarial attacks (400 HarmBench + 900 MedSafetyBench), compared to 3.7-43.9% for existing approaches
3. Methodological: We introduce TELOSCOPE, a research instrument for observable AI governance through counterfactual analysis
4. Practical: We provide complete reproducible validation with healthcare-specific implementation achieving HIPAA compliance
