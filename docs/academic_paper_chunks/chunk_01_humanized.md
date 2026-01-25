# TELOS Academic Paper - Chunk 1: Abstract and Introduction (HUMANIZED)

## Abstract

We present TELOS, a runtime AI governance system that achieves a 0% observed Attack Success Rate (ASR) across 1,300 adversarial attacks (95% CI: [0%, 0.28%]). This result is unprecedented in AI safety literature. While current top systems accept violation rates of 3.7% to 43.9% as unavoidable, TELOS shows that mathematical enforcement of constitutional boundaries can provide near-perfect defense. It uses a novel three-tier structure that combines embedding-space mathematics, authoritative policy retrieval, and human expert escalation.

Our key innovation applies quality control methods (Lean Six Sigma DMAIC/SPC) to AI governance. We treat constitutional enforcement as a statistical process control problem rather than a prompt engineering challenge. This insight, executed with Primacy Attractor (PA) mathematics and Lyapunov-stable basin dynamics, creates a solid governance model against tested attack vectors.

We validate our method across 1,300 attacks (400 from HarmBench general-purpose and 900 from MedSafetyBench healthcare-specific), covering various harm categories from direct violations to complex jailbreaks. TELOS-governed models achieve 0% ASR on both small and large language models. In contrast, baseline methods using system prompts show an ASR of 3.7-11.1%, while raw models exhibit an ASR of 30.8-43.9%.

Beyond the main governance system, we introduce TELOSCOPE, a research tool that makes AI governance observable and measurable through counterfactual analysis and forensic decision tracing. All results are fully reproducible with the provided code and attack libraries.

Keywords: AI safety, constitutional AI, adversarial robustness, embedding space, Lyapunov stability, governance verification

## 1. Introduction

The deployment of Large Language Models (LLMs) in regulated fields such as healthcare, finance, and education presents a fundamental conflict. These systems offer transformative capabilities, but they lack reliable ways to enforce regulatory boundaries. This conflict has become legally urgent. The European Union's AI Act requires runtime monitoring and ongoing compliance for high-risk AI systems by August 2026. Meanwhile, California's SB 243 is the first state law aimed explicitly at AI chatbot safety for minors, effective January 2026. These regulations demand mechanisms that current governance approaches cannot provide.

Current methods for AI governance, whether through fine-tuning, prompt engineering, or post-hoc filtering, often fail against adversarial attacks. The HarmBench benchmark found that leading AI systems show attack success rates of 4.4-90% across 400 standardized attacks. MedSafetyBench revealed similar weaknesses in healthcare contexts with 900 domain-specific attacks. Leading guardrail systems, such as NVIDIA NeMo Guardrails and Llama Guard, accept violation rates between 3.7% and 43.9% as unavoidable, which is incompatible with emerging regulatory requirements.

We question this accepted failure rate. By applying new insights from industrial quality control to AI governance, we demonstrate that constitutional violations are not inevitable but a choice to accept flawed governance.

### 1.1 The Governance Problem

Consider a healthcare AI assistant that must never disclose Protected Health Information (PHI) under HIPAA regulations. Current methods fail in predictable ways:

1. Prompt Engineering: System prompts stating "never disclose PHI" can easily be bypassed using social engineering or prompt injection.
2. Fine-tuning: RLHF/DPO methods embed constraints into model weights but remain vulnerable to jailbreaks.
3. Output Filtering: Filtering after generation captures obvious violations but overlooks semantic equivalents.

The core issue is that all current methods treat governance as a linguistic problem (what the model states) rather than a geometric problem (the location of the query in semantic space).

### 1.2 Our Approach: Governance as Geometric Control

TELOS redefines AI governance through three key insights:

1. Fixed Reference Points: Instead of relying on the model's shifting attention for self-governance, we set fixed reference points (Primacy Attractors) in the embedding space.
2. Mathematical Enforcement: Cosine similarity in the embedding space offers a clear, non-bypassable measure of constitutional alignment.
3. Three-Tier Defense: The system ensures that mathematical (PA), authoritative (RAG), and human (Expert) layers must all fail simultaneously for a violation to occur.

### 1.3 Contributions

This paper makes four main contributions:

1. Theoretical: We demonstrate that external reference points in the embedding space enable stable governance with defined basin geometry (r = 2/ρ).
2. Empirical: We show 0% ASR across 1,300 adversarial attacks (400 HarmBench + 900 MedSafetyBench), compared to 3.7-43.9% for existing methods.
3. Methodological: We introduce TELOSCOPE, a research tool for visible AI governance through counterfactual analysis.
4. Practical: We provide full reproducible validation with a healthcare-specific implementation that achieves HIPAA compliance.
