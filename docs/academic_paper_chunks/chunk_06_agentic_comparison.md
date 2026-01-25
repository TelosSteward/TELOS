# TELOS Academic Paper - Chunk 6: Agentic AI and System Comparison

### 8.6 Agentic AI and Multi-Agent Systems

The emergence of agentic AI systems creates new governance challenges not addressed by conversational safety alone. Recent work on "Super Agents" focuses on capability enhancement through multi-agent coordination but does not address the governance gap this creates: when AI agents can invoke tools and execute multi-step plans, each tool invocation represents a potential governance failure point.

Emerging regulatory frameworks explicitly anticipate tool use governance. California's SB 53 mandates transparency about AI system capabilities including tool invocation, while the EU AI Act requires human oversight for autonomous AI decisions. TELOS's architecture—measuring fidelity at each decision point—provides a foundation for extending conversational governance to agentic contexts, though we note this represents future work beyond the scope of our current empirical validation.

### 8.7 Quantitative Comparison

| System | Approach | Reported ASR Range | Source |
|--------|----------|-------------------|--------|
| Constitutional AI | RLHF training | 3.7-8.2% | HarmBench eval |
| OpenAI Moderation | Post-generation filter | 5.1-12.3% | HarmBench eval |
| NeMo Guardrails | Colang rules | 4.8-9.7% | Self-reported |
| Llama Guard | Classifier-based | 4.4-7.3% | HarmBench eval |
| TELOS | PA + 3-Tier | 0.0% (95% CI: 0-0.28%) | This work |

Note: Baseline ASR ranges are approximate and derived from HarmBench evaluations where available or self-reported benchmarks. Latency comparisons are omitted as we did not conduct head-to-head latency testing. TELOS latency (less than 50ms) is measured on our validation infrastructure.

TELOS achieves the lowest observed ASR. Direct comparison should be interpreted with caution due to differences in attack sets, evaluation methodology, and model configurations across studies.

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. Embedding Model Dependency: Results tied to Mistral embeddings; other models may require retuning
2. Domain Specificity: Healthcare validation doesn't guarantee finance/legal performance
3. Computational Overhead: ~50ms latency per query (acceptable for most applications)
4. Human Scalability: Tier 3 doesn't scale to millions of daily queries

### 9.2 Future Directions

1. Multi-Modal Extension: Expand PA to image/audio inputs using CLIP-style embeddings
2. Adaptive PAs: Federated learning for PA updates across consortium sites
3. Formal Verification: Prove stronger properties beyond Lyapunov stability
4. Economic Analysis: Cost-benefit study of TELOS vs. manual compliance

### 9.3 Extension to Agentic AI (Anticipatory Work)

The emergence of agentic AI systems—LLMs that can invoke tools, execute code, and orchestrate multi-step plans—creates governance challenges beyond conversational safety. When an AI agent proposes to execute DELETE FROM patients WHERE status='inactive', the governance question extends beyond "is this query appropriate?" to "is this action aligned with the agent's authorized purpose?"

We anticipate that regulatory frameworks will require per-tool governance for agentic AI. California's SB 53 already mandates transparency about AI capabilities including autonomous actions, while the EU AI Act requires human oversight for high-risk autonomous decisions. This regulatory trajectory suggests that the type of per-action fidelity measurement TELOS provides will become mandatory.

Architectural Foundation: TELOS's Primacy Attractor architecture naturally extends to agentic contexts. Each proposed tool invocation can be measured against the PA before execution:

Tool_Fidelity = cosine(embed(tool_call + arguments), PA)

Tool calls with fidelity below threshold would trigger the same three-tier escalation: mathematical block to policy guidance to human expert review.

Important Caveat: We emphasize that our empirical validation covers conversational governance only. The 0% ASR claim applies to our 1,300 conversational attacks from HarmBench and MedSafetyBench. Extension to agentic tool governance represents future work requiring separate validation against agentic attack patterns, which the research community has only begun to characterize. We include this section to demonstrate proactive alignment with anticipated regulatory requirements, not to claim empirical validation of agentic governance.

## 10. Conclusion

TELOS demonstrates that AI constitutional violations are not inevitable. Through mathematical enforcement in embedding space, we achieve 0% observed Attack Success Rate across 1,300 adversarial tests (95% CI: [0%, 0.28%])—unprecedented in AI safety literature.

Our three contributions—theoretical (Lyapunov-stable PA mathematics), empirical (0% ASR validation), and methodological (TELOSCOPE observability)—provide a foundation for trustworthy AI deployment in regulated sectors.

The path from research to production is clear: healthcare organizations can deploy TELOS today for HIPAA compliance, while financial and educational institutions can adapt the framework for their regulatory requirements.

We invite the research community to reproduce our results, extend to new domains, and join us in building mathematically enforceable AI governance. The code is open source (Apache 2.0), the validation protocol is automated, and the societal need is urgent.

The future of trustworthy AI depends not on accepting imperfect governance, but on building systems that make violations statistically improbable.
