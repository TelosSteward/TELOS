# TELOS Academic Paper - Chunk 6: Agentic AI and System Comparison (HUMANIZED)

### 8.6 Agentic AI and Multi-Agent Systems

The rise of agentic AI systems presents new governance challenges that conversational safety alone does not address. Recent studies on "Super Agents" focus on improving capabilities through coordination among multiple agents. However, they fail to tackle the governance gap. When AI agents can use tools and carry out complex plans, each tool usage presents a potential point of governance failure.

New regulatory frameworks specifically consider governance around tool usage. California's SB 53 requires transparency about AI system capabilities, including tool usage. Meanwhile, the EU AI Act calls for human oversight regarding autonomous AI decisions. TELOS's architecture measures fidelity at each decision point, forming a basis for extending conversational governance to agentic contexts. We note that this represents future work beyond what our current empirical validation covers.

### 8.7 Quantitative Comparison

| System | Approach | Reported ASR Range | Source |
|--------|----------|-------------------|--------|
| Constitutional AI | RLHF training | 3.7-8.2% | HarmBench eval |
| OpenAI Moderation | Post-generation filter | 5.1-12.3% | HarmBench eval |
| NeMo Guardrails | Colang rules | 4.8-9.7% | Self-reported |
| Llama Guard | Classifier-based | 4.4-7.3% | HarmBench eval |
| TELOS | PA + 3-Tier | 0.0% (95% CI: 0-0.28%) | This work |

Note: The ASR ranges are approximate and taken from HarmBench evaluations or self-reported benchmarks. We have excluded latency comparisons since we did not perform direct latency testing. TELOS latency is less than 50ms, measured on our validation infrastructure.

TELOS achieves the lowest observed ASR. Any direct comparisons should be interpreted cautiously due to variations in attack sets, evaluation methods, and model configurations across studies.

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. Model Dependency: Results depend on Mistral embeddings; other models may need retuning.
2. Domain Specificity: Validation in healthcare does not guarantee performance in finance or legal areas.
3. Computational Overhead: Approximately 50ms latency per query, which is acceptable for most applications.
4. Human Scalability: Tier 3 does not scale to millions of daily queries.

### 9.2 Future Directions

1. Multi-Modal Extension: Expand PA to include image and audio inputs using CLIP-style embeddings.
2. Adaptive PAs: Implement federated learning for PA updates across consortium sites.
3. Formal Verification: Prove stronger properties beyond Lyapunov stability.
4. Economic Analysis: Conduct a cost-benefit study of TELOS versus manual compliance.

### 9.3 Extension to Agentic AI (Anticipatory Work)

The arrival of agentic AI systems, such as LLMs that can use tools, run code, and organize complex plans, introduces governance challenges that exceed conversational safety. When an AI agent suggests executing a command like DELETE FROM patients WHERE status='inactive', the governance question changes from "is this command appropriate?" to "is this action consistent with the agent's approved purpose?"

We expect that regulatory frameworks will require governance for each tool used by agentic AI. California's SB 53 already demands transparency regarding AI capabilities, including autonomous actions. The EU AI Act also requires human oversight for high-risk autonomous decisions. This trend suggests that the type of fidelity measurement TELOS offers for each action may soon become a requirement.

Architectural Foundation: TELOS's Primacy Attractor architecture extends naturally to agentic contexts. Each proposed tool use can be evaluated against the PA before execution:

Tool_Fidelity = cosine(embed(tool_call + arguments), PA)

Tool uses that fall below a certain fidelity threshold would lead to a three-tier escalation process: starting with a mathematical block, moving to policy guidance, and finally to human expert review.

Important Caveat: We stress that our empirical validation only covers conversational governance. The 0% ASR claim pertains to our 1,300 conversational attacks from HarmBench and MedSafetyBench. Extending to governance of agentic tools requires separate validation against patterns of agentic attacks, which the research community has only begun to outline. We include this section to show our proactive stance ahead of expected regulatory needs, not to assert empirical validation of agentic governance.

## 10. Conclusion

TELOS shows that AI constitutional violations are not unavoidable. With mathematical enforcement in embedding space, we achieve a 0% observed Attack Success Rate across 1,300 adversarial tests (95% CI: [0%, 0.28%]), which is unprecedented in AI safety literature.

Our three contributions—theoretical (Lyapunov-stable PA mathematics), empirical (0% ASR validation), and methodological (TELOSCOPE observability)—set a foundation for the reliable deployment of AI in regulated fields.

The route from research to implementation is straightforward: healthcare organizations can use TELOS now for HIPAA compliance, while financial and educational institutions can adapt the framework to meet their regulatory needs.

We encourage the research community to reproduce our findings, apply them to new fields, and collaborate with us in developing mathematically enforceable AI governance. The code is open source (Apache 2.0), the validation protocol is automated, and the need for this work is urgent.

The future of trustworthy AI relies on creating systems that make violations statistically unlikely, rather than settling for imperfect governance.
