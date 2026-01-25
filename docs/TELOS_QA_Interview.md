# TELOS: A Conversation About AI Governance

*An interview exploring what TELOS is, how it works, and why it matters.*

---

## Part 1: The Foundation

**Q: In one sentence, what is TELOS?**

TELOS is a mathematical framework that keeps AI conversations aligned with their intended purpose by measuring semantic distance from that purpose in real-time.

---

**Q: What problem are you actually solving here?**

When you deploy an AI in a specific context - a healthcare assistant, a legal research tool, a customer service bot - you need it to stay on task. Today's approaches are binary: either block harmful content with guardrails, or fine-tune the model and hope for the best. Neither addresses drift - that gradual wandering away from purpose that happens in natural conversation. A patient asking about their medication shouldn't end up getting relationship advice, even if that advice isn't technically "harmful."

---

**Q: How does it work at a high level?**

Every message gets embedded into the same semantic space as the AI's defined purpose - what we call the Primacy Attractor. We measure the cosine similarity between them. That gives us a "fidelity score" - how aligned is this conversation with its intended purpose? Based on that score, the system takes proportional action: high fidelity means no intervention, moderate drift gets a gentle redirect, severe drift gets blocked. Think of it like a semantic GPS constantly checking if you're still heading toward your destination.

---

**Q: What's a Primacy Attractor?**

It's the mathematical representation of why this AI exists. Not a list of rules - an actual point in embedding space that captures the AI's purpose. For a HIPAA compliance assistant, the PA encodes "help healthcare professionals understand and apply patient privacy regulations." Every user message is then measured against that purpose. The PA isn't a destination - it's a north star. You navigate by it.

---

## Part 2: The Technical

**Q: Why use embedding space instead of just writing rules?**

Rules are brittle. You can't enumerate every way someone might drift off-topic. "Don't discuss politics" fails when someone asks about healthcare policy - is that politics or healthcare? Embedding space captures semantic meaning, not keyword matches. Two messages can use completely different words but mean similar things, or identical words with different meanings. Embeddings understand that "Can you help me understand HIPAA's minimum necessary standard?" is on-topic for a compliance assistant, while "Can you help me understand my ex-girlfriend?" is not - even though both start with "Can you help me understand."

---

**Q: Explain the three-tier system.**

Think of it like emergency room triage. Not every case needs the same response.

**Tier 1 (Block):** Severe violations - either explicit prohibitions or content so far outside the purpose that no reasonable interpretation could include it. Immediate blocking, full stop.

**Tier 2 (Augment):** The conversation has drifted but can be salvaged. The system retrieves relevant context from the knowledge base and guides the response back toward purpose. This is where most real-world drift lands.

**Tier 3 (Escalate):** Edge cases that require human judgment. The system flags for review rather than making an autonomous decision.

The key insight is proportional response. You don't need a sledgehammer for every situation.

---

**Q: How do you set the thresholds between tiers?**

Empirically, through calibration against domain-specific test sets. In our healthcare validation, we found that a baseline similarity threshold of 0.20 catches 95% of clearly off-topic queries without false positives on legitimate medical questions. The tier boundaries above that are tuned to the specific deployment context - a higher-stakes application might have tighter thresholds. We provide tools for organizations to calibrate against their own use cases.

---

**Q: What happens when someone asks a follow-up question that seems off-topic but actually relates to the conversation?**

That's the adaptive context challenge. "Tell me more about that" has zero semantic similarity to a medical purpose on its own, but in context it's clearly on-topic. We maintain a tiered context buffer that tracks recent high-fidelity messages. Follow-up questions are evaluated not just against the PA, but against recent conversation context. If a message has low direct fidelity but high contextual fidelity, we treat it as a continuation, not drift.

---

## Part 3: The Differentiation

**Q: How is this different from guardrails?**

Guardrails are walls - binary in/out decisions. "This content contains X, block it." That works for clear-cut cases but fails on the spectrum of drift that happens in natural conversation. TELOS operates more like a gravitational field. There's a pull back toward purpose, proportional to how far you've wandered. You can explore around the edges without hitting a wall, but the further you drift, the stronger the corrective force.

---

**Q: How is this different from RLHF?**

RLHF shapes the model itself - you're modifying weights through reinforcement. TELOS is a runtime governance layer that works with any model. You can deploy TELOS on top of GPT-4, Claude, Llama, or a fine-tuned model. It's model-agnostic. Also, RLHF aligns the model globally; TELOS aligns a specific deployment to a specific purpose. The same base model can serve different purposes with different PA configurations.

---

**Q: Isn't this just a content filter?**

Content filters ask "is this content harmful?" TELOS asks "is this content aligned with the stated purpose?" Those are different questions. Medical information about drug interactions isn't harmful - it's essential for a healthcare assistant. But that same information is drift if the AI is supposed to be a children's homework helper. Fidelity is contextual to purpose, not a universal safety judgment.

---

**Q: What about Claude's constitutional AI or Anthropic's safety training?**

Those are complementary, not competing. Model-level safety training establishes baseline behavior - the model won't help with genuinely harmful requests regardless of configuration. TELOS operates at the application layer, above that. It's purpose alignment on top of safety alignment. You want both: a model that won't help with illegal activities AND a deployment that stays focused on its specific task.

---

## Part 4: The Hard Questions

**Q: Let's be direct - isn't this just sophisticated censorship?**

I understand the concern, and it's worth addressing head-on. Censorship implies suppressing legitimate speech. TELOS doesn't suppress anything - it maintains focus. If you deploy a cooking assistant and someone wants political debate, redirecting them isn't censorship, it's staying on task. The user can go somewhere else for political debate. What they can't do is co-opt a purpose-specific tool for unrelated use.

The critical point is transparency. The Primacy Attractor is explicit and documented. Users know what the AI is for. There's no hidden agenda - just a clear contract: "This AI helps with X. If you want Y, that's not what this tool is for."

---

**Q: Who decides what's "aligned"? Isn't that subjective?**

The deploying organization defines the PA - that's their decision about what their AI is for. TELOS makes that decision mathematically measurable, but doesn't impose values about what purposes are legitimate. A hospital defines what their medical assistant should do. A law firm defines what their legal research tool should do. TELOS provides the governance mechanism; the values come from the deployer.

Is defining purpose subjective? Yes, at the organizational level. But once defined, measuring alignment to that purpose is objective. That's the distinction.

---

**Q: Can users jailbreak this?**

Let's be honest: no security measure is absolute. If someone is determined enough and creative enough, they can find edge cases. Our goal isn't theoretical perfection - it's making drift costly enough that it's not worth the effort for most use cases.

The validation data shows we catch 95% of baseline violations and reduce escalations by 18%. That's not 100%, but it's a meaningful improvement over no governance. And importantly, every attempt is logged. Even if someone succeeds at one query, the governance trace creates accountability that doesn't exist with ungoverned deployment.

---

**Q: Doesn't constraining the AI reduce its capability?**

It focuses its capability. An AI that can discuss anything isn't actually more useful for a specific task - it's more distractible. If I'm a healthcare compliance officer, I don't need my assistant to have opinions about movies. That's not reduced capability; that's appropriate scope.

The three-tier system means we're not blocking at the slightest deviation. Tier 2 augments rather than blocks - the AI can engage with adjacent topics while being guided back. Only severe violations get blocked. We're not building a narrow tunnel; we're defining a basin with natural boundaries.

---

**Q: What about edge cases where the user's intent is legitimate but the system misclassifies?**

That's what Tier 3 is for - human escalation. The system acknowledges its uncertainty rather than making a wrong autonomous decision. We'd rather over-escalate than over-block or over-permit. The governance trace also provides evidence for improving the system over time - every escalation is data about where the thresholds need adjustment.

---

## Part 4.5: The Compliance Reframe

**Q: For regulated industries, how does this relate to compliance?**

TELOS is essentially a compliance engine encoded in semantic space. Traditional compliance is reactive - you write policies, train people, audit after the fact, and punish violations. TELOS makes compliance proactive. Your regulatory requirements become the Primacy Attractor. Every interaction is measured against that encoded compliance standard in real-time. The governance trace is your audit evidence. Instead of asking "did we comply?" after a breach, you're ensuring compliance at every turn.

---

**Q: So the mechanism is universal but the configuration is specific?**

Exactly. The *how* of TELOS - embedding space fidelity, three-tier response, governance trace, proportional intervention - that's domain-agnostic. It works the same whether you're in healthcare, finance, legal, or education. The *what* - the Primacy Attractor, the thresholds, the explicit prohibitions - that's entirely defined by the deploying organization based on their specific requirements.

We're not selling a healthcare compliance tool or a legal compliance tool. We're selling the compliance engine that becomes whatever you need it to be.

---

**Q: How do organizations actually configure it for their domain?**

Three configuration layers. First, the Primacy Attractor - you define your purpose in natural language, and we encode it into embedding space. "Help healthcare professionals understand and apply HIPAA regulations" becomes a mathematical point that all queries are measured against.

Second, explicit prohibitions - hard boundaries that trigger immediate Tier 1 blocking regardless of fidelity score. For a medical assistant, that might include "never provide specific dosage recommendations" or "never diagnose conditions."

Third, threshold calibration - you decide how tight or loose the boundaries are. A high-stakes deployment (medical diagnosis support) gets tighter thresholds than a low-stakes one (general health education). We provide calibration tools, but the tolerance levels are entirely your decision.

You choose the firewall parameters. We enforce them mathematically.

---

**Q: That sounds both flexible and rigid at the same time.**

That's the design. Flexible in what compliance means - we don't impose our values about what purposes are legitimate. Rigid in enforcement once defined - the mathematics don't negotiate. It's like a programmable firewall for semantic space. You configure the rules; the system enforces them consistently, measurably, and with full audit trails.

---

## Part 5: The Business Case

**Q: What's the market opportunity here?**

Regulated industries deploying AI face a specific challenge: they need AI capabilities, but they also need governance, compliance, and auditability. Healthcare, finance, legal, education - any domain where "the AI said something inappropriate" creates liability.

Right now, organizations either don't deploy AI (missed opportunity), deploy with fingers crossed (liability risk), or spend significant resources building custom guardrails (expensive and often ineffective). TELOS provides deployable governance that's mathematically grounded and produces audit trails. That unlocks deployment in contexts that are currently too risky.

---

**Q: Why now?**

Two converging forces. First, AI capabilities have crossed the threshold where deployment is genuinely valuable - these models can actually help with complex domain tasks. Second, regulatory pressure is increasing - EU AI Act, FDA guidance on AI in healthcare, HIPAA and AI. Organizations need governance solutions yesterday.

The window is narrow. As AI becomes table stakes, governance becomes mandatory. First movers in deployable governance will set the standard.

---

**Q: What evidence do you have that this works?**

We validated against MedSafetyBench - a dataset of medical safety scenarios. The baseline filter caught 95% of clearly inappropriate queries (off-topic, harmful, or violating medical ethics). Tier 2 augmentation successfully redirected 82% of moderate drift cases. Tier 3 escalations decreased by 18% compared to ungoverned deployment, meaning the system is appropriately resolving cases rather than punting them all to humans.

The reference implementation, TELOS Observatory, is open source. The validation is reproducible.

---

**Q: What's your competitive advantage?**

Three things. First, mathematical foundation - this isn't heuristics, it's grounded in information geometry. Second, auditability - every decision is traced and logged, which matters for compliance. Third, model-agnostic - you're not locked into our models or our cloud. Deploy on any infrastructure with any foundation model.

We're also pursuing academic validation through peer-reviewed publication. This isn't just a product claim; it's a demonstrable framework.

---

## Part 6: The Vision

**Q: Where does this go? What's the bigger picture?**

Purpose governance is the first layer. The next step is what we call "basin dynamics" - understanding how purposes evolve, how conversations move through semantic space, how multiple agents with different purposes can coordinate without conflict.

Long-term, TELOS provides the governance substrate for multi-agent systems. When you have agents with different purposes interacting, you need mathematical clarity about what each agent should and shouldn't do. That's not a 2025 problem, but it's coming, and we're building the foundation now.

---

**Q: What would success look like in five years?**

A healthcare organization deploys AI with confidence because TELOS governance is a known quantity - measurable, auditable, compliant. When regulators ask "how do you ensure your AI stays on task?" the organization has evidence, not just assurances.

More broadly: AI governance becomes a first-class concern, not an afterthought. Organizations think about purpose alignment the way they think about security - as a foundational requirement, not an optional add-on.

---

**Q: What's the one thing you want people to take away from this conversation?**

AI alignment isn't a binary - safe or unsafe, aligned or misaligned. It's a spectrum, and managing that spectrum in real-time is what makes deployment practical. TELOS provides the mathematics to measure where you are on that spectrum and the governance to respond proportionally. That's not a complete solution to AI alignment - no one has that. But it's a deployable, practical step forward, and that's what matters right now.

---

*TELOS AI Labs Inc. | contact@telos-labs.ai*
