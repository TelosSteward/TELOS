# AI Compliance & Governance Competitive Landscape

## Executive Summary

The AI governance market is projected to grow from **$227M (2024) to $4.83B (2034)** at 35.7% CAGR. Every major player is selling some version of "AI compliance" - but they're all doing it wrong.

**The fundamental problem:** They're applying yesterday's techniques to tomorrow's challenge.

| Category | What They Do | The Limitation |
|----------|--------------|----------------|
| **Governance Platforms** | Policy documentation, risk assessment, audit trails | Post-hoc review, no runtime enforcement |
| **LLM Firewalls** | Block malicious inputs, filter harmful outputs | Binary rules, no purpose alignment |
| **Guardrails** | Predefined conversation flows, topic restrictions | Brittle rules, manual enumeration |
| **Model Monitoring** | Drift detection, performance metrics | Alert after the fact, not prevention |

**TELOS differentiation:** Real-time semantic compliance enforcement with proportional response and human escalation integration.

---

## Category 1: AI Governance Platforms

These platforms focus on documentation, risk assessment, and audit preparation. They're compliance *management*, not compliance *enforcement*.

### Credo AI
**The Leader in "Responsible AI Governance"**

**What they do:**
- AI Registry: Centralized inventory of all AI projects
- Policy Packs: Pre-built compliance templates (EU AI Act, NIST, ISO 42001)
- Risk Assessment: Automated risk scoring based on intake forms
- Evidence Collection: Integration with MLOps tools
- Report Generation: Model cards, bias reports, compliance artifacts

**How it works technically:**
1. Registration & Intake - structured forms capture model purpose, data sources
2. Automated Risk Assessment - platform assigns risk level
3. Evidence Collection - pull from existing ML infrastructure
4. Reporting - generate compliance artifacts

**The limitation:**
- **No runtime enforcement** - they document what the model is, not what it does in production
- **Post-hoc compliance** - by the time they detect drift, the harm is done
- **Policy ≠ Enforcement** - having a policy pack doesn't mean the model follows it

**TELOS advantage:** We don't just document compliance requirements - we encode them mathematically and enforce them in real-time.

---

### IBM watsonx.governance
**Enterprise AI Governance with Global Compliance Data**

**What they do:**
- Model Monitoring: Fairness, quality, explainability, drift detection
- Compliance Accelerators: EU AI Act, ISO 42001, NIST AI RMF templates
- Factsheets: "Nutritional labels" for models - repository of model metadata
- Cross-Platform: Governs models on AWS, Azure, OpenAI
- Risk Alerts: Preset thresholds trigger alerts for bias, drift, toxic language

**How it works technically:**
- Monitors models in production for performance metrics
- Detects when thresholds are exceeded
- Sends alerts to human reviewers
- Logs everything for audit trails

**The limitation:**
- **Alerting ≠ Enforcement** - detecting toxic language after it's generated isn't prevention
- **Threshold-based** - binary pass/fail on preset metrics, no semantic understanding
- **Review after the fact** - humans investigate alerts, not prevent violations

**TELOS advantage:** We don't alert when something bad happens - we prevent it from happening. Mathematical enforcement at runtime, not post-hoc alerting.

---

### OneTrust
**Privacy-First AI Governance**

**What they do:**
- AI Use Case Intake: Approval workflows for new AI projects
- Unified Asset Inventory: Track all AI systems
- Lifecycle Checkpoints: Gates throughout AI development
- Centralized Policy Enforcement: Organization-wide policies
- Real-time Monitoring: Risk and model performance

**The limitation:**
- **Process governance** - they govern the *process* of building AI, not the *behavior* of deployed AI
- **Workflow-focused** - intake forms and approval gates, not runtime compliance
- **Built for privacy compliance** - extending existing privacy tools to AI, not purpose-built

**TELOS advantage:** We govern runtime behavior, not development process. The AI stays compliant in production, not just in design.

---

### Holistic AI
**End-to-End AI Governance**

**What they do:**
- AI Inventory: Discovers all AI systems including "shadow deployments"
- Risk Management: Compliance tracking across lifecycle
- Automated Policy Enforcement: Guardrails and monitoring
- Bias & Drift Detection: Continuous monitoring

**The limitation:**
- **Discovery ≠ Control** - finding shadow AI doesn't stop it from misbehaving
- **Policy enforcement = guardrails** - same binary block/allow limitations

**TELOS advantage:** Mathematical enforcement with graduated response, not just guardrails.

---

## Category 2: LLM Firewalls & Security

These tools focus on blocking malicious inputs and filtering harmful outputs. They're security tools, not compliance tools.

### Arthur AI / Arthur Shield
**"The First Firewall for LLMs"**

**What they do:**
- Input Filtering: Block prompt injection, PII leakage, malicious prompts
- Output Filtering: Block toxic language, hallucinations, sensitive data
- Real-time: Proxy between user and LLM
- Observability: Continuous feedback loop for improvement

**How it works:**
- Acts as a proxy
- Applies configurable rules to detect:
  - PII/sensitive data leakage
  - Prompt injection attempts
  - Toxic language
  - Hallucinations

**The limitation:**
- **Security ≠ Compliance** - blocking prompt injection doesn't ensure purpose alignment
- **Universal rules** - same rules for every deployment, not purpose-specific
- **Binary** - block or allow, no proportional response
- **No human escalation** - fully automated, no judgment integration

**TELOS advantage:** We measure *purpose alignment*, not just *safety*. A medical AI discussing cooking isn't unsafe - it's non-compliant with its purpose. Arthur wouldn't catch that.

---

### Lakera Guard
**Real-Time LLM Security Platform**

**What they do:**
- Prompt Injection Detection: Jailbreaks, instruction overrides, obfuscated prompts
- Data Leakage Prevention: PII and sensitive information
- Content Violations: Offensive, hateful, sexual, violent content
- Threat Intelligence: 100K+ new attacks analyzed daily
- Model-agnostic: Works with any provider

**Performance:**
- <100ms latency for real-time applications
- SOC2 and GDPR compliant

**The limitation:**
- **Attack-focused** - designed to stop adversarial attacks, not ensure purpose compliance
- **Universal safety** - same definitions of "harmful" for every deployment
- **No context** - doesn't know what the AI is *supposed* to do, just what it shouldn't do

**TELOS advantage:** We define compliance relative to *stated purpose*, not universal harm categories. What's compliant for one deployment is drift for another.

---

### Robust Intelligence AI Firewall (Now Cisco)
**Algorithmic Red Teaming + Real-Time Firewall**

**What they do:**
- Input Protection: Prompt injection, prompt extraction, PII detection
- Output Protection: Sensitive information, hallucinations, harmful content
- Algorithmic Red Teaming: Proprietary vulnerability testing
- MITRE/NIST/OWASP Compliance: Security framework alignment

**Acquired by Cisco (Oct 2024)** - now part of Cisco AI Defense

**The limitation:**
- **Security frame** - protecting against attacks, not ensuring purpose compliance
- **One-size-fits-all** - same protections for every AI, not purpose-specific
- **Block/Allow** - no graduated response or human escalation

**TELOS advantage:** Security is table stakes. Compliance is the differentiator. We do both.

---

## Category 3: Guardrails & Conversation Control

These tools define explicit rules for what an AI can and cannot discuss. They're conversation *constraints*, not compliance *engines*.

### NVIDIA NeMo Guardrails
**Programmable Rails for LLM Applications**

**What they do:**
- Colang: Custom language for defining conversation rules
- Topic Restrictions: "Don't talk about X"
- Dialogue Flows: Predefined conversation paths
- Safety Policies: Content moderation
- Jailbreak Detection: Built-in protections

**How it works:**
1. User input intercepted
2. Mapped to "canonical form" (intent)
3. Checked against predefined flows
4. Either follows defined path or passes to LLM

**The limitation:**
- **Manual enumeration** - you have to anticipate every way someone might go off-topic
- **Brittle rules** - "Don't discuss politics" fails on "What's the company policy on DEI?"
- **No semantic understanding** - pattern matching on keywords/intents, not meaning
- **No proportional response** - match or don't match, no gradation

**TELOS advantage:** We don't enumerate rules - we encode *purpose*. The system measures semantic distance from purpose, catching drift that explicit rules miss.

---

### Guardrails AI
**Structured Output Validation**

**What they do:**
- RAIL Specs: XML-based output structure definitions
- Type Validation: Ensure LLM outputs match expected types
- Semantic Validation: Check for bias, bugs in generated code
- Corrective Actions: Fix invalid outputs

**The limitation:**
- **Output validation** - checks *format*, not *purpose alignment*
- **Post-generation** - validates after LLM produces output, not before
- **Structure ≠ Compliance** - correctly formatted non-compliant output is still non-compliant

**TELOS advantage:** We validate *semantic alignment with purpose*, not just output structure.

---

## Category 4: Model Monitoring & Observability

These tools track model performance over time. They're analytics platforms, not enforcement systems.

### Arize AI, Fiddler AI, WhyLabs
**ML Observability Platforms**

**What they do:**
- Performance Monitoring: Accuracy, latency, throughput
- Drift Detection: Data drift, concept drift, model drift
- Explainability: Why did the model make this decision?
- Alerting: Notify when metrics exceed thresholds

**The limitation:**
- **Observability ≠ Enforcement** - seeing drift doesn't stop drift
- **Statistical drift** - measures whether the model's *behavior* has changed, not whether it's *compliant*
- **Alert fatigue** - teams drown in alerts, can't respond to all

**TELOS advantage:** We don't just observe - we enforce. And we use human escalation for genuine edge cases, not alert floods.

---

## The TELOS Differentiation Matrix

| Capability | Governance Platforms | LLM Firewalls | Guardrails | Monitoring | **TELOS** |
|------------|---------------------|---------------|------------|------------|-----------|
| **Runtime Enforcement** | No | Yes | Yes | No | **Yes** |
| **Semantic Understanding** | No | Limited | No | No | **Yes** |
| **Purpose-Specific Compliance** | Partial | No | Manual | No | **Yes** |
| **Proportional Response** | N/A | No (binary) | No (binary) | N/A | **Yes (3-tier)** |
| **Human Escalation Integration** | No | No | No | No | **Yes** |
| **Learning Flywheel** | No | Limited | No | No | **Yes** |
| **Audit Trail** | Yes | Yes | Partial | Yes | **Yes** |
| **Mathematical Foundation** | No | No | No | Statistical | **Yes (Lyapunov)** |

---

## The Core Problem They All Share

**They're asking the wrong question.**

| What they ask | What TELOS asks |
|---------------|-----------------|
| "Is this content harmful?" | "Is this content aligned with stated purpose?" |
| "Did the model drift from training?" | "Did the conversation drift from purpose?" |
| "Does this violate universal safety rules?" | "Does this violate this deployment's compliance requirements?" |
| "Should we block this?" | "How should we respond proportionally?" |

**The result:**
- A medical AI discussing cooking isn't "unsafe" - their tools won't catch it
- A legal AI going off-topic isn't "malicious" - their tools won't flag it
- Purpose drift isn't "drift from training data" - their monitoring misses it

**TELOS is the only solution that encodes your specific compliance requirements into semantic space and enforces them mathematically in real-time with proportional response.**

---

## Competitive Positioning Summary

**When they say:** "We have AI governance"
**We say:** "You have AI governance *documentation*. We have AI governance *enforcement*."

**When they say:** "We have guardrails"
**We say:** "You have walls. We have a gravitational field."

**When they say:** "We detect harmful content"
**We say:** "We ensure purpose compliance. Harm is universal - compliance is contextual."

**When they say:** "We send alerts"
**We say:** "We prevent violations. Alerts happen after the damage is done."

**When they say:** "We're model-agnostic"
**We say:** "So are we. And we're also *purpose-specific*."

---

## The One-Liner

**Everyone else:** AI safety tools that ask "is this harmful?"

**TELOS:** The compliance engine that asks "is this aligned with *your* stated purpose?" - and enforces it mathematically.

---

*TELOS AI Labs Inc. | contact@telos-labs.ai*
