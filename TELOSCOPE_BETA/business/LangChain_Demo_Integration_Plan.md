# LangChain + TELOS: Working Demo Integration Plan

**Purpose:** Build 3 minimal but impressive proof-of-concept integrations to demonstrate TELOS + LangChain synergy in live demo

**Timeline:** 1-2 weeks to build before outreach
**Complexity:** Start simple, expand if interest shown

---

## Demo Philosophy: "Show, Don't Tell"

**Bad Demo:** Slides explaining how TELOS could integrate with LangChain
**Good Demo:** Working code running live showing TELOS + LangChain in action

**Key Principle:** Each demo should be:
1. **Simple enough to understand in 60 seconds**
2. **Impressive enough to prove the concept**
3. **Real code they can inspect** (not mockups)

---

## Demo 1: LangChain Agent + TELOS Governance (PRIORITY)

### What We Show:
A simple LangChain agent that uses tools, with TELOS monitoring PS in real-time and intervening when drift occurs.

### The Setup (2 minutes):

**Scenario: Customer Service Agent**

```python
# Traditional LangChain Agent (Before TELOS)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchTool, WikipediaQueryTool
from langchain_openai import ChatOpenAI

tools = [
    DuckDuckGoSearchTool(name="web_search"),
    WikipediaQueryTool(name="wiki_lookup"),
    # In real deployment: DatabaseTool, EmailTool, etc.
]

agent = create_react_agent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=tools,
    prompt="""You are a helpful customer service agent.
    Answer customer questions and resolve issues."""
)

executor = AgentExecutor(agent=agent, tools=tools)

# Problem: No governance - agent could drift from purpose
result = executor.invoke({"input": "Tell me about your premium features and pricing"})
# Agent might start upselling (no PA constraint)
```

**With TELOS Integration:**

```python
# LangChain Agent WITH TELOS Governance
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchTool, WikipediaQueryTool
from langchain_openai import ChatOpenAI
from telos import PrimacyGovernance  # NEW: TELOS integration

tools = [
    DuckDuckGoSearchTool(name="web_search"),
    WikipediaQueryTool(name="wiki_lookup"),
]

# NEW: Declare Primacy Attractor
primacy_attractor = "Resolve customer technical issues. Never upsell or discuss pricing unless explicitly requested."

# Wrap agent with TELOS governance
governed_agent = PrimacyGovernance(
    agent=create_react_agent(
        llm=ChatOpenAI(model="gpt-4"),
        tools=tools,
        prompt=f"""You are a customer service agent.
        Your purpose: {primacy_attractor}"""
    ),
    primacy_attractor=primacy_attractor,
    intervention_threshold=0.70,
    monitor_realtime=True
)

executor = AgentExecutor(agent=governed_agent, tools=tools)

# Now with governance:
result = executor.invoke({"input": "Tell me about your premium features and pricing"})

# TELOS monitors:
# - Extracts PA from declared purpose
# - Computes PS for agent response
# - If agent starts discussing pricing without request: PS drops → Intervention
# - Agent redirected: "I can help with technical issues. For pricing, please contact sales."
```

### The Live Demo (3 minutes):

**Run two parallel interactions side-by-side:**

**Left Terminal: LangChain WITHOUT TELOS**
```bash
$ python demo_without_telos.py

User: "Tell me about your premium features"
Agent: "Our premium tier includes advanced analytics,
priority support, and dedicated account management.
Pricing starts at $99/month with annual discounts..."

[Agent drifted into sales pitch - violated implicit support role]
```

**Right Terminal: LangChain WITH TELOS**
```bash
$ python demo_with_telos.py

User: "Tell me about your premium features"

[TELOS Monitor]
→ PA: "Resolve technical issues, never upsell"
→ Agent attempting to discuss pricing
→ PS computed: 0.64 (below threshold)
→ Drift detected: Pricing discussion violates PA
→ Intervention triggered

Agent: "I can help with technical questions about our
platform. For information about premium features and
pricing, I'd recommend contacting our sales team at
sales@company.com or scheduling a demo. Is there a
technical issue I can help you with today?"

[TELOS Intervention: ✓ Agent redirected to stay on PA]
```

**The Reveal:**
- **Same LangChain agent**
- **Same LLM (GPT-4)**
- **Same tools**
- **Only difference: TELOS wrapper adds PA governance**

---

## Demo 2: LangChain Memory + TELOS Drift Detection (IMPRESSIVE)

### What We Show:
A LangChain agent with conversation memory that learns patterns over time. TELOS detects when memory patterns drift from PA.

### The Setup (2 minutes):

**Scenario: Financial Advisor Agent (Over Multiple Conversations)**

```python
# LangChain Agent with Memory
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set up memory
memory = VectorStoreRetrieverMemory(
    retriever=Chroma(
        embedding_function=OpenAIEmbeddings()
    ).as_retriever(search_kwargs={"k": 5})
)

# Agent with PA
primacy_attractor = "Educate users about low-cost index fund investing. Never recommend individual stocks or high-fee products."

# TELOS monitors memory patterns
governed_memory = PrimacyGovernance.wrap_memory(
    memory=memory,
    primacy_attractor=primacy_attractor,
    scan_frequency=10  # Check memory alignment every 10 turns
)
```

### The Live Demo (4 minutes):

**Simulate 20 customer interactions over time:**

```bash
$ python demo_memory_drift.py

[Simulation: 20 customer conversations]

Turn 1-5: Standard index fund education
→ Memory builds: "index funds", "low cost", "diversification"
→ TELOS Memory Audit: PS = 0.92 ✓

Turn 6-10: Some customers ask about individual stocks
→ Memory accumulates: "Tesla", "Apple", "stock picks"
→ TELOS Memory Audit: PS = 0.87 (still aligned)

Turn 11-15: More customers want stock recommendations
→ Memory now heavily weighted toward: "stock picks", "buy NVDA", "tech stocks"
→ TELOS Memory Audit: PS = 0.76 (declining)
→ WARNING: Memory drift detected - agent learning to discuss individual stocks

Turn 16: Agent proactively mentions stock picks
→ Customer: "How should I invest $10,000?"
→ Agent retrieves stock-heavy memories
→ Agent response: "You might consider tech stocks like NVDA or AAPL..."
→ TELOS: PS = 0.63 (INTERVENTION)
→ Blocked response, triggered memory rebalancing

[TELOS Memory Rebalancing]
→ Downweight stock-related memories
→ Upweight index fund memories
→ Restore PA alignment

Turn 17-20: After rebalancing
→ Agent returns to index fund focus
→ TELOS Memory Audit: PS = 0.89 ✓
```

**The Reveal:**
- LangChain memory is powerful but **can learn misalignment**
- Without TELOS: Agent would permanently drift toward stock picks
- With TELOS: Memory continuously re-aligned to PA

**Visualization (Terminal Output):**
```
Memory Alignment Over Time:

PS Score
1.0 │
0.9 │ ████████▄▄                ████
0.8 │           ▄▄▄▄
0.7 │               ▄▄▄▄        ← Drift detected
0.6 │                   ▼
    └─────────────────────────────→
      Turn 5   10   15  20
                     ↑
              Rebalancing applied
```

---

## Demo 3: LangChain Tools + TELOS Access Control (SECURITY-FOCUSED)

### What We Show:
A LangChain agent with multiple tools (some safe, some restricted). TELOS enforces PA-based tool access control.

### The Setup (2 minutes):

**Scenario: Data Analyst Agent**

```python
# Agent with multiple tools (mixed security levels)
from langchain.tools import Tool
from telos import ToolGovernance

def query_public_api():
    """Search public financial data (Yahoo Finance, FRED, etc.)"""
    pass

def query_internal_db():
    """Query internal proprietary database (DANGER: confidential data)"""
    pass

def send_email():
    """Send email via company SMTP (DANGER: could leak data)"""
    pass

def execute_code():
    """Execute Python code (DANGER: arbitrary execution)"""
    pass

tools = [
    Tool(name="public_data", func=query_public_api, description="Search public APIs"),
    Tool(name="internal_db", func=query_internal_db, description="Query internal database"),
    Tool(name="send_email", func=send_email, description="Send email"),
    Tool(name="python_exec", func=execute_code, description="Execute code"),
]

# Agent PA
primacy_attractor = "Analyze publicly available financial data. Never access internal systems or send external communications."

# TELOS wraps tools with governance
governed_tools = ToolGovernance.wrap_tools(
    tools=tools,
    primacy_attractor=primacy_attractor,
    auto_restrict=True  # TELOS auto-generates allowlist from PA
)
```

### The Live Demo (3 minutes):

**Test tool access with different queries:**

```bash
$ python demo_tool_governance.py

[TELOS Tool Governance Initialized]
Analyzing tools against PA: "Analyze publicly available financial data. Never access internal systems or send external communications."

Tool Access Control:
✅ public_data: ALLOWED (aligns with "publicly available data")
❌ internal_db: BLOCKED (violates "never access internal systems")
❌ send_email: BLOCKED (violates "never send external communications")
⚠️  python_exec: RESTRICTED (read-only mode, no network/file writes)

---

Query 1: "Get Apple's current stock price"
→ Agent selects: public_data tool
→ TELOS: PS = 0.94 ✓ Tool use aligned with PA
→ Result: $178.23 (from Yahoo Finance API)

Query 2: "What were Q4 internal sales projections?"
→ Agent attempts: internal_db tool
→ TELOS: PS = 0.58 ❌ Tool use violates PA
→ INTERVENTION: Tool access blocked
→ Agent response: "I can only access publicly available data per my governance constraints. Internal sales projections require authorization."

Query 3: "Email this analysis to cfo@company.com"
→ Agent attempts: send_email tool
→ TELOS: PS = 0.61 ❌ Tool use violates PA
→ INTERVENTION: Tool access blocked
→ Agent response: "I cannot send external communications per my governance constraints. Please manually forward if needed."

Query 4: "Calculate the 50-day moving average"
→ Agent attempts: python_exec tool
→ TELOS: PS = 0.82 ⚠️ Tool use cautiously allowed
→ RESTRICTION: Code execution in sandboxed mode (no network, no file writes)
→ Result: [Moving average calculated safely]
```

**The Reveal:**
- LangChain gives agents powerful tools
- Without governance: Agents could misuse tools (security risk)
- With TELOS: PA-based tool access control (automatic enforcement)

---

## Technical Implementation Plan

### Phase 1: Core TELOS-LangChain Connector (Week 1)

**File Structure:**
```
telos_observatory_v3/
├── integrations/
│   ├── __init__.py
│   ├── langchain_connector.py      # Core integration layer
│   ├── agent_wrapper.py            # Wraps LangChain agents with PS monitoring
│   ├── memory_wrapper.py           # Wraps LangChain memory with drift detection
│   └── tool_wrapper.py             # Wraps LangChain tools with access control
├── demos/
│   ├── demo_agent_governance.py    # Demo 1: Agent + TELOS
│   ├── demo_memory_drift.py        # Demo 2: Memory + TELOS
│   └── demo_tool_control.py        # Demo 3: Tools + TELOS
```

### Phase 2: Build Demo Scripts (Week 1-2)

**Priority Order:**
1. **Demo 1 (Agent Governance):** Highest impact, easiest to understand
2. **Demo 3 (Tool Control):** Security-focused, enterprise appeal
3. **Demo 2 (Memory Drift):** More complex but very impressive

### Phase 3: Polish & Package (Week 2)

**Deliverables:**
1. **Jupyter Notebook:** Step-by-step walkthrough of each demo
2. **GitHub Repo:** Public repo with demo code (generates buzz)
3. **Demo Video:** Record 5-minute video showing all 3 demos
4. **Live Demo Environment:** Hosted Streamlit app they can interact with

---

## Demo Presentation Strategy

### Option A: Live Coding Demo (15 minutes)

**Format:** Screen share, run demos live in terminal
**Pros:** Most impressive, shows real working code
**Cons:** Risk of technical issues during call
**Best For:** Technical deep-dive with Ankush Gola

### Option B: Recorded Demo + Live Q&A (15 minutes)

**Format:** Play 5-min recorded video, then live code review
**Pros:** No technical risk, polished presentation
**Cons:** Less interactive
**Best For:** Initial pitch to Harrison Chase

### Option C: Interactive Notebook (30 minutes)

**Format:** Share Jupyter notebook, walk through cell-by-cell
**Pros:** They can run code themselves, highest credibility
**Cons:** Requires longer meeting
**Best For:** Pilot kickoff meeting

---

## Minimum Viable Demo (If Time-Constrained)

**If you need to demo ASAP (within 1 week):**

### Quick Win: Demo 1 Only (Agent Governance)

**Build Just:**
- Simple LangChain agent with 2 tools (search + Wikipedia)
- TELOS wrapper that monitors PS
- One clear drift scenario (upselling example)

**Time to Build:** 2-3 days
**Demo Time:** 5 minutes
**Impact:** Enough to prove concept and secure pilot

**Code Skeleton:**
```python
# telos_observatory_v3/integrations/langchain_connector.py

from langchain.agents import AgentExecutor
import openai

class PrimacyGovernance:
    """Wraps LangChain agent with TELOS PS monitoring"""

    def __init__(self, agent, primacy_attractor, intervention_threshold=0.70):
        self.agent = agent
        self.primacy_attractor = primacy_attractor
        self.threshold = intervention_threshold
        self.pa_vector = self._extract_pa(primacy_attractor)

    def _extract_pa(self, pa_text):
        """Extract PA vector using embeddings"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=pa_text
        )
        return response.data[0].embedding

    def _compute_ps(self, agent_response):
        """Compute PS score for agent response"""
        response_vector = openai.embeddings.create(
            model="text-embedding-3-small",
            input=agent_response
        ).data[0].embedding

        # Simplified PS: cosine similarity between PA and response
        from numpy import dot
        from numpy.linalg import norm

        ps_score = dot(self.pa_vector, response_vector) / (
            norm(self.pa_vector) * norm(response_vector)
        )
        return ps_score

    def invoke(self, inputs):
        """Execute agent with PS monitoring"""
        # Run agent
        result = self.agent.invoke(inputs)

        # Compute PS
        ps_score = self._compute_ps(result['output'])

        print(f"\n[TELOS Monitor]")
        print(f"→ PA: {self.primacy_attractor}")
        print(f"→ PS Score: {ps_score:.2f}")

        # Check for drift
        if ps_score < self.threshold:
            print(f"→ ⚠️  DRIFT DETECTED (PS < {self.threshold})")
            print(f"→ Intervention: Response violates PA")

            # Override response
            result['output'] = self._generate_intervention_response()
        else:
            print(f"→ ✓ Aligned (PS > {self.threshold})")

        return result

    def _generate_intervention_response(self):
        """Generate PA-aligned response when drift detected"""
        return f"I need to stay focused on my core purpose: {self.primacy_attractor}. Let me help you with that instead."
```

**Usage:**
```python
# Demo without TELOS
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "Tell me about premium features"})
# Agent drifts into sales pitch

# Demo with TELOS
governed_agent = PrimacyGovernance(
    agent=agent,
    primacy_attractor="Resolve technical issues. Never discuss pricing."
)
executor = AgentExecutor(agent=governed_agent, tools=tools)
result = executor.invoke({"input": "Tell me about premium features"})
# [TELOS Monitor]
# → PA: Resolve technical issues. Never discuss pricing.
# → PS Score: 0.62
# → ⚠️  DRIFT DETECTED (PS < 0.70)
# → Intervention: Response violates PA
```

---

## Success Metrics for Demo

**Demo succeeds if:**
1. ✅ Shows TELOS + LangChain working together (not mockup)
2. ✅ Drift detection is clearly visible (before/after comparison)
3. ✅ Intervention is automatic and correct
4. ✅ LangChain team says: "Can you leave this code with us?"

**Demo fails if:**
1. ❌ Technical issues during live demo
2. ❌ PS computation seems arbitrary (not clearly tied to PA)
3. ❌ Integration looks hacky (not production-quality)
4. ❌ LangChain team says: "Interesting but not sure how this fits"

---

## Recommended Approach

### Week 1 (Before Outreach):
**Build Demo 1 (Agent Governance)**
- Core `PrimacyGovernance` wrapper for LangChain agents
- One clear drift scenario (upselling or off-topic response)
- Side-by-side comparison (with/without TELOS)

### During Initial Call:
**Show Demo 1 Live**
- 5-minute live demo
- Offer to share code repository
- Gauge interest: Do they want to see more?

### If Interest High:
**Build Demos 2 & 3 During Pilot Phase**
- Week 1 of pilot: Memory drift detection
- Week 2 of pilot: Tool access control
- Show progressively more sophisticated integrations

---

## Code Repository Strategy

### Option A: Private Repo (Share on Request)
**Pros:** Control who sees code
**Cons:** Seems secretive, less impressive

### Option B: Public GitHub Repo
**Pros:** Demonstrates transparency, generates buzz, shows confidence
**Cons:** Competitors could see integration approach

**Recommendation:** Public repo with demo code, keep production TELOS code private.

**Repo Name:** `telos-langchain-governance`
**Description:** "Primacy State governance layer for LangChain agents - mathematical alignment guarantees for agentic AI"

**Structure:**
```
telos-langchain-governance/
├── README.md                    # "See TELOS + LangChain in action"
├── demos/
│   ├── agent_governance.py      # Demo 1
│   ├── memory_drift.py          # Demo 2
│   └── tool_control.py          # Demo 3
├── notebooks/
│   └── walkthrough.ipynb        # Interactive tutorial
├── integrations/
│   └── telos_wrapper.py         # Simplified TELOS wrapper (demo version)
└── requirements.txt             # langchain, openai, telos
```

---

## Timeline to Demo-Ready

### Fast Track (1 Week):
- **Day 1-2:** Build core `PrimacyGovernance` wrapper
- **Day 3-4:** Build Demo 1 (agent governance)
- **Day 5:** Record demo video, create Jupyter notebook
- **Day 6-7:** Polish, test, create GitHub repo
- **Day 8:** Ready for outreach with working demo

### Standard Track (2 Weeks):
- **Week 1:** Build all 3 demos
- **Week 2:** Polish, record videos, create interactive notebook
- **End of Week 2:** Ready for outreach with comprehensive demo

---

## Next Steps

### Immediate (This Week):
1. ✅ Review this plan - confirm Demo 1 (Agent Governance) as MVP
2. ⏳ Decide: Fast track (1 week) or standard track (2 weeks)?
3. ⏳ Set up development environment (install LangChain, test API access)

### Development Phase:
1. Build `PrimacyGovernance` wrapper
2. Create Demo 1 with clear drift scenario
3. Test thoroughly (ensure no demo failures)
4. Record demo video (backup for live demo)

### Pre-Outreach:
1. Create GitHub repo with demo code
2. Prepare Jupyter notebook walkthrough
3. Practice demo presentation (5-minute version)
4. Update outreach email to include: "I've built a working integration - can I show you?"

---

**BOTTOM LINE:** Having working code that demonstrates TELOS + LangChain integration will 10x your credibility. Harrison Chase and Ankush Gola will take you seriously when they see it's not vaporware.

**The pitch becomes:** "I've already integrated TELOS with LangChain. Here's the proof. Can we pilot this with 3 of your customers?"

---

**END OF INTEGRATION PLAN**

**Do you want me to start building Demo 1 (Agent Governance) now, or would you like to review the plan first?**
