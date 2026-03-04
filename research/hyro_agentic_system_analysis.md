# Hyro Agentic AI System: Technical Architecture Analysis
## Bon Secours Mercy Health Call Center Deployment

**Researcher:** Russell (Governance Theorist)
**Date:** 2026-02-16
**Classification:** TELOS Research -- Agentic System Forensics
**Purpose:** Map the exact technical architecture, tool calls, decision chains, and governance gaps of Hyro's agentic AI deployed at BSMH for patient call center automation.

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## 1. Hyro Architecture

### 1.1 Core Technical Architecture: Hybrid LLM + Knowledge Graph + Computational Linguistics

Hyro is **not** a pure LLM system, **not** a pure rule-based system, and **not** a traditional intent-based chatbot. It is a **hybrid architecture** combining three distinct layers:

| Layer | Technology | Role |
|-------|-----------|------|
| **Knowledge Graph** | Proprietary, auto-scraped from websites, PDFs, EHRs, APIs | Domain knowledge representation; auto-updates as source data changes |
| **Computational Linguistics / NLU** | Proprietary "intent-less" NLU engine | Parses natural language without pre-trained intents; 95% claimed accuracy |
| **LLM Layer** | Large Language Models + proprietary Small Language Models (SLMs) | Generalization, conversational flexibility, complex reasoning |

**Key architectural distinction:** Hyro calls itself an "Adaptive Communications" platform. Rather than training ML models per client or building decision trees per use case, it:

1. **Scrapes** data from multiple organizational sources (websites, EMRs, databases, APIs, PDFs)
2. **Maps** scraped data into a **knowledge graph** with entities, relationships, and dynamic ontologies
3. **Layers** NLU on top using **computational linguistics** (not ML-based intent classification)
4. **Queries** the knowledge graph to construct responses and determine next actions

This means the system can deploy in as few as **3 days** without training data, and the knowledge graph auto-updates as source data changes -- no manual retraining required.

**The NLU engine is described as "intent-less":** Rather than matching user utterances to pre-defined intents (the approach used by Dialogflow, Lex, LUIS, etc.), it parses sentences to understand context and entities, then queries the knowledge graph dynamically. Hyro claims this covers **10x more topics** than intent-based systems.

**SLM (Small Language Model) layer:** Hyro has built healthcare-specific Small Language Models that sit alongside the larger LLMs. These SLMs are purpose-built for healthcare vocabulary, workflows, and compliance requirements. The exact SLM architecture is not publicly documented.

### 1.2 Responsible AI Framework: "Triple C Standard"

Hyro's governance framework is called the **Triple C Standard**:

| Principle | Mechanism | What It Claims to Do |
|-----------|-----------|---------------------|
| **Clarity** | Explainability engine | Complete explainability into logic pathways and information sources used to formulate each response. Every response is auditable and traceable to verifiable data sources. |
| **Control** | Data source restriction | Eliminates hallucinations by clearly defining and restricting the data sources used to generate outputs. Only trusted, approved internal data sources feed the conversational interface. |
| **Compliance** | Regulatory alignment | Adapts to CMS, Joint Commission, HIPAA, FDA regulations. Meets security standards to prevent data breaches. |

**Additional claimed safety features:**
- Red-flag detection for clinical risk
- Fallback logic when confidence is low
- Override protocols for clinical review
- PII/PHI redaction capabilities
- Built-in safeguards to prevent hallucinations

**Governance gap (TELOS analysis):** The Triple C framework is a **design philosophy**, not a **runtime governance system**. There is no published evidence of:
- Quantified confidence thresholds that trigger escalation
- Real-time drift detection during conversations
- Statistical process control on agent behavior over time
- Independent third-party validation of hallucination prevention claims
- Continuous monitoring of decision accuracy post-deployment

### 1.3 NLU/NLP Models

| Component | Technology |
|-----------|-----------|
| **NLU Engine** | Proprietary "intent-less" computational linguistics engine |
| **Knowledge Representation** | Proprietary knowledge graph with dynamic ontologies |
| **LLM Integration** | Uses LLMs (including ChatGPT/OpenAI) for generalization |
| **SLMs** | Healthcare-specific Small Language Models (proprietary) |
| **ASR/STT** | Automatic Speech Recognition for voice channel (specific engine not disclosed -- likely third-party via telephony integration) |
| **Accuracy Claim** | 95% NLU accuracy rate; 98% of patient questions answered correctly over 90 days |

### 1.4 Multi-Turn Conversation Handling

The knowledge graph architecture enables multi-turn conversations by:

1. **Entity tracking:** The knowledge graph maintains entity state across turns (patient identity, appointment details, medication names, etc.)
2. **Context accumulation:** Each turn adds context that narrows the knowledge graph query space
3. **Dynamic next-question generation:** The system determines what questions to ask next based on the data it has and what it still needs, rather than following a fixed dialogue tree
4. **Cross-channel continuity:** Can transition from call to text mid-conversation while maintaining context

### 1.5 Handoff Protocol: Contextual Transfer to Human Agents

When the Hyro agent cannot resolve a request:

1. **Intent mapping:** The NLU engine identifies the caller's intent and determines it cannot be resolved autonomously
2. **Context packaging:** All conversation context (identified patient, stated intent, collected information, conversation history) is packaged
3. **Smart routing:** NLU-based routing determines the correct department/agent based on the nature of the request
4. **Contextual transfer:** The packaged context is transferred to the human agent's desktop (via Cisco Finesse, Salesforce, NICE, Five9, Genesys, etc.)
5. **Agent receives:** Full conversation history, patient identification status, intent classification, and any partial task completion

**Claimed results:** 85% drop in abandonment rates and 79% improvement in speed-to-answer with smart routing.

**Governance gap:** The decision boundary between "resolve autonomously" and "transfer to human" is not publicly documented. There are no published confidence thresholds, no published error rates for incorrect autonomous resolution, and no published rates of inappropriate escalation (false negatives/positives in the handoff decision).

---

## 2. Tool Calls / Actions the Agent Takes Autonomously

### 2.1 Action Map: Every Known Autonomous Action at BSMH

| Action Category | Specific Actions | System Written To | Autonomy Level |
|----------------|-----------------|-------------------|----------------|
| **Scheduling** | Find doctors (by specialty, location, gender, language); browse available time slots; book appointments; confirm appointments; cancel appointments; reschedule appointments | Epic (via API -- likely Cadence scheduling module) | **Fully autonomous** -- no human in the loop |
| **Patient Identification** | Verify patient identity using PII; authenticate callers; look up patient records | Epic EMR (MyChart patient records) | **Fully autonomous** -- uses PII verification against EHR |
| **Registration** | Collect patient data; fill registration forms during chat; update patient records | Epic EMR | **Partially autonomous** -- data collection automated, some workflows may require human completion |
| **Prescription Management** | Verify patient via PII + prescription number; confirm refill eligibility; check remaining refills; process refill requests; send SMS with delivery status/pick-up address | Epic **Willow Ambulatory** module | **Fully autonomous** -- end-to-end refill processing |
| **Routing** | Classify caller intent; determine target department; transfer call with context | Telephony system (Cisco/NICE/etc.) + CRM | **Fully autonomous** -- NLU-based smart routing |
| **Billing** | Generate invoices; provide information on insurance claims; calculate estimates; process payments | Epic revenue cycle modules (likely Resolute/Prelude) | **Partially autonomous** -- information retrieval automated, payment processing extent unclear |
| **Insurance Verification** | Verify insurance coverage; calculate cost estimates | Epic eligibility modules | **Partially autonomous** -- verification automated, complex cases likely escalated |
| **MyChart Support** | Password resets; portal navigation assistance; appointment verification via portal | Epic MyChart | **Fully autonomous** for common support tasks |
| **Provider Search** | Search physician directory by multiple attributes; return matching providers with availability | Epic provider directory / scheduling | **Fully autonomous** |
| **Lab/Blood Test Scheduling** | Check patient EMR for correct test; offer available times and locations; provide pre-test instructions (fasting, etc.) | Epic (lab scheduling modules) | **Fully autonomous** |
| **SMS/Text Messaging** | Send appointment confirmations; send prescription status updates; send appointment details via text | SMS gateway integration | **Fully autonomous** -- triggered by completed actions |
| **Outbound Proactive Communication** | Appointment reminders; referral reminders; billing follow-ups; coverage re-enrollment (ARMR) | Epic + SMS/voice outbound | **Fully autonomous** (Proactive Px platform) |

### 2.2 Integration Partners for Tool Execution

| Integration Category | Specific Systems |
|---------------------|-----------------|
| **EHR/EMR** | Epic (certified app on Epic Showroom), Cerner, athenahealth, eClinicalWorks, 80+ EHR/PM systems |
| **CRM** | Salesforce (Agentforce Health integration), ServiceNow |
| **Telephony/CCaaS** | Cisco (Webex Contact Center, Finesse), NICE inContact, Genesys, Five9, Twilio Flex |
| **Data Exchange** | FHIR APIs, SFTP, modern REST APIs |
| **Messaging** | SMS gateways (for appointment confirmations, prescription status) |
| **Patient Portal** | Epic MyChart |

---

## 3. The Action Chain

### 3.1 Inbound Patient Call -- Full Action Chain

```
STEP 1: CALL INTAKE / VOICE PROCESSING
   Trigger:     Patient calls BSMH call center number
   System:      Telephony (Cisco) routes to Hyro via SIP/DID integration
   Action:      ASR converts speech to text
   Agent:       Hyro voice agent ("Aimee" at TGH; agent name at BSMH unknown) greets caller
   Tool call:   [telephony_answer] -> SIP session established
   Governance:  None documented -- no published latency or ASR accuracy monitoring

STEP 2: INTENT RECOGNITION
   Trigger:     Patient states their need in natural language
   System:      Intent-less NLU engine + knowledge graph
   Action:      Computational linguistics parses utterance; entities extracted;
                knowledge graph queried to determine intent category
   Categories:  Scheduling | Rx Refill | Billing | Insurance | MyChart |
                Provider Search | Lab Scheduling | General Inquiry | Other
   Tool call:   [nlu_classify_intent] -> intent_category + confidence_score
   Decision:    IF resolvable autonomously -> proceed to Step 3
                IF not resolvable -> proceed to Step 6 (contextual transfer)
   Governance:  Confidence threshold not publicly documented

STEP 3: PATIENT IDENTIFICATION / VERIFICATION
   Trigger:     Intent requires patient-specific data
   System:      Epic EMR via API
   Action:      Agent requests identifying information (name, DOB, etc.)
                Verifies against Epic patient records
   Tool call:   [epic_patient_lookup] -> patient_id + verification_status
   For Rx:      Also requests prescription number
                [epic_rx_verify] -> prescription_id + eligibility_status
   PHI access:  Patient demographics, appointment history, prescription records
   Governance:  HIPAA-compliant PII handling claimed; PII/PHI redaction in logs claimed;
                specific verification protocol not publicly documented

STEP 4: INFORMATION RETRIEVAL FROM EHR
   Trigger:     Patient verified; specific data needed
   System:      Epic via plug-and-play API (certified Epic app)
   Action:      Retrieves relevant data based on intent:
                - Scheduling: provider availability, open slots, locations
                - Rx: refill count, delivery status, pharmacy info
                - Billing: claim status, balance, estimate calculations
                - Labs: required tests, available times, prep instructions
   Tool calls:  [epic_get_availability] | [epic_get_rx_status] |
                [epic_get_billing_info] | [epic_get_lab_orders]
   Read access: Provider schedules, patient appointments, prescription records,
                billing records, lab orders, MyChart data
   Governance:  Read access scope not publicly documented per module

STEP 5: AUTONOMOUS ACTION (WRITE-BACK TO EHR)
   Trigger:     All required information collected; action confirmed by patient
   System:      Epic via API write-back
   Actions:
     - SCHEDULING: Book appointment -> [epic_schedule_appointment]
     - SCHEDULING: Cancel/reschedule -> [epic_modify_appointment]
     - RX REFILL: Process refill request -> [epic_rx_refill_request]
                  (Willow Ambulatory module)
     - BILLING: Process payment -> [epic_process_payment] (extent unclear)
   Write access: Appointment creation/modification, prescription refill requests,
                 potentially payment processing
   Governance:  NO PUBLISHED CONFIRMATION LOOP before EHR write
                NO PUBLISHED UNDO/ROLLBACK mechanism
                NO PUBLISHED AUDIT TRAIL for autonomous write actions

STEP 6: CONFIRMATION TO PATIENT
   Trigger:     Action completed (or transfer initiated)
   System:      Voice (ASR/TTS) + SMS gateway
   Action:      Verbal confirmation to patient on call
                SMS sent with appointment details, prescription status,
                or other relevant confirmation
   Tool call:   [sms_send_confirmation] -> delivery_status
   Governance:  Confirmation content not independently validated

STEP 7: EHR UPDATE / LOGGING
   Trigger:     Interaction complete
   System:      Epic EMR + Hyro Patient Intelligence Dashboard
   Actions:
     - All changes persisted in Epic (appointment, Rx, etc.)
     - Interaction logged in Hyro analytics dashboard
     - Conversational metrics captured: appointments scheduled,
       speed to answer, agent hours saved, conversion rates,
       satisfaction scores, NLU understanding rate, error rate
   Tool call:   [hyro_log_interaction] -> dashboard_updated
   Governance:  Hyro dashboard provides operational metrics but NO published
                governance-grade audit trail (no immutable log, no
                cryptographic verification, no per-action tracing)
```

### 3.2 Outbound Proactive Call -- Action Chain (Proactive Px)

```
STEP 1: TRIGGER IDENTIFICATION
   System:      Epic EHR + Proactive Px engine
   Action:      Identifies patients requiring outreach:
                - Upcoming appointment reminders
                - Referral follow-ups
                - Billing reminders
                - Coverage re-enrollment (ARMR)
   Tool call:   [epic_query_patient_cohort] -> patient_list

STEP 2: OUTBOUND CONTACT
   System:      Voice or SMS channel
   Action:      AI agent initiates contact with personalized message
                based on EHR data
   Tool call:   [outbound_call_initiate] | [sms_send_outbound]

STEP 3: SELF-SERVICE FLOW
   Action:      Patient can confirm, cancel, or reschedule appointments;
                update insurance information; complete billing actions
                -- all within the automated outbound interaction
   Tool calls:  Same as inbound Steps 3-5

STEP 4: ESCALATION (if needed)
   Action:      Complex cases escalated to live agent with full context
   Tool call:   [transfer_to_agent] with context_payload
```

---

## 4. EHR Integration

### 4.1 How Hyro Integrates with Epic

| Integration Aspect | Detail |
|-------------------|--------|
| **Integration method** | Plug-and-play non-intrusive API (REST). NOT RPA. NOT screen scraping. |
| **Certification** | Fully certified Epic app; listed on Epic Showroom (formerly AppOrchard) |
| **Data flow** | Bidirectional: reads patient data FROM Epic, writes actions (appointments, Rx requests) BACK to Epic |
| **Authentication** | Meets Epic's stringent security and quality requirements (specific auth protocol not disclosed -- likely OAuth 2.0 via Epic on FHIR) |
| **Deployment model** | Layers on top of existing Epic infrastructure; no system overhaul required |
| **Data ingestion** | Automatically scrapes and ingests physician profiles and patient EMRs stored within Epic infrastructure |
| **FHIR** | Supports FHIR/APIs and SFTP. Modern APIs and FHIR ensure consistent AI behavior across multi-site and multi-specialty environments |
| **Update mechanism** | Knowledge graph auto-updates as Epic data changes |

### 4.2 Epic Modules Accessed

| Epic Module | Access Type | What Hyro Does With It |
|-------------|-------------|----------------------|
| **Cadence** (Scheduling) | Read + Write | Reads provider availability, slot times, locations. Writes new appointments, modifications, cancellations. |
| **MyChart** (Patient Portal) | Read + limited Write | Patient identification, portal support (password resets), appointment verification. "Critical MyChart skills for managing patient data." |
| **Willow Ambulatory** (Pharmacy) | Read + Write | Reads prescription records, refill eligibility, remaining refills. Writes refill requests. "Seamlessly integrates Rx Management with Epic Willow Ambulatory module." |
| **Prelude** (Registration) | Read + Write (likely) | Patient demographics, registration data collection |
| **Provider Directory** | Read | Physician profiles by specialty, location, gender, language, availability |
| **Resolute** (Billing) | Read (confirmed) + Write (unclear) | Insurance claims information, billing estimates, invoice generation |

### 4.3 Read/Write Access Summary

| Access Level | Confirmed Capabilities |
|-------------|----------------------|
| **READ** | Patient demographics, appointment history, provider schedules, prescription records (refill count, delivery status), billing/claims data, lab orders, MyChart account status, provider directory |
| **WRITE** | Appointment creation/modification/cancellation, prescription refill requests, patient record updates during registration |
| **UNCLEAR** | Payment processing (claimed but extent unknown), insurance verification write-back, lab order creation |

### 4.4 FHIR Layer

Hyro supports FHIR APIs as part of its interoperability stack. Epic's FHIR endpoints (via Epic on FHIR / open.epic.com) provide standardized access to:
- Patient resources (demographics, conditions, medications)
- Scheduling resources (slots, appointments)
- Clinical resources (observations, diagnostic reports)

However, it is unclear whether Hyro uses **exclusively** FHIR or also uses Epic's proprietary APIs (Interconnect, web services) for deeper integration.

---

## 5. Governance & Safety

### 5.1 What Happens When the Agent Is Uncertain

| Mechanism | What We Know | What We Don't Know |
|-----------|-------------|-------------------|
| **Fallback logic** | Claimed "fallback logic when confidence is low" | Specific confidence thresholds are NOT published |
| **Smart routing** | Low-confidence or unresolvable intents are routed to live agents | The decision boundary between autonomous resolution and human escalation is not documented |
| **Red-flag detection** | Claimed "red-flag detection for clinical risk" | What constitutes a red flag, how it's detected, and what action follows are not documented |
| **Override protocols** | Claimed "override protocols for clinical review" | Who can override, under what conditions, and how overrides are logged are not documented |

**Industry context:** Enterprise AI systems typically implement dynamic confidence thresholds (e.g., 95% for medical coding, 85% for scheduling). Hyro does not publish its threshold values, making independent governance assessment impossible.

### 5.2 Audit Trail

| Feature | Status |
|---------|--------|
| **Patient Intelligence Dashboard** | Real-time analytics on conversational metrics, automation success, agent workload reduction, cost per interaction |
| **NLU metrics** | Understanding rate, error rate, with actionable analytics |
| **Engagement analytics** | Conversion rates, CTA breakdowns, satisfaction scores, drop-off analytics |
| **Per-conversation explainability** | Claimed ability to "unpack each conversation, identify knowledge sources utilized, understand root cause of wrong outputs" |
| **Immutable audit log** | NOT published |
| **Cryptographic verification of actions** | NOT published |
| **Per-action governance trace** | NOT published |
| **Independent audit capability** | NOT published |

**Governance gap:** The Patient Intelligence Dashboard is an **operational analytics** tool, not a **governance audit trail**. It tracks business metrics (calls handled, appointments booked, hours saved) but there is no published evidence of:
- Tamper-evident logging of every autonomous action
- Per-action decision provenance (why did the agent choose action X over action Y?)
- Statistical drift detection over time
- Independent third-party audit access

### 5.3 HIPAA Compliance

| Compliance Area | Status |
|----------------|--------|
| **HIPAA Compliant** | Claimed; specific safeguards not fully disclosed |
| **SOC 2** | Reported SOC 2 posture (specific type -- Type I or Type II -- not publicly confirmed) |
| **BAA** | Enterprise BAA support offered |
| **PHI handling** | PII/PHI redaction capabilities claimed |
| **Encryption** | Not publicly specified (industry standard would be AES-256 at rest, TLS 1.2+ in transit) |
| **Data residency** | Not publicly specified |
| **PHI retention** | Not publicly specified |
| **Voice recording** | Voice data containing health information is PHI under HIPAA; handling protocol not disclosed |

### 5.4 PHI Accessed by the Agent

Based on documented capabilities, the Hyro agent accesses:

| PHI Category | Access Context |
|-------------|---------------|
| Patient name | Identification/verification |
| Date of birth | Identification/verification |
| Phone number | Caller ID, SMS messaging |
| Address | Registration, appointment logistics |
| Medical record number | EHR lookup |
| Appointment history | Scheduling management |
| Prescription records | Refill management (medication names, refill counts, pharmacy) |
| Insurance information | Verification, billing estimates |
| Lab orders | Lab scheduling, pre-test instructions |
| Billing/claims data | Billing inquiries, payment processing |
| Provider relationships | Scheduling, referrals |

**This is an extensive PHI footprint for an autonomous agent with no published independent governance layer.**

### 5.5 Published Error Rates and Safety Incidents

| Metric | Published Value | Source |
|--------|----------------|--------|
| **NLU accuracy** | 95% | Hyro marketing materials |
| **Response accuracy** | 98% of questions answered correctly (90-day window) | Hyro marketing materials |
| **Call containment** | 60%+ | Hyro marketing materials |
| **Autonomous resolution** | Up to 85% of routine interactions | Hyro marketing materials |
| **Safety incidents** | **None published** | No public incident reports, no FDA MAUDE entries, no published adverse event data |
| **Hallucination rate** | **Not published** (claims elimination via data source restriction) | No independent validation |
| **Misrouting rate** | **Not published** | |
| **Incorrect scheduling rate** | **Not published** | |
| **Incorrect Rx refill rate** | **Not published** | |

**Critical governance gap:** All published metrics are self-reported by Hyro or co-reported with customer health systems. No independent third-party validation of accuracy, error rates, or safety has been published.

---

## 6. Multi-Channel Architecture

### 6.1 Channel Capabilities

| Channel | Capabilities | Technical Integration |
|---------|-------------|----------------------|
| **Phone (Voice)** | Full NLU, scheduling, Rx refills, billing, routing, identity verification, contextual transfer to live agents | SIP/DID integration with Cisco, NICE, Genesys, Five9; ASR/TTS for voice processing |
| **Web Chat** | Full NLU, scheduling, Rx refills, provider search, registration, FAQ, MyChart support | Embedded widget on health system websites; knowledge graph-powered |
| **SMS/Text** | Appointment confirmations, Rx status updates, self-service deflection from voice, outbound reminders | SMS gateway integration; can receive and process inbound SMS requests |
| **Mobile App** | Similar to web chat capabilities within health system mobile applications | SDK or embedded integration |
| **Social Messaging** | Business messaging app integration | Platform-specific APIs |
| **Email** | Outbound communications | Email gateway integration |

### 6.2 Omnichannel Architecture

The same underlying knowledge graph, NLU engine, and API integrations power all channels. The key architectural features are:

1. **Single knowledge graph:** All channels query the same knowledge graph, ensuring consistent answers
2. **Unified analytics:** The Patient Intelligence Dashboard aggregates metrics across all channels
3. **Cross-channel transitions:** A conversation can transition from phone to SMS mid-interaction with context preserved
4. **Channel-appropriate formatting:** Same data, different presentation (voice prompts vs. text messages vs. chat widgets)

### 6.3 Channel-Specific Differences

| Capability | Voice | Chat | SMS |
|-----------|-------|------|-----|
| Scheduling (end-to-end) | Yes | Yes | Yes (simplified) |
| Patient identification | Voice-based PII | Form-based PII | Limited |
| Rx refills | Yes | Yes | Yes |
| Billing inquiries | Yes | Yes | Limited |
| Smart routing to live agent | Yes (warm transfer) | Yes (chat handoff) | Yes (callback) |
| Outbound proactive | Yes (Proactive Px) | No | Yes (Proactive Px) |

---

## 7. Comparable Healthcare Agentic Systems

### 7.1 Commure Agents (130+ Health Systems)

**Overview:** Commure delivers AI infrastructure for enterprise health systems, integrating ambient intelligence, agentic AI, and revenue cycle automation on a single platform. It integrates with 60+ EHRs and powers millions of encounters and tens of billions in annual claims.

**What the agents actually do:**

| Agent | Function | Actions |
|-------|----------|---------|
| **Call Center Agent** | Handles inbound/outbound patient calls | Schedules, confirms, reschedules, cancels appointments; handles billing inquiries; processes pre-authorizations autonomously |
| **Scheduling Agent (Sherpa)** | Manages provider calendars | Resolves scheduling conflicts; optimizes calendars; supports inbound and outbound scheduling workflows |
| **Prior Authorization Agent** | Automates insurance approvals | Collects medication details from patients; initiates authorization requests via structured forms or EHR-integrated prompts; tracks approvals in real time |
| **Referral Management Agent** | Automates specialist referrals | Tracks and processes referrals; automates follow-ups with external providers |
| **Ambient Documentation** | Real-time clinical note generation | Converts clinician-patient conversations to clinical notes using speech recognition and LLMs; autonomous coding; clinical guidance |
| **Medical Coding Agent** | Automated coding | AI-assisted coding integrated with EHR; shared architecture with ambient AI for complete clinical picture |
| **Revenue Cycle Management** | End-to-end billing automation | Eligibility, intake, documentation, AR, collections, denials, payment posting |

**Technical architecture:**
- Shared architecture between ambient AI and coding models -- same patient conversation data flows through both
- Full EHR integration (60+ EHRs including Epic)
- Available in Epic Toolbox for ambient voice recognition
- Embedded in MEDITECH Expanse
- Forward Deployed Engineering teams work directly with clinicians
- $200M raised (General Catalyst)

**Governance gap (from TELOS perspective):** Commure agents are deeply embedded in clinical workflows (coding, documentation, prior authorization) with autonomous decision-making capability. No published independent governance framework, no published confidence thresholds, no published error rates for autonomous coding or authorization decisions.

### 7.2 Qventus AI (150+ Hospital Facilities)

**Overview:** Qventus uses AI to automate hospital operations, leveraging Generative AI, machine learning, and behavioral science. Deeply integrated with EHRs.

**How the OR scheduling agent works:**

| Component | Function | Technical Detail |
|-----------|----------|-----------------|
| **Block Prediction** | Predicts unused OR blocks | ML models predict with high confidence which blocks won't be used, up to a month in advance |
| **TimeFinder** | Real-time OR time reservation | Intuitive interface with ML algorithms that filter and prioritize slots based on surgeon's past case time performance |
| **Nudging System** | Encourages block release | Uses behavioral science to nudge block owners to release unused time |
| **Marketing Engine** | Fills released time | Automatically markets available time to best-fit surgeons based on strategic goals |
| **Fax Digitizer** | Processes surgical faxes | AI teammate digitizes, categorizes, and summarizes thousands of pages of faxes; routes to correct teams |
| **Robotics Assistant** | Optimizes robotic surgery | Drives >10% incremental robotic volume; reduces non-robotic cases in robotic ORs by 11% |
| **Risk Stratification** | Patient safety | ML continuously stratifies patients by risk throughout perioperative journey; flags patient-specific risks from EHR data, chart notes, and patient conversations |

**Technical architecture:**
- Integrated with EHRs with write-back functionality
- ML models for predictive scheduling and risk stratification
- Behavioral science layer for human-facing nudges
- EHR write-back eliminates double data entry
- Saves estimated 100 hours per OR scheduler per month

**Key governance feature:** Qventus operates primarily through **recommendations and nudges** rather than fully autonomous execution. Surgeons and schedulers retain decision authority. This is a more conservative autonomy model than Hyro's fully autonomous scheduling.

### 7.3 VoiceCare AI at Mayo Clinic

**Overview:** VoiceCare AI is an agentic AI startup automating back-office conversations between providers and payers. Its voice agent "Joy" handles administrative tasks.

**Architecture:**

| Component | Detail |
|-----------|--------|
| **Agent name** | "Joy" |
| **Architecture type** | Multi-modal agentic architecture with RLHF |
| **Training data** | Proprietary healthcare conversational data |
| **Safety claim** | "Zero-skip, hallucination-free" architecture -- does not bypass critical questions; delivers context-bound, verifiable responses |
| **Call handling** | Supports conversations from a few minutes to 3 hours; can hold for up to 2.5 hours |
| **Accuracy claim** | 100% call completion accuracy rate; 87-90% of calls completed autonomously |
| **Compliance** | HIPAA compliant; SOC 2 Type II attested |

**Mayo Clinic pilot scope:**
- Patient pre-authorization and benefit confirmations
- Department of Neurology
- Department of Pediatrics
- Medical and Administrative Support Operations

**What Joy actually does:**
1. Verifies benefits
2. Obtains prior authorizations
3. Follows up on claims
4. Generates comprehensive call summaries and documentation
5. Integrates with existing systems and processes

**Key differentiator from Hyro:** VoiceCare AI focuses on **provider-to-payer** conversations (calling insurance companies on behalf of providers), not **patient-to-provider** conversations. Joy navigates insurance company phone trees, holds for extended periods, and completes administrative tasks that would otherwise consume staff time. This is a fundamentally different use case than Hyro's patient-facing call center automation.

---

## 8. Governance Gap Analysis (TELOS Perspective)

### 8.1 Critical Governance Gaps in Hyro at BSMH

| Gap | Severity | Detail |
|-----|----------|--------|
| **No published confidence thresholds** | CRITICAL | The system makes autonomous decisions to schedule appointments, process Rx refills, and modify patient records with no published threshold for when it should escalate vs. act autonomously |
| **No published audit trail architecture** | CRITICAL | The Patient Intelligence Dashboard tracks operational metrics, not governance-grade per-action provenance with tamper-evident logging |
| **No published error rates** | CRITICAL | Self-reported 98% accuracy with no independent validation. No published rates for misscheduling, incorrect Rx refills, or misrouting |
| **No published drift detection** | HIGH | No mechanism to detect if agent behavior changes over time as knowledge graph updates or LLM weights change |
| **No published PHI access scope documentation** | HIGH | The agent accesses extensive PHI (demographics, Rx records, billing, appointments) but the exact scope of read/write access per Epic module is not publicly documented |
| **No published rollback mechanism** | HIGH | When the agent books a wrong appointment or processes an incorrect Rx refill, there is no published undo/correction protocol |
| **No published voice data handling** | HIGH | Voice recordings containing health information are PHI; retention, encryption, and deletion policies are not publicly documented |
| **Self-reported metrics only** | MEDIUM | All published performance data (95% NLU accuracy, 98% response accuracy, 85% resolution rate) comes from Hyro or its customers, with no independent third-party validation |
| **No published bias audit** | MEDIUM | No published analysis of whether the system performs differently across patient demographics (age, language, accent, health literacy) |
| **Hallucination prevention is design-level, not runtime** | MEDIUM | The "Control" pillar (restricting data sources) is a design decision, not a runtime monitoring capability. There is no published mechanism to detect if a hallucination occurs despite the design restriction |

### 8.2 What TELOS Would Govern

If TELOS were deployed alongside Hyro at BSMH, it would provide:

1. **Runtime fidelity measurement** on every agent decision (is this scheduling action aligned with the declared purpose?)
2. **Statistical drift detection** over time (is the agent's behavior changing as the knowledge graph updates?)
3. **Per-action governance trace** with cryptographic verification (immutable audit trail for every EHR write)
4. **Confidence threshold validation** (is the agent escalating appropriately when uncertain?)
5. **Cross-system governance** (governing Hyro + Viz.ai + Ambient AI + Catherine + Lirio under a unified framework)
6. **Demographic bias detection** (is the agent performing differently across patient populations?)
7. **Independent third-party measurement** (not self-reported metrics)

---

## Sources

### Hyro Architecture & Deployment
- [Hyro: Conversational AI Solution for Enterprises](https://www.hyro.ai/)
- [Hyro Responsible AI Framework](https://www.hyro.ai/responsible-ai/)
- [Hyro Raises $45M Strategic Growth Round](https://www.prnewswire.com/news-releases/hyro-raises-45m-strategic-growth-round-to-accelerate-ai-agent-adoption-in-healthcare-302589268.html)
- [Autonomous AI Agents in Healthcare: Moving Beyond Chatbots](https://www.hyro.ai/blog/autonomous-ai-agents-in-healthcare/)
- [Hyro Conversational AI Platform](https://www.hyro.ai/platform/)
- [Omnichannel AI Assistant](https://www.hyro.ai/ai-assistant/)
- [Hyro AI Call Center Automation](https://www.hyro.ai/call-center-automation/)
- [What is Adaptive Communications](https://www.hyro.ai/glossary/what-is-adaptive-communications/)
- [Intent-less NLU with Hyro CEO (VUX World)](https://vux.world/intentness-nlu-and-knowledge-management-with-hyro-ceo-israel-krush/)
- [What We're Building at Hyro](https://www.hyro.ai/blog/what-were-building-at-hyro/)

### Epic Integration
- [Hyro Announces New Integration With Epic Systems](https://www.hyro.ai/blog/hyro-announces-new-integration-with-epic/)
- [Integrate Epic EMR with Hyro's Conversational AI](https://www.hyro.ai/integration/epic/)
- [Hyro Healthcare Scheduling Management](https://www.hyro.ai/healthcare/scheduling-management/)
- [Hyro Rx Management Automation](https://www.hyro.ai/healthcare/rx-management/)
- [Hyro Enterprise Integrations](https://www.hyro.ai/integrations/)

### BSMH & Customer Deployments
- [Tampa General Hospital Implements Hyro's Voice AI Agents](https://www.hyro.ai/blog/tgh-call-center-implements-hyros-voice-ai-agents/)
- [How Tampa General and Hyro Rolled Out AI Voice Agents in 3 Months (Fierce Healthcare)](https://www.fiercehealthcare.com/ai-and-machine-learning/how-tampa-general-hospital-and-startup-hyro-rolled-out-ai-voice-agents-3)
- [Tampa General Cuts Call Wait Times 58% (Becker's)](https://www.beckershospitalreview.com/healthcare-information-technology/ai/tampa-general-cuts-call-wait-times-58-with-ai-rollout/)
- [In Tampa General's Call Center, AI Drives ROI (Healthcare IT News)](https://www.healthcareitnews.com/news/tampa-generals-call-center-ai-drives-roi)
- [Tampa General's Arnold on Voice Agent (Health System CIO)](https://healthsystemcio.com/2026/01/12/tampa-generals-arnold-says-voice-agent-hitting-right-notes-for-overwhelmed-call-center/)
- [Sutter Health Teams Up with Hyro](https://vitals.sutterhealth.org/sutter-health-teams-up-with-hyro-to-revolutionize-patient-communications-through-ai/)
- [How BSMH Became a Leader in Digital Investing (Becker's)](https://www.beckershospitalreview.com/healthcare-information-technology/digital-health/how-bon-secours-mercy-health-became-a-leader-in-digital-investing/)

### Proactive Px & Salesforce
- [Hyro Launches Proactive Px with ARMR Agent](https://www.prnewswire.com/news-releases/hyro-launches-proactive-px-with-ai-agent-to-help-healthcare-organizations-handle-coverage-disruption-caused-by-obbba-302544700.html)
- [Proactive Patient Engagement at Scale](https://www.hyro.ai/blog/proactive-patient-engagement-at-scale/)
- [Hyro on Salesforce AppExchange](https://appexchange.salesforce.com/appxListingDetail?listingId=1d79ef2e-5c95-40ff-99df-dde980f04bb2)
- [Hyro on Salesforce Agentforce](https://appexchange.salesforce.com/appxListingDetail?listingId=6e08b254-32c8-4230-b6fa-8d5c6f0e476d)
- [Hyro + Cisco Webex Contact Center](https://www.prnewswire.com/news-releases/hyro-transforms-patient-care-with-ai-powered-assistants-and-ciscos-webex-contact-center-302285253.html)

### Compliance & Security
- [5 Voice AI Platforms Compliant With Healthcare Regulations](https://www.getprosper.ai/blog/5-voice-ai-platforms-compliant-with-healthcare-regulations)
- [7 Best HIPAA Compliant AI Tools (2026)](https://aisera.com/blog/hipaa-compliance-ai-tools/)
- [Hyro on Webex App Hub](https://apphub.webex.com/applications/hyro-responsible-ai-powered-communications)

### Analytics
- [Hyro Patient Journey Analytics](https://www.hyro.ai/healthcare/conversational-analytics/)
- [Hyro Conversation Intelligence & Analytics](https://www.hyro.ai/conversational-intelligence-analytics/)

### Comparable Systems
- [Commure Agents](https://www.commure.com/agents)
- [How AI Agents Are Transforming the Healthcare Call Center (Commure)](https://www.commure.com/blog/how-ai-agents-are-transforming-the-healthcare-call-center)
- [Commure Launches Agents (Press Release)](https://www.commure.com/press-releases/commure-launches-commure-agents---ai-assistants-that-fully-automate-physician-workflows)
- [Commure Secures $200M](https://www.commure.com/blog/commure-secures-200m-to-accelerate-ai-powered-healthcare-transformation)
- [A Foundational Architecture for AI Agents in Healthcare (Cell Reports Medicine)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12629813/)
- [Qventus AI Platform](https://www.qventus.com/)
- [Qventus Surgical Growth Solution](https://www.qventus.com/solutions/operating-room-utilization/)
- [Qventus AI Teammates Perioperative Journey](https://www.qventus.com/resources/blog/how-qventus-ai-teammates-drive-strategic-surgical-growth-and-transform-the-perioperative-care-journey/)
- [VoiceCare AI Launches (PRNewswire)](https://www.prnewswire.com/news-releases/agentic-ai-startup-voicecare-ai-launches-to-automate-healthcare-back-office-and-super-staff-workforce-302376944.html)
- [VoiceCare AI Pilot with Mayo Clinic (Fierce Healthcare)](https://www.fiercehealthcare.com/ai-and-machine-learning/voicecare-ai-new-agentic-ai-startup-kicks-pilot-mayo-clinic-automate-back)

### Market & Regulatory Context
- [Top 9 Companies Building AI Agents in Healthcare 2026](https://lightit.io/blog/top-9-companies-building-ai-agents-in-healthcare-2026/)
- [Agentic AI Is Reshaping Healthcare in 2026 (Hyro)](https://www.hyro.ai/blog/is-your-organization-agentic-ai-ready-for-2026/)
- [Hyro Responsible AI Platform Wins 2025 Fierce Healthcare Innovation Award](https://third-news.com/article/3f88a3ba-c57c-11f0-8ab8-9ca3ba0a67df)

---

*This analysis was produced as part of TELOS research into the governance gaps of production agentic AI systems in healthcare. All findings are based on publicly available information as of February 2026. Specific technical details that are marked as "not publicly documented" may exist in Hyro's proprietary documentation but are not accessible for independent review.*
