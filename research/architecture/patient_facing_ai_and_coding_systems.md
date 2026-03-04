# Patient-Facing Conversational AI & AI-Assisted Medical Coding/Billing Systems
# Technical Workflows, Regulatory Implications, and Governance Gaps

**Prepared by:** Schaake (Regulatory Analyst), TELOS Research Team
**Date:** 2026-02-16
**Purpose:** Operational-level reference documenting how patient-facing AI and AI-assisted coding/billing systems work in production -- technical pipelines, tool calls, regulatory exposure points, and enforcement landscape.

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Table of Contents

1. [Patient-Facing Conversational AI](#1-patient-facing-conversational-ai)
   - 1.1 Catherine (Brado) at BSMH
   - 1.2 Epic MyChart In-Basket ART
   - 1.3 General Patient-Facing AI Chatbot Taxonomy
   - 1.4 Known Failure Modes (ECRI 2026 Hazard List)
   - 1.5 Regulatory Frameworks for Patient-Facing AI
   - 1.6 HIPAA Exposure When Patients Share PHI with Chatbots
2. [AI-Assisted Medical Coding and Billing](#2-ai-assisted-medical-coding-and-billing)
   - 2.1 Technical Pipeline: Clinical Note to Billing Code
   - 2.2 Major AI Coding Systems in Production
   - 2.3 The "Coding Arms Race" Problem
   - 2.4 Stanson Health HCC Alerts at BSMH
   - 2.5 CMS WISeR (AI Prior Authorization) System
3. [Action Chain: Patient-Facing AI](#3-action-chain-patient-facing-ai)
4. [Action Chain: AI-Assisted Coding](#4-action-chain-ai-assisted-coding)
5. [Regulatory Exposure Matrix](#5-regulatory-exposure-matrix)
6. [Behavioral AI / Precision Nudging (Lirio at BSMH)](#6-behavioral-ai--precision-nudging-lirio-at-bsmh)
7. [Sources](#7-sources)

---

## 1. Patient-Facing Conversational AI

### 1.1 Catherine (Brado) at BSMH

**What it is:** Catherine is a Conversational AI Engagement Platform developed by Brado and deployed at Bon Secours Mercy Health (BSMH). Named after Mercy Health founder Sr. Catherine McAuley. Launched October 2023 for dementia caregiver support; orthopedic service line launched Fall 2024 in Cincinnati. Expansion to Greenville and Richmond markets planned for August 2025, with cardiac health, primary care, and oncology under research.

**How it was built:**
- Brado's proprietary customer journey mapping, informed by "hundreds of hours of 1-on-1 patient journey interviews" with patients and caregivers navigating different types of chronic pain and dementia caregiving.
- Physician review was built into the development process: a clinical team "aligned on how Catherine could drive value, identified relevant topics, reviewed question and answer pairs for accuracy, and tested the platform through prompts and user exercises."
- The platform was fed a curated knowledge base rather than open-ended generative responses.

**Patient interaction workflow (reconstructed from available documentation):**

```
Step 1: Patient/caregiver accesses Catherine via web (catherine.brado.ai)
        |
Step 2: Catherine initiates empathetic conversation
        | - Asks specific questions to keep conversations going
        | - "Knows when it's appropriate to show empathy, compassion or even curiosity"
        |
Step 3: Journey-based navigation
        | - Identifies where the patient is in their healthcare journey
        | - Tests different audience segments
        | - Provides tailored support and curated information
        |
Step 4: Clinical information delivery
        | - Orthopedic: "Catherine will tell you if your condition is urgent or not"
        | - Points patients "in the right direction to speak to a doctor"
        | - Dementia: Helps caregivers "navigate the dementia caregiver journey"
        |
Step 5: Referral / scheduling
        | - "Leads to better retention rates and improved satisfaction"
        | - Finds "prospective patients at different points in the orthopedic journey"
        |
Step 6: Learning loop
        | - "Learns from their activity, using this learning to enhance the depth
        |    of guidance and increase the rate at which people chat"
```

**What questions does it ask?** Specific details not publicly documented. From the platform description: it asks questions designed to identify the patient's position in the healthcare journey (e.g., dementia caregiving stage, orthopedic pain severity and type) and keep conversations engaged.

**Clinical information provided:** Curated, clinician-reviewed content about dementia caregiving and orthopedic conditions. Not open-ended generative clinical advice. The platform provides urgency assessment ("tells you if your condition is urgent or not") and directional guidance ("points you in the right direction to speak to a doctor").

**Triage decisions:** Catherine makes urgency assessments and directional referrals. This is not symptom-based algorithmic triage in the Infermedica/Ada Health sense -- it is journey-based navigation toward appropriate care settings.

**Guardrails:** The available documentation emphasizes:
- Physician review of Q&A pairs for accuracy
- HIPAA compliance
- Clinically aligned conversations
- Knowledge base is curated, not open-ended generative

**Published validation data:** No peer-reviewed clinical validation studies identified. Brado reports "meaningful results" and "notable metrics" from the first three months but specific outcome data (accuracy rates, patient safety events, clinical appropriateness scores) are not publicly available. This is a significant governance gap for a patient-facing AI system.

**TELOS governance relevance:** Catherine represents a "curated knowledge base + conversational AI" architecture where the boundary corpus is hand-built by clinicians. The guardrails are content-based (restricted knowledge domain) rather than algorithmic (real-time safety filtering). This creates governance vulnerability if the knowledge base has gaps or if the conversational AI generates responses that interpolate beyond the curated content.

---

### 1.2 Epic MyChart In-Basket ART (Augmented Response Technology)

**What it is:** A generative AI feature integrated into Epic's EHR that pre-drafts responses to patient MyChart messages. Clinicians review, edit, and send (or discard) the drafts.

**Model and infrastructure:**
- **Model:** OpenAI GPT-4o, accessed through Microsoft Azure OpenAI Service within a HIPAA-compliant pipeline
- **Environment:** Azure OpenAI Service (HIPAA BAA in place between Microsoft and Epic)
- **Data accessed:** Patient's latest MyChart message text, trust-specific prompts, and information from the patient's medical record including current prescriptions and recent results. As of late 2025, Epic expanded context to include results, medications, and other details a doctor might need.

**Clinician review workflow:**

```
Step 1: Patient sends message via MyChart portal
        |
Step 2: Epic routes message to clinician's In-Basket
        |
Step 3: ART system triggers automatically
        |  - Reads: patient message text
        |  - Reads: patient chart data (medications, recent results, problems)
        |  - Reads: trust-specific prompt configuration
        |  - Calls: Azure OpenAI Service (GPT-4o)
        |
Step 4: AI generates draft response
        |  - Personalized based on chart data
        |  - Designed to adopt the clinician's communication style
        |
Step 5: Draft appears in clinician's In-Basket alongside patient message
        |  - Clinician sees: original patient message + AI-generated draft
        |  - Clear visual indication that draft is AI-generated
        |
Step 6: Clinician review decision
        |  - OPTION A: Accept draft as-is and send
        |  - OPTION B: Edit draft, then send
        |  - OPTION C: Discard draft entirely and write own response
        |
Step 7: Response sent to patient via MyChart
        |  - Patient sees response attributed to their clinician
        |  - No indication to patient that AI was involved in drafting
        |     (varies by organization policy)
```

**Published quality/accuracy data:**

| Metric | Finding | Source |
|--------|---------|--------|
| Empathy ratings | AI responses 125% more likely to be rated empathetic vs. physician-drafted | UC San Diego study, 2024 |
| Readability | AI scored statistically significantly higher than doctors | Multiple studies, systematic review 2025 |
| Draft utilization rate | 12-20% of generated drafts were used by clinicians | NPJ Digital Medicine, 2025 |
| Time savings | ~30 seconds per message for nurses | Mayo Clinic pilot |
| Draft volume | >1 million drafts generated monthly across 150+ organizations | Epic, August 2024 |
| Editing patterns | Most edits removed recommendations for appointments or corrected inaccuracies | Systematic review, 2025 |

**OSU Wexner Medical Center pilot:** Wexner conducted a pilot of Microsoft Dragon Ambient eXperience Copilot (DAX Copilot) for ambient clinical documentation, not specifically Epic MyChart ART. Results: AI saved up to 4 minutes per visit, 80% of clinicians completed the pilot, and the medical center allowed continued use afterward. No published study specifically on ART quality metrics from Wexner was identified.

**Critical observation -- low utilization:** The 12-20% utilization rate is telling. Despite saving time, clinicians discard 80-88% of AI-generated drafts. This suggests either quality concerns, trust deficit, or workflow friction. The drafts may excel at empathy and tone but fall short on clinical accuracy for the specific patient context.

**Regulatory status:** ART is not FDA-regulated. It operates as a clinical communication tool, not a diagnostic device. The clinician review step provides the physician supervision layer. However, if clinicians accept drafts without meaningful review (rubber-stamping), the supervision safeguard is nominal rather than substantive.

---

### 1.3 General Patient-Facing AI Chatbot Taxonomy

Patient-facing AI chatbots in healthcare fall into several distinct categories with different risk profiles:

| Category | Examples | Risk Level | Regulatory Status |
|----------|----------|------------|-------------------|
| **Symptom checkers / triage** | Ada Health, Babylon, Buoy, Infermedica, Symptoma | HIGH | Some FDA-cleared (SaMD); most unregulated |
| **Appointment scheduling** | Clearstep, various vendor chatbots | LOW | Not regulated as medical devices |
| **Mental health support** | Woebot, Wysa | MODERATE-HIGH | Woebot: FDA Breakthrough Device designation |
| **Patient portal AI** | Epic MyChart ART | MODERATE | Not FDA-regulated; clinician review required |
| **Care navigation** | Catherine (Brado), Clearstep | MODERATE | Not FDA-regulated |
| **FAQ / administrative** | Various vendor chatbots | LOW | Not regulated |
| **Medication management** | Various | MODERATE | Depends on clinical claims |
| **General-purpose LLMs used by patients** | ChatGPT, Claude, Gemini, Copilot, Grok | HIGH (uncontrolled) | Not regulated as medical devices |

**Key technology distinction:** AI chatbots based on NLP and generative AI (LLMs such as GPT, Claude, Gemini) produce flexible, human-sounding responses but carry hallucination risk. Rule-based chatbots following predefined algorithms are more predictable but less adaptable.

---

### 1.4 Known Failure Modes (ECRI 2026 Hazard List)

ECRI named **"Misuse of AI Chatbots"** as the #1 health technology hazard for 2026. Specific documented failure modes:

| Failure Mode | Description | Severity |
|--------------|-------------|----------|
| **Incorrect diagnoses** | Chatbots suggest wrong diagnoses that could lead to delayed or inappropriate treatment | Critical |
| **Unnecessary testing** | Recommendations for tests that are not clinically indicated | Moderate |
| **Inventing body parts** | LLM hallucination produces references to anatomical structures that do not exist | Critical |
| **Subpar medical supply recommendations** | Promoting products that are inadequate for the clinical situation | Moderate |
| **Incorrect safety guidance** | ECRI test: chatbot incorrectly stated electrosurgical return electrode placement over shoulder blade was appropriate -- would risk patient burns | Critical |

**Root cause:** LLMs are not validated for healthcare purposes. They produce "human-like and expert-sounding responses" that create false confidence. The tools "are not regulated as medical devices nor validated for healthcare purposes but are increasingly used by clinicians, patients, and healthcare personnel."

**ECRI recommendation:** "Patients, clinicians, and other chatbot users can reduce risk by educating themselves on the tools' limitations and always verifying information obtained from a chatbot with a knowledgeable source."

---

### 1.5 Regulatory Frameworks for Patient-Facing AI

#### FDA

**Current status (as of 2026):**
- Over 1,250 AI-enabled medical devices authorized for marketing in the US
- 97% cleared via 510(k) pathway (substantial equivalence to predicate device)
- FDA created a new regulatory category with Viz.ai: "Radiological Computer Aided Triage and Notification Software"

**CDS exemption under 21st Century Cures Act (Section 3060):**
Clinical Decision Support software is exempt from FDA regulation as a medical device IF it meets ALL four criteria:
1. Displays, analyzes, or prints medical information
2. Supports or provides recommendations to a healthcare professional (not directly to patients)
3. Enables the professional to independently review the basis for recommendations (transparency)
4. Does NOT acquire, process, or analyze medical images, signals from IVD devices, or patterns from signal acquisition systems

**January 2026 update:** Revised FDA guidance superseding the 2022 version clarifies which CDS functions are regulated vs. exempt. Software functions that match patient data with current treatment guidelines for common illnesses may be exempt. Functions making specific diagnostic or treatment recommendations fall under FDA regulation.

**Patient-facing chatbots:** Most patient-facing chatbots currently operate in a regulatory gray zone. They are not FDA-cleared, but they may make de facto clinical recommendations. Generative AI devices that "treat or diagnose a psychiatric condition or substitute for a mental healthcare provider" fall under FDA's regulatory purview.

**FDA questions on GenAI chatbots (November 2025):** FDA raised questions about GenAI-enabled chatbots at an expert panel, signaling increased regulatory interest in this category.

#### FTC

**Operation AI Comply (September 2024):** FTC launched enforcement sweep against deceptive AI claims and practices.

**Healthcare-specific enforcement:**
- **Pieces Technologies / Texas AG (August 2024):** First-ever state AG settlement involving healthcare AI accuracy claims. Pieces had claimed its AI products had a "critical and severe hallucination rate" of "<.001%" and "<1 per 100,000" -- claims the Texas AG alleged were false and misleading. Settlement required: accuracy metric disclosure, known limitation disclosure, risk-to-patient disclosure, and misuse prevention documentation. **Five-year compliance term.** This creates binding precedent for healthcare AI accuracy claims.

- **DoNotPay (January 2025):** FTC settled with AI service claiming to be "world's first robot lawyer" -- product was not sufficiently trained on applicable law.

- **Consumer-facing AI chatbot orders (September 2025):** FTC issued orders to seven major tech companies providing consumer-facing AI chatbots, requesting information on safety assessments, data collection, and usage mitigations.

**FTC principle:** Existing Section 5 authority (unfair or deceptive acts or practices) applies to AI claims. No new legislation needed for enforcement.

#### State Laws

| State | Law | Key Requirements | Effective |
|-------|-----|-------------------|-----------|
| **California** | AB 489 | Prohibits AI from using terms/design suggesting it holds a healthcare license; bars AI advertising suggesting care by licensed natural person | Jan 1, 2026 |
| **California** | SB 243 | Regulates "companion chatbots" providing emotional support; requires AI disclosure; mandates protocols for suicidal ideation responses | 2025 |
| **California** | 2024 AI disclosure law | Healthcare providers must include disclaimer when communications were generated by AI | 2025 |
| **Colorado** | CAIA (Colorado AI Act) | Comprehensive consumer protection for AI; postponed to June 30, 2026 | June 30, 2026 |
| **Multiple states** | 21 bills, 7 laws in 2024 | Five of seven passed laws had specific mental health AI chatbot focus | Various |

**Trend:** 46 states have some form of AI healthcare regulation in development or enacted. The regulatory patchwork creates significant compliance complexity for multi-state health systems like BSMH.

---

### 1.6 HIPAA Exposure When Patients Share PHI with Chatbots

**The core problem:** Patient-facing AI chatbots may gather symptoms, health histories, insurance details -- all PHI under HIPAA. The HIPAA exposure depends on the chatbot's relationship to the covered entity.

| Scenario | HIPAA Status | Risk Level |
|----------|-------------|------------|
| Hospital-deployed chatbot (Catherine, Epic ART) | Covered -- BAA required with vendor | Managed if compliant |
| Third-party health chatbot (not under BAA) | Not covered by HIPAA | HIGH -- patients may not understand data is not protected |
| Patient uses ChatGPT/Claude to discuss health | Not covered by HIPAA | HIGH -- free versions may use data for training; no BAA |
| Patient shares screenshots of PHI with chatbot | PHI leaves covered entity control | CRITICAL |

**Specific HIPAA risks:**
- **Improper disclosure:** Chatbot backends may store or transmit PHI without proper encryption
- **Secondary data use:** AI training on patient conversations could constitute unauthorized use
- **Data retention:** Patient conversations may be retained beyond clinical necessity
- **Model memorization:** Some AI algorithms "inadvertently retain PHI from training data, raising concerns about unintended data leaks"
- **Free AI tools:** Free versions of ChatGPT are NOT HIPAA compliant. OpenAI does not sign BAAs for free accounts. Patient data may be used to train the model.

**January 2025 HIPAA Security Rule update (proposed):** First major update in 20 years. Removes distinction between "required" and "addressable" safeguards. Introduces stricter encryption, risk management, and resilience requirements. AI systems processing PHI will be subject to enhanced standards. 67% of healthcare organizations report being unprepared.

---

## 2. AI-Assisted Medical Coding and Billing

### 2.1 Technical Pipeline: Clinical Note to Billing Code

The end-to-end pipeline from clinical documentation to billing code submission involves the following technical stages:

```
STAGE 1: Clinical Documentation Creation
         |
         |  Sources: Progress notes, discharge summaries, operative reports,
         |           pathology reports, radiology reports, vital signs, lab results,
         |           medication orders, nursing notes
         |
         |  Format: Primarily unstructured free text (dictated or typed),
         |          plus structured EHR data fields
         |
STAGE 2: NLP/AI Processing
         |
         |  a) Document ingestion and parsing
         |     - OCR if needed for scanned documents
         |     - Structured data extraction from EHR fields
         |     - Unstructured text processing via NLP
         |
         |  b) Clinical concept extraction
         |     - Named Entity Recognition (NER) for diagnoses, procedures,
         |       medications, anatomical sites
         |     - Contextual analysis (negation detection, uncertainty markers,
         |       temporal references, attribution to other providers)
         |
         |  c) Code mapping
         |     - Map extracted concepts to:
         |       * ICD-10-CM diagnosis codes (70,000+ codes)
         |       * CPT procedure codes (10,000+ codes)
         |       * HCC (Hierarchical Condition Category) risk adjustment codes
         |       * DRG (Diagnosis Related Group) assignments
         |       * E/M (Evaluation and Management) level selection
         |
         |  d) Confidence scoring
         |     - Each suggested code assigned confidence score
         |     - Threshold-based routing: high confidence = auto-queue;
         |       low confidence = flag for human review
         |
STAGE 3: Human Review / Quality Assurance
         |
         |  Reviewed by: Certified medical coders, CDI specialists,
         |               and/or clinicians depending on workflow
         |
         |  Review actions: Accept, modify, reject AI suggestions
         |
STAGE 4: Code Finalization and Claim Submission
         |
         |  - Finalized codes assembled into claim format (CMS-1500 or UB-04)
         |  - Submitted to payer (Medicare, Medicaid, commercial)
         |
STAGE 5: Adjudication
         |  - Payer reviews claim
         |  - Approve, deny, or request additional documentation
         |
STAGE 6: Denial Management
         |  - If denied: AI analyzes denial reason, evaluates documentation
         |  - Automated appeal letter generation
         |  - Resubmission or escalation
```

**LLM accuracy for code generation (published benchmarks):**

| Model | ICD-9-CM Exact Match | ICD-10-CM Exact Match | CPT Exact Match |
|-------|---------------------|-----------------------|-----------------|
| GPT-4 | 45.9% | 33.9% | 49.8% |
| GPT-3.5 | Lower | Lower | Lower |
| Llama2-70b Chat | <3% | <3% | <3% |

Source: NEJM AI, 2024. These exact-match rates underscore why human oversight is essential -- LLMs "often generated codes that were conceptually similar but lacked the precision required for clinical use, sometimes producing generalized or even fabricated codes."

---

### 2.2 Major AI Coding Systems in Production

#### Solventum 360 Encompass Autonomous Coding System (formerly 3M)

**Architecture:** Deep learning neural network models + Natural Language Understanding (NLU)

**Autonomy level:** FULL AUTONOMOUS for qualifying encounters. The system works behind the scenes, sending all outpatient visits through a "chart confidence workflow." Visits that pass all system and client-defined automation criteria are "fully automated and final coded ready for the next step in the billing process, without any coder interaction."

**Human review:** When visits don't meet confidence thresholds, a "semi-autonomous workflow" activates. Coders receive contextual information explaining why the visit didn't qualify for automation. Facilities determine the percentage of qualified visits presented for QA coder review.

**Code types generated:** CPT and ICD-10-CM codes

**Key risk:** This is the closest to fully autonomous coding in production. The system can submit codes to billing without any human reviewing the specific encounter.

#### SmarterDx (Clinical AI Prebill)

**Architecture:** Hybrid AI combining language models for unstructured text with purpose-built models for structured clinical indicators (labs, vitals, medications). Reviews 30,000+ data points per patient visit (chart notes, labs, medications, vital signs). Claims to analyze 150,000 data points per hospitalization.

**What it does:** Automated second-pass review of every patient chart (PreBill platform). Flags clinical evidence lacking optimal associated diagnoses AND identifies diagnoses without supporting evidence. "Recreates physician thought processes."

**Autonomy level:** SUGGEST with automated review. Does not auto-code claims but performs comprehensive pre-bill review flagging missed diagnoses and unsupported diagnoses.

**Revenue impact focus:** Explicitly positioned for revenue optimization -- described as "the AI-powered revenue boost hospitals didn't know they needed."

#### Fathom Health

**Architecture:** Deep learning + NLP for medical coding automation

**Autonomy level:** FULL AUTONOMOUS for qualifying encounters. "Automatically processing charts and sending medical records directly to billing with no human intervention." Handles "over 93% of encounters at superhuman accuracy."

**Code types:** E/M levels, procedure codes, ICD codes, provider assignment, deficiencies

**Claimed accuracy:** 90%+ automation rate validated by KLAS Research. 100% customer satisfaction rating.

**Specialties:** Primary care, radiology, multi-specialty

**Cost impact:** Reduces coding spend by 30-50%, cuts turnaround from days/weeks to minutes

#### Iodine Software (now Waystar)

**Architecture:** Clinical Intelligence Engine analyzing 160 million+ patient encounters and 1.5 billion+ medical concepts. Integrates live vitals, labs, and medications.

**Products:**
- **Concurrent** (formerly IodineCDI): Prioritizes charts with inconsistencies during patient stay
- **PreBill** (formerly Retrospect): Post-discharge auditing for revenue capture and denial risk
- **Interact**: Physician query management

**Autonomy level:** SUGGEST -- CDI workflow optimization. The system identifies at-risk diagnoses, recommends compliant Clinical Validation Queries (CVQs), and validates diagnoses against real-time clinical data. If documentation doesn't match clinical evidence (e.g., "acute kidney injury" documented but creatinine values trending normally), it alerts CDI teams.

#### Stanson Health (Premier / PINC AI)

See Section 2.4 for detailed BSMH-specific analysis.

---

### 2.3 The "Coding Arms Race" Problem

#### How ambient AI scribes drive revenue optimization

A landmark policy brief published in NPJ Digital Medicine (2025) documented the emerging "coding arms race" between providers using AI to optimize documentation and payers responding with automated downcoding:

**Evidence of revenue impact from ambient AI scribes:**

| Health System | Metric | Change |
|---------------|--------|--------|
| Riverside Health (VA) | Physician wRVUs | +11% |
| Riverside Health (VA) | Documented HCC diagnoses per encounter | +14% |
| Northwestern Medicine | High-level E/M visits | Increase (unspecified) |
| Texas Oncology | Documented diagnoses per encounter | 3.0 to 4.1 (+37%) |
| General | RVU increase per physician per week | +1.81 RVUs ($3,044/year at 2025 Medicare rate) |

**The core tension:** "Potential rises in wRVUs or HCCs do not necessarily mean upcoding; they often reflect previously omitted details now captured." However, "if revenue optimization becomes [ambient AI's] defining purpose, we risk repeating a familiar cycle -- an arms race that ends with higher administrative friction, payer pushback, and little improvement at the bedside."

**The collapse of distance between care and coding:** "Ambient AI collapses the distance between care and coding more completely than any prior documentation tool." The scribe simultaneously documents clinical care AND captures billing-relevant details, creating an inherent dual-purpose that blurs the line between clinical completeness and revenue optimization.

#### Payer response: Automated downcoding

**Cigna Policy R49 (announced 2025):** Beginning October 1, 2025, Cigna would automatically downcode Level 4-5 E/M visits (99204-99205, 99214-99215, 99244-99245) by one level unless documentation clearly supports higher complexity. Aetna announced a similar policy. This is a direct payer response to observed increases in high-level E/M billing.

**Physician pushback:** AAFP and other physician groups opposed the policy. Cigna temporarily paused Policy R49 implementation.

**Arms race dynamics:** Providers use AI to capture more complete documentation -> payers use AI to flag and downcode higher-billed claims -> providers adjust AI to generate documentation specifically supporting higher codes -> payers tighten criteria further. This is the classic arms race the policy brief warns about.

#### CMS/OIG enforcement landscape

**OIG audit focus:** Of 44 managed care audits conducted by HHS OIG since 2017, 42 have focused on diagnosis coding accuracy. Risk adjustment coding is a primary enforcement target because "there is an incentive to upcode or submit unsupported diagnosis codes" for higher capitated Medicare Advantage rates.

**False Claims Act exposure:** Submitting unsupported diagnostic codes constitutes false claims. Penalties: fines up to 3x the program's loss plus $11,000 per claim. Even unintentional errors trigger costly audits. The 60-day overpayment rule requires self-reporting once overpayments are identified.

**RADV audits:** CMS initiated risk adjustment audits reaching Payment Year 2018 -- the first year extrapolation applies, meaning a small sample of errors can be projected across the entire claims portfolio.

**2025-2026 enforcement outlook:** DOJ, HHS-OIG, and CMS are deploying machine learning tools to "rapidly flag outlier billing, telehealth spikes, and risk-adjustment irregularities -- increasing the speed and scope of investigations." The U.S. healthcare fraud takedown in 2025 demonstrates continued aggressive enforcement.

**AI coding oversight gap:** As of June 2025, there are "no current regulations mandating universal human review of AI-autonomously coded claims." However, agencies have "signaled that due diligence should be conducted on AI coded claims," and payers are updating contracts to require human validation of AI-generated codes.

**Governance recommendation from policy brief:** "Physicians and health systems must retain authorship by disabling auto-accept and requiring active review of diagnoses and billing elements." Random audits comparing audio to signed notes can detect "chart-stuffing" drift.

---

### 2.4 Stanson Health HCC Alerts at BSMH

**What it is:** Stanson Health (now part of Premier/PINC AI) provides clinical decision support alerts for HCC (Hierarchical Condition Category) coding, integrated into Epic's EHR workflow at BSMH.

**Technical implementation:**

```
Step 1: Clinician opens patient chart in Epic
        |
Step 2: Stanson AI analyzes EHR data in real-time
        |  - Reviews: clinical notes, lab results, medications, vitals
        |  - Checks for MEAT criteria (Monitored, Evaluated,
        |    Assessed/Addressed, Treated)
        |  - Compares documentation against potential HCC codes
        |
Step 3: Alert triggering
        |  - Alert fires ONLY if patient meets specific clinical profile
        |  - Evidence-based, clinician-designed logic
        |  - "Only provides recommendations relevant to the clinician's decision"
        |  - Fires during ordering workflow (point-of-care)
        |
Step 4: Clinician sees alert in Epic
        |  - Alert suggests whether a new/different HCC code should be considered
        |  - Actionable: designed to limit administrative burden
        |  - Clinician can: accept (add code to visit diagnosis/problem list)
        |                   or dismiss
        |
Step 5: Analytics tracking
        |  - Stanson analytics platform tracks alert trigger frequency
        |  - Tracks "follow" rates (acceptance rates)
        |  - Data used to optimize alert performance
```

**BSMH results:**
- **30 HCC alerts** activated
- **35,000+ HCC categories documented** in just six months
- **"Significant financial impact"** (specific dollar amounts not publicly disclosed)
- Implementation followed a collaborative approach tailored to BSMH's EHR configuration

**Alert design philosophy:** Alerts are designed to be actionable while limiting alert fatigue. This is achieved through clinical logic that restricts triggering to cases with genuine documentation gaps, rather than firing on every possible HCC opportunity.

**Revalidation capability:** Next-generation technology includes "revalidation" to suggest more accurate HCC codes based on current patient data in the EHR, addressing the annual recertification requirement for HCC codes.

**Governance concern:** The 35,000+ HCC categories captured in 6 months is an extraordinary number. While this may reflect legitimate documentation improvement (previously omitted conditions now properly coded), it also represents a massive revenue uplift through risk adjustment. The line between "complete documentation" and "aggressive coding" depends on whether each captured HCC is supported by clinical evidence meeting MEAT criteria. Audit exposure is significant if any material portion lacks supporting documentation.

---

### 2.5 CMS WISeR (Wasteful and Inappropriate Services Reduction) System

**What it is:** A six-year CMS model using AI/ML to implement prior authorization for select Medicare FFS services. Launched January 1, 2026.

**Geographic scope:** Six pilot states: New Jersey, Ohio, Oklahoma, Texas, Arizona, Washington

**Services covered:**
- Skin and tissue substitutes
- Electrical nerve stimulators and knee arthroscopy for knee osteoarthritis
- Electrical nerve stimulator implants
- Epidural steroid injections for pain management (excluding facet joint)
- Percutaneous vertebral augmentation for vertebral compression fracture
- Percutaneous image-guided lumbar decompression for spinal stenosis

**Technology vendor participants (6):**
1. Cohere Health, Inc.
2. Genzeon Corporation
3. Humata Health, Inc.
4. Innovaccer Inc.
5. Virtix Health LLC
6. Zyter Inc.

**Technical workflow:**

```
PATHWAY A: Prior Authorization (proactive)
         |
Step 1:  Provider submits PA request
         |  - Methods: mail, fax, electronic portal
         |
Step 2:  Technology vendor receives request
         |  - AI screens for medical necessity
         |  - Clinical review using existing Medicare coverage policies
         |  - Standard requests: processed within 3 days
         |  - Urgent requests: processed within 2 days
         |
Step 3:  Decision
         |  - Affirmed: Valid for 120 calendar days
         |  - Non-affirmed: Provider can resubmit (unlimited),
         |    request peer-to-peer review
         |  - When denied: HUMAN CLINICIAN with relevant expertise must be involved
         |
PATHWAY B: Post-Service Review (if no PA submitted)
         |
Step 1:  Service furnished without prior authorization
         |
Step 2:  Automatic prepayment medical review initiated by vendor
         |  - Documentation request sent to provider
         |  - Provider has 45 days to respond
         |
Step 3:  AI + clinical review of documentation
         |
Step 4:  Approve or deny claim
```

**Gold carding (planned mid-2026):** Clinicians with consistent approval histories would be exempted from future PA or pre-payment review -- creating an AI-managed trust score for providers.

**TELOS relevance:** WISeR is the first major CMS deployment of AI in claims adjudication. It creates a direct government AI-to-provider interaction that carries legal force. The 120-day authorization window, unlimited resubmission policy, and peer-to-peer review rights create specific governance requirements for any AI system assisting providers with WISeR compliance.

---

## 3. Action Chain: Patient-Facing AI

```
Step 1: PATIENT INITIATES CONTACT
        |  Channels: Web chat (Catherine), MyChart message (Epic ART),
        |            phone (IVR + AI), SMS, mobile app
        |
        |  REGULATORY: HIPAA applies from first contact if PHI is shared.
        |              BAA must be in place with chatbot vendor.
        |              State AI disclosure laws may require notice that AI is involved.
        |  AUDIT TRAIL: Timestamp, channel, session identifier
        |
Step 2: IDENTITY VERIFICATION
        |  - Patient portal: Authentication via MyChart credentials (pre-verified)
        |  - Web chat: May be anonymous (Catherine) or require login
        |  - Phone: Multi-factor verification (DOB, MRN, address)
        |
        |  REGULATORY: HIPAA minimum necessary standard.
        |              Must verify identity before exposing PHI.
        |              21 CFR Part 11 for electronic signatures if applicable.
        |  AUDIT TRAIL: Verification method, success/failure, timestamp
        |  LIABILITY: Incorrect identity match -> PHI breach exposure
        |
Step 3: INTENT RECOGNITION
        |  - NLP/LLM classifies patient intent
        |  - Categories: clinical question, appointment request, billing inquiry,
        |    prescription refill, urgent/emergency, emotional distress
        |
        |  REGULATORY: FDA SaMD if intent classification constitutes triage
        |              or diagnostic function.
        |              FTC Section 5 if classification is unreliable.
        |  AUDIT TRAIL: Classified intent, confidence score, raw input text
        |  LIABILITY: Misclassification of emergency as non-urgent = patient harm risk
        |
Step 4: INFORMATION RETRIEVAL
        |  - Access patient chart (Epic ART: medications, results, problems)
        |  - Access knowledge base (Catherine: curated clinical content)
        |  - Access scheduling systems (available appointments)
        |
        |  REGULATORY: HIPAA minimum necessary; only access data needed for response.
        |              HIPAA audit log for every PHI access.
        |  AUDIT TRAIL: Data elements accessed, timestamp, purpose
        |  LIABILITY: Accessing unnecessary PHI = HIPAA violation
        |
Step 5: RESPONSE GENERATION
        |  - LLM generates draft response (Epic ART: GPT-4o)
        |  - Knowledge base response (Catherine: curated content matching)
        |  - Structured response (scheduling bots: deterministic)
        |
        |  REGULATORY: If response contains clinical advice -> potential FDA SaMD.
        |              CDS exemption requires transparency of reasoning basis.
        |              California AB 489: cannot suggest AI holds healthcare license.
        |  AUDIT TRAIL: Model version, prompt template, generated output, input context
        |  LIABILITY: Hallucinated clinical information = malpractice exposure
        |
Step 6: SAFETY CHECK / GUARDRAILS
        |  - Emergency detection (suicide, chest pain, stroke symptoms)
        |  - Clinical scope check (is response within permitted domain?)
        |  - Hallucination detection (does response reference real information?)
        |  - Harmful content filtering
        |  - Regulatory compliance check (disclaimers, disclosures)
        |
        |  REGULATORY: State mental health chatbot laws (CA SB 243).
        |              Duty to warn / mandatory reporting if applicable.
        |  AUDIT TRAIL: Safety check results, any flags triggered, override decisions
        |  LIABILITY: Failed emergency detection = wrongful death exposure
        |
Step 7: CLINICIAN REVIEW (if applicable)
        |  - Epic ART: Clinician reviews, edits, sends or discards
        |  - Catherine: No clinician review (automated response)
        |  - Symptom checkers: No clinician review (direct to patient)
        |
        |  REGULATORY: Physician supervision requirement. Clinician who sends
        |              message bears liability regardless of AI involvement.
        |  AUDIT TRAIL: Reviewer identity, edits made, accept/reject decision, timestamp
        |  LIABILITY: "A physician who in good faith relies on an AI/ML system
        |             may still face liability if actions fall below standard of care"
        |
Step 8: RESPONSE DELIVERY TO PATIENT
        |  - Via authenticated channel (MyChart, secure chat, SMS)
        |
        |  REGULATORY: HIPAA transmission security. Encryption in transit required.
        |              California: disclosure that communication was AI-generated.
        |  AUDIT TRAIL: Delivery confirmation, timestamp
        |
Step 9: ESCALATION / SCHEDULING / REFERRAL
        |  - Emergency detected -> immediate escalation to human agent / 911 guidance
        |  - Clinical question beyond scope -> schedule with provider
        |  - Appointment requested -> schedule in EHR
        |  - Referral needed -> generate referral in system
        |
        |  REGULATORY: EMTALA if emergency. Duty to refer for conditions beyond scope.
        |  AUDIT TRAIL: Escalation reason, destination, outcome, timestamp
        |  LIABILITY: Failure to escalate emergency = EMTALA violation + malpractice
        |
Step 10: DOCUMENTATION IN EHR
         |  - Conversation summary logged to patient chart
         |  - Clinical actions documented
         |  - AI involvement flagged in metadata
         |
         |  REGULATORY: Medical records requirements (state law).
         |              21st Century Cures Act information blocking rules.
         |  AUDIT TRAIL: Full conversation record, AI model version, all decisions
```

---

## 4. Action Chain: AI-Assisted Coding

```
Step 1: CLINICAL NOTE CREATED / FINALIZED
        |  - Clinician completes encounter documentation
        |  - Ambient AI scribe generates draft note (if applicable)
        |  - Clinician signs/attests note
        |
        |  REGULATORY: Clinician attestation required. Cannot delegate to AI.
        |              Medicare CoP for documentation standards.
        |  AUDIT TRAIL: Note author, attestation timestamp, AI scribe involvement (Y/N),
        |               edits made to AI draft before signing
        |  LIABILITY: Clinician owns note content regardless of AI drafting.
        |             If AI inflates documentation -> clinician liability for attestation.
        |
Step 2: NLP EXTRACTION OF DIAGNOSES / PROCEDURES
        |  - AI system ingests finalized clinical note
        |  - NLP extracts: diagnoses mentioned, procedures performed,
        |    medications administered, lab/test results, clinical reasoning
        |  - Structured data extracted from EHR fields
        |  - Negation detection: "no evidence of..." must NOT generate codes
        |  - Temporal context: "history of..." vs. "current..."
        |
        |  REGULATORY: No specific regulation on NLP extraction step.
        |              Accuracy of extraction directly impacts billing compliance.
        |  AUDIT TRAIL: Raw extracted concepts, source text spans,
        |               NLP model version, confidence scores
        |  LIABILITY: Extraction errors propagate through entire pipeline
        |
Step 3: CODE SUGGESTION GENERATION
        |  - Map extracted concepts to ICD-10-CM, CPT, HCC, DRG codes
        |  - Cross-reference with coding guidelines (ICD-10 Official Guidelines,
        |    CPT Assistant, CMS coding manual)
        |  - E/M level determination based on documented complexity
        |
        |  REGULATORY: CMS coding guidelines have force of law for Medicare claims.
        |              AMA CPT guidance for procedure coding.
        |              Code specificity requirements (highest level of specificity).
        |  AUDIT TRAIL: Suggested codes, mapping rationale, guideline references,
        |               confidence scores per code
        |  LIABILITY: Incorrect code suggestions that persist to submission = FCA exposure
        |
Step 4: CONFIDENCE SCORING / FLAG FOR REVIEW
        |  - Each code assigned confidence score
        |  - Routing logic:
        |    HIGH confidence (above threshold): auto-queue for billing
        |      (Solventum, Fathom: may go directly to claim submission)
        |    MODERATE confidence: routed to coder with AI suggestions highlighted
        |    LOW confidence: flagged for senior coder or CDI specialist review
        |    CONFLICTING evidence: flagged for physician query
        |
        |  REGULATORY: No mandated human review requirement (as of June 2025).
        |              However, CMS has signaled due diligence expectations.
        |              Payer contracts increasingly require human validation.
        |  AUDIT TRAIL: Confidence scores, routing decision, threshold values,
        |               algorithm version
        |  LIABILITY: Auto-coding without human review = highest FCA risk
        |
Step 5: CODER / CLINICIAN REVIEW
        |  - Certified coder reviews AI suggestions against documentation
        |  - CDI specialist may query physician for documentation gaps
        |  - Physician may be asked to clarify or add diagnoses
        |  - Stanson Health alerts fire at point of care (before this step)
        |
        |  REGULATORY: CMS Program Integrity Manual requirements.
        |              Coder credential requirements (CPC, CCS, etc.).
        |              Physician query compliance (no leading queries).
        |  AUDIT TRAIL: Reviewer identity, credentials, accept/modify/reject per code,
        |               time spent, query text if issued
        |  LIABILITY: Coder accepting incorrect AI suggestions shares liability.
        |             "AI cannot sign attestations or testify in an audit --
        |              the human reviewer remains accountable."
        |
Step 6: CODE FINALIZATION
        |  - Final code set assembled
        |  - Claim form generated (CMS-1500 or UB-04)
        |  - Pre-submission edits check (scrubbing)
        |  - Compliance checks (bundling rules, modifier requirements,
        |    medical necessity indicators)
        |
        |  REGULATORY: CMS claim submission requirements.
        |              False Claims Act: "knowingly" standard includes
        |              deliberate ignorance and reckless disregard.
        |              60-day overpayment rule for self-identified errors.
        |  AUDIT TRAIL: Final code set, claim form, scrubbing results,
        |               compliance check outcomes
        |
Step 7: CLAIM SUBMISSION
        |  - Electronic submission to clearinghouse
        |  - Clearinghouse routes to payer (Medicare MAC, commercial payer)
        |  - Payer adjudication (may use AI on payer side)
        |
        |  REGULATORY: HIPAA transaction standards (EDI 837).
        |              CMS enrollment and billing privileges.
        |  AUDIT TRAIL: Submission timestamp, clearinghouse confirmation,
        |               payer receipt acknowledgment
        |
Step 8: DENIAL MANAGEMENT / APPEAL IF REJECTED
        |  - Payer issues denial with reason code
        |  - AI evaluates denial reason against documentation
        |  - Automated appeal letter generation:
        |    a) Analyzes medical necessity criteria
        |    b) Reviews clinical documentation
        |    c) Provides recommendations based on historical data
        |    d) Generates appeal narrative
        |  - If justified: automated appeal submission
        |  - If not justified: flag for write-off or further review
        |
        |  REGULATORY: Timely filing requirements (varies by payer).
        |              CMS appeal rights and deadlines.
        |              State prompt payment laws.
        |  AUDIT TRAIL: Denial reason, appeal rationale, AI-generated vs. human-drafted,
        |               appeal outcome, recovery amount
        |  LIABILITY: Appealing claims known to be incorrect = additional FCA exposure
```

---

## 5. Regulatory Exposure Matrix

### Patient-Facing AI: Regulatory Exposure by Action Step

| Step | HIPAA | FDA | FTC | State Law | CMS | Physician Supervision |
|------|-------|-----|-----|-----------|-----|----------------------|
| 1. Patient initiates | PHI collection triggers | -- | -- | AI disclosure may be required | -- | -- |
| 2. Identity verification | Minimum necessary | -- | -- | -- | -- | -- |
| 3. Intent recognition | -- | SaMD if triage function | Section 5 if unreliable | -- | -- | -- |
| 4. Information retrieval | Access logging required | -- | -- | -- | -- | -- |
| 5. Response generation | PHI in response | SaMD if clinical advice | Deceptive if inaccurate | AB 489 (CA) license claims | -- | Required for clinical advice |
| 6. Safety check | -- | -- | -- | SB 243 (CA) suicidal ideation | -- | -- |
| 7. Clinician review | -- | -- | -- | -- | -- | **Critical control point** |
| 8. Response delivery | Transmission security | -- | -- | AI disclosure | -- | Clinician bears liability |
| 9. Escalation | EMTALA for emergencies | -- | -- | -- | EMTALA | Required for emergency |
| 10. EHR documentation | Medical records law | -- | -- | State records requirements | -- | -- |

### AI-Assisted Coding: Regulatory Exposure by Action Step

| Step | CMS | OIG/FCA | HIPAA | State Law | Physician Attestation |
|------|-----|---------|-------|-----------|----------------------|
| 1. Note created | CoP documentation | -- | -- | Medical records law | **Required** |
| 2. NLP extraction | -- | -- | Data access logging | -- | -- |
| 3. Code suggestion | Coding guidelines | Incorrect codes = FCA | -- | -- | -- |
| 4. Confidence scoring | Due diligence expectation | Reckless disregard standard | -- | -- | -- |
| 5. Coder review | Program Integrity Manual | Coder shares liability | -- | State coding requirements | Query compliance |
| 6. Finalization | Claim requirements | **Primary FCA exposure point** | Transaction standards | -- | -- |
| 7. Submission | Filing requirements | Submission = certification | HIPAA EDI | State prompt payment | -- |
| 8. Denial management | Appeal deadlines | Appealing known-bad claims | -- | -- | -- |

---

## 6. Behavioral AI / Precision Nudging (Lirio at BSMH)

### 6.1 What It Is

Lirio provides a "Precision Nudging" platform that combines behavioral science and AI to deliver hyper-personalized health behavior change messages. BSMH invested directly in Lirio, with BSMH CEO John Starcher joining Lirio's board. The partnership focuses on co-developing behavior change programs for chronic diseases including diabetes and hypertension.

### 6.2 Technical Architecture

**Core technology:** Lirio's "Large Behavior Model" (LBM) -- a foundation model trained on health-related behavioral patterns. This is distinct from Large Language Models in that it models human behavior rather than language.

**How the LBM works:**
1. **Data ingestion:** Patient demographic data, health journey context, prior engagement history, clinical data from EHR integration
2. **Behavioral barrier prediction:** The LBM predicts each person's specific barriers to action (e.g., why they haven't scheduled a screening, why they stopped taking medication)
3. **Content assembly:** Behavioral design creates "content elements infused with behavioral science." AI assembles these elements into "hundreds of unique message combinations" tailored to each individual
4. **Channel and timing optimization:** Reinforcement learning selects optimal delivery channel (email, SMS, MyChart, mail), timing, and frequency for each patient
5. **Continuous learning:** The model learns from patient interactions and behavioral responses, continuously refining predictions

**Reinforcement learning component:**
- Multi-agent system ensures engagement remains dynamic across different behaviors, organizations, and time
- Avoids static segmentation -- each patient's engagement strategy evolves
- "Behavioral reinforcement learning agent learns what behavioral science solutions will work for specific people"
- Results feed back into the LBM to improve predictions across the entire population

**Infrastructure:** Integrates through Microsoft Dynamics 365, Azure data warehouses, and EHR systems. Available on both AWS Marketplace and Microsoft Azure Marketplace.

### 6.3 Communication Channels

| Channel | Capability | Notes |
|---------|-----------|-------|
| Email | Primary channel | Personalized content, imagery, behavioral framing |
| SMS/Text | Direct messaging | Shorter nudges, action-oriented |
| MyChart | Patient portal integration | Integrates with existing patient engagement |
| Chatbot | Conversational nudging | Interactive behavioral interventions |

### 6.4 What a "Nudge" Looks Like Concretely

A Lirio nudge is NOT a generic "reminder to schedule your appointment." It is a behaviorally-targeted message that:

1. **Identifies the specific behavioral barrier** for that individual (e.g., fear of diagnosis, cost concerns, time constraints, mistrust of healthcare system)
2. **Applies a behavioral science technique** to address that barrier (e.g., loss aversion framing, social proof, implementation intentions, present bias correction)
3. **Personalizes content elements** -- messaging text, visual imagery, tone, call-to-action
4. **Optimizes delivery** -- when to send, which channel, what frequency

**Example domain:** COVID-19 vaccination -- Lirio sent 2.2 million messages encouraging vaccination, with messages leveraging incentives like "things that seemed impossibly far away" (e.g., returning to normal life).

**Example domain:** Workplace mental health -- pilot study showed 80% email open rate and 22% click-through to EAP site (vs. 5% benchmark annual EAP usage rate).

### 6.5 Opt-Out Mechanisms

Specific opt-out mechanisms for Lirio at BSMH are not documented in available public sources. Standard requirements would include:
- CAN-SPAM compliance for email (unsubscribe link)
- TCPA compliance for SMS (opt-out instructions)
- Patient communication preferences in EHR
- MyChart notification settings

**Governance gap:** It is unclear whether patients are informed that behavioral AI is being used to personalize their health communications, or whether they have granular control over the types of nudging they receive vs. a binary opt-in/opt-out.

### 6.6 Disparate Impact Risks

**Known AI healthcare bias patterns that apply to behavioral nudging:**

| Risk | Description | Mitigation Status |
|------|-------------|-------------------|
| **Data representation bias** | Training data primarily from urban, connected, insured populations. Rural, minority, and uninsured patients underrepresented. | Unknown for Lirio specifically |
| **Digital access bias** | Email/SMS/MyChart channels exclude patients without smartphones, internet, or digital literacy | Physical mail channel partially addresses |
| **Language bias** | English-language content may not reach non-English-speaking patients effectively | Unknown |
| **Behavioral model bias** | Behavioral science techniques validated primarily on WEIRD (Western, Educated, Industrialized, Rich, Democratic) populations | Lirio's BReLL lab may be researching this |
| **Reinforcement learning feedback loops** | If AI learns that certain demographics respond less, it may reduce engagement attempts -- creating a self-fulfilling prophecy of disengagement | Critical concern; unknown if addressed |
| **Cultural context** | Behavioral framing (loss aversion, social proof) operates differently across cultural contexts | Unknown |

**Historical precedent:** A healthcare algorithm used across American hospitals was "systematically biased against Black patients, affecting care decisions for approximately millions of Americans, denying necessary care to thousands of Black patients nationwide." The VBAC calculator "included race-based correction factors that systematically assigned lower success probabilities to African American and Hispanic women."

**Regulatory exposure:** Colorado's AI Act (effective June 30, 2026) will require deployers of "high-risk AI systems" to conduct impact assessments including disparate impact analysis. The FTC has signaled that algorithmic discrimination can constitute an unfair practice under Section 5.

---

## 7. Sources

### Catherine / Brado
- [Mercy Health Cincinnati First to Launch AI Powered Digital Assistant](https://www.mercy.com/news-events/news/cincinnati/2024/mercy-health-cincinnati-first-to-launch-ai-powered-digital-assistant)
- [Catching up with Catherine - Brado](https://brado.net/catching-up-with-catherine/)
- [Brado Healthcare CEP](https://brado.net/cep-healthcare/)
- [Brado and Accrete Expand Conversational AI Engagement Platform into Orthopedic Service Line](https://brado.net/brado-and-accrete-expand-conversational-ai-engagement-platform-into-orthopedic-service-line/)
- [Brado Conversational-AI Engagement](https://brado.net/conversational-ai-engagement/)
- [Bon Secours Mercy Health's next digital ventures move - Becker's](https://www.beckershospitalreview.com/healthcare-information-technology/innovation/bon-secours-mercy-healths-next-digital-ventures-move/)

### Epic MyChart ART
- [Epic and Microsoft Bring GPT-4 to EHRs](https://www.epic.com/epic/post/epic-and-microsoft-bring-gpt-4-to-ehrs/)
- [Epic MyChart In-Basket Automated Response Technology - Health AI DB](https://www.healthaidb.com/software/epic-mychart-in-basket-automated-response-technology/)
- [Gen AI Saves Nurses Time by Drafting Responses to Patient Messages - EpicShare](https://www.epicshare.org/share-and-learn/mayo-ai-message-responses)
- [AI for Clinicians - Epic](https://www.epic.com/software/ai-clinicians/)
- [Physicians using genAI to respond to MyChart messages - eMarketer](https://www.emarketer.com/content/physicians-using-genai-respond-their-patients--mychart-messages)
- [Systematic review of early evidence on generative AI for drafting responses to patient messages - NPJ Health Systems](https://www.nature.com/articles/s44401-025-00032-5)
- [Utilization of Generative AI-drafted Responses - NPJ Digital Medicine](https://www.nature.com/articles/s41746-025-01972-w)
- [AI-Generated Draft Replies Integrated Into Health Records - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019394/)
- [Study Reveals AI Enhances Physician-Patient Communication - UC San Diego](https://health.ucsd.edu/news/press-releases/2024-04-15-study-reveals-ai-enhances-physician-patient-communication/)

### ECRI / Failure Modes
- [Misuse of AI chatbots tops annual list of health technology hazards - ECRI](https://home.ecri.org/blogs/ecri-news/misuse-of-ai-chatbots-tops-annual-list-of-health-technology-hazards)
- [ECRI names misuse of AI chatbots as top health tech hazard for 2026 - MedTech Dive](https://www.medtechdive.com/news/ecri-health-tech-hazards-2026/810195/)
- [Examining the Greatest Health Technology Threats of 2026 - ECRI](https://home.ecri.org/blogs/ecri-news/examining-the-greatest-health-technology-threats-of-2026)

### FDA Regulation
- [FDA AI-Enabled Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices)
- [FDA Oversight: Understanding the Regulation of Health AI Tools - Bipartisan Policy Center](https://bipartisanpolicy.org/issue-brief/fda-oversight-understanding-the-regulation-of-health-ai-tools/)
- [The AI Chatbot is In - FDA Law Blog](https://www.thefdalawblog.com/2025/12/the-ai-chatbot-is-in/)
- [FDA questions on genAI-enabled chatbots - RAPS](https://www.raps.org/news-and-articles/news-articles/2025/11/fda-questions-on-genai-enabled-chatbots-raise-conc)
- [Changes to Existing Medical Software Policies - FDA](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/changes-existing-medical-software-policies-resulting-section-3060-21st-century-cures-act)
- [FDA Releases Significantly Revised Final Clinical Decision Support Software Guidance - Arnold & Porter](https://www.arnoldporter.com/en/perspectives/advisories/2026/01/fda-cuts-red-tape-on-clinical-decision-support-software)

### FTC / State Enforcement
- [FTC Announces Crackdown on Deceptive AI Claims and Schemes](https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes)
- [FTC's Foray Into Consumer-Facing AI Chatbots - ABA Health Law](https://www.americanbar.org/groups/health_law/news/2025/ftc-consumer-ai-chatbots-health-care/)
- [Texas AG Settlement with Pieces Technologies](https://www.texasattorneygeneral.gov/news/releases/attorney-general-ken-paxton-reaches-settlement-first-its-kind-healthcare-generative-ai-investigation)
- [Texas AG Reaches Novel Generative AI Settlement - Orrick](https://www.orrick.com/en/Insights/2024/09/Texas-Attorney-General-Reaches-Novel-Generative-AI-Settlement)
- [AI Chatbots at the Crossroads: Navigating New Laws - Cooley](https://www.cooley.com/news/insight/2025/2025-10-21-ai-chatbots-at-the-crossroads-navigating-new-laws-and-compliance-risks)
- [States stepping up on health AI regulation - AMA](https://www.ama-assn.org/practice-management/digital-health/states-are-stepping-health-ai-regulation)
- [California AB 489 - Duane Morris](https://www.duanemorris.com/alerts/california_passes_novel_law_governing_generative_ai_healthcare_1224.html)

### HIPAA
- [AI Chatbots and Challenges of HIPAA Compliance - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10937180/)
- [HIPAA Compliance AI in 2025 - Sprypt](https://www.sprypt.com/blog/hipaa-compliance-ai-in-2025-critical-security-requirements)
- [AI and HIPAA Compliance - TechTarget](https://www.techtarget.com/healthtechanalytics/feature/AI-and-HIPAA-compliance-How-to-navigate-major-risks)

### AI Medical Coding Systems
- [Solventum 360 Encompass Autonomous Coding System](https://www.solventum.com/en-us/home/health-information-technology/solutions/360-encompass-autonomous/)
- [SmarterDx](https://www.smarterdx.com/)
- [How SmarterDx Became a Generational Healthcare AI Company - Transformation Capital](https://transformcap.com/perspectives/how-smarterdx-became-a-generational-healthcare-ai-company)
- [Fathom - Medical coding automation powered by AI](https://www.fathomhealth.com/)
- [Iodine Software / Waystar CDI](https://iodinesoftware.com/solutions/clinical-documentation-integrity/)
- [Large Language Models Are Poor Medical Coders - NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2300040)

### Coding Arms Race
- [Policy brief: ambient AI scribes and the coding arms race - NPJ Digital Medicine](https://www.nature.com/articles/s41746-025-02272-z)
- [Ambient Artificial Intelligence Scribes and Physician Financial Productivity - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12789954/)
- [AI Scribes Boost Physician RVUs - Medscape](https://www.medscape.com/viewarticle/ai-scribes-boost-physician-rvus-could-also-raise-healthcare-2026a10001g4)
- [Cigna downcoding policy - AAFP](https://www.aafp.org/pubs/fpm/blogs/gettingpaid/entry/cigna-downcoding-em.html)
- [Insurance companies on a slippery slope: downcoding - STAT](https://www.statnews.com/2025/09/29/cigna-downcoding-prior-authorization-doctors-bureaucracy/)

### CMS/OIG Enforcement
- [Risk Adjustment Continues to be A Major Focus in Medicare Advantage - Morgan Lewis](https://www.morganlewis.com/pubs/2025/04/risk-adjustment-continues-to-be-a-major-focus-in-medicare-advantage)
- [OIG Semiannual Report 2025 Reveals Billing Red Flags - Bulwark Health](https://www.bulwarkhealth.info/oig-semiannual-report-2025-reveals-billing-red-flags/)
- [Healthcare Fraud Enforcement Trends 2026 - AGG](https://www.agg.com/news-insights/publications/healthcare-fraud-enforcement-trends-to-expect-in-2026/)
- [Medical Coding Compliance in 2025 - Nym Health](https://blog.nym.health/medical-coding-compliance)

### Stanson Health / HCC
- [BSMH Documents Over 35K HCC Categories - Premier/PINC AI](https://www.pinc-ai.com/stanson-health/bon-secours-mercy-health-documents-over-35000-hcc-categories-in-just-six-months-leading-to-a-significant-financial-impact)
- [Stanson Health HCC Solutions](https://stansonhealth.com/hcc)
- [Premier's Stanson Health - Smart Clinical Decision Support](https://premierinc.com/stanson-health)

### CMS WISeR
- [CMS WISeR Model Details - ASRA](https://asra.com/news-publications/asra-update-item/asra-updates/2025/10/24/cms-provides-more-details-on-wiser-prior-authorization-model)
- [Federal Register: WISeR Model Implementation](https://www.federalregister.gov/documents/2025/07/01/2025-12195/medicare-program-implementation-of-prior-authorization-for-select-services-for-the-wasteful-and)
- [WISeR Model Using AI in New Era of Medicare Prior Auths - Ensemble](https://www.ensemblehp.com/blog/the-wiser-model/)
- [Meet the 6 vendors participating in CMS WISeR Model - TechTarget](https://www.techtarget.com/revcyclemanagement/feature/Meet-the-6-vendors-participating-in-the-CMS-WISeR-Model)

### Lirio / Behavioral AI
- [BSMH Invests in Lirio - Healthcare IT News](https://www.healthcareitnews.com/news/bon-secours-mercy-health-invests-lirio-artificial-intelligence-platform)
- [BSMH and Lirio Partnership Announcement](https://bsmhealth.org/bon-secours-mercy-health-and-lirio-announce-partnership-and-investment/)
- [Lirio Launches Precision Nudging for Diabetes Care](https://www.prnewswire.com/news-releases/lirio-launches-precision-nudging-solution-to-close-gaps-in-diabetes-care-301314106.html)
- [Power of Large Behavior Models in Healthcare Consumer Engagement - Lirio](https://lirio.com/blog/the-power-of-large-behavior-models-in-healthcare-consumer-engagement/)
- [Lirio Approach](https://lirio.com/approach/)
- [Community Health Network and Lirio with Microsoft Technology](https://www.prnewswire.com/news-releases/community-health-network-and-lirio-utilize-microsoft-technology-to-put-patients-first-with-hyper-personalized-patient-journeys-302275644.html)

### Liability / Physician Supervision
- [When Does Physician Use of AI Increase Liability? - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8679587/)
- [AI and Professional Liability in Healthcare - Frontiers](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1337335/full)
- [Malpractice concerns impact physician decisions to consult AI - Johns Hopkins](https://carey.jhu.edu/articles/research/malpractice-concerns-physician-consult-ai)
- [The New Malpractice Frontier - Medical Economics](https://www.medicaleconomics.com/view/the-new-malpractice-frontier-who-s-liable-when-ai-gets-it-wrong-)

### AI Bias / Disparate Impact
- [Bias in Medical AI: Implications for Clinical Decision-Making - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11542778/)
- [Bias Recognition and Mitigation Strategies - Nature Digital Medicine](https://www.nature.com/articles/s41746-025-01503-7)
- [Evolving Perspectives on Healthcare Algorithmic and AI Bias - HHS OMH](https://minorityhealth.hhs.gov/news/evolving-perspectives-healthcare-algorithmic-and-artificial-intelligence-bias)

### Denial Management
- [Battle of the bots: Payers use AI to drive denials, providers fight back - HFMA](https://www.hfma.org/revenue-cycle/denials-management/health-systems-start-to-fight-back-against-ai-powered-robots-driving-denial-rates-higher/)
- [How AI and Automation Support Denial Management - BDO](https://www.bdo.com/insights/industries/healthcare/how-ai-and-automation-can-support-the-denial-management-process)

---

*End of document. This research is current as of 2026-02-16. The regulatory landscape for healthcare AI is evolving rapidly; all enforcement positions and regulatory frameworks described here should be verified against current federal and state guidance before reliance.*
