# Epic AI Ecosystem: Technical Architecture, APIs, and Integration Points

**Researcher:** Nell (Research Methodologist), TELOS Research Team
**Date:** 2026-02-16
**Purpose:** Technical architecture reference for TELOS deployment at Epic-based health systems (Mercy Health, Wexner Medical Center)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Table of Contents

1. [Epic's Own AI Tools](#1-epics-own-ai-tools)
2. [Epic's Integration APIs for Third-Party AI](#2-epics-integration-apis-for-third-party-ai)
3. [Epic Workflow Integration Points](#3-epic-workflow-integration-points)
4. [What Clinicians Actually See](#4-what-clinicians-actually-see)
5. [Epic's Governance and Audit Capabilities](#5-epics-governance-and-audit-capabilities)
6. [Other EHR AI Platforms](#6-other-ehr-ai-platforms)
7. [Implications for TELOS](#7-implications-for-telos)

---

## 1. Epic's Own AI Tools

### 1.1 CoMET (Cosmos Medical Event Transformer) / Curiosity

**What it is:** CoMET is a family of decoder-only transformer models trained on Epic Cosmos data. It is the foundational model behind Epic's "Curiosity" clinical intelligence product. The brand name "Curiosity" is the clinical-facing product; "CoMET" is the underlying model architecture.

**Architecture:**
- **Model type:** Decoder-only transformer (Qwen2 architecture with random initialization -- no pretrained weights loaded)
- **Training data:** 118 million patients, 115 billion discrete medical events (151 billion tokens), from de-identified longitudinal records spanning 16.3 billion encounters across 300+ million unique patient records from 310 health systems
- **Model sizes:** Three model sizes trained, with optimal compute and training tokens determined by scaling-law analysis; up to 1 billion parameters
- **Training methodology:** Autoregressive next-event prediction on sequences of time-ordered medical events (diagnoses, labs, medications, encounters)
- **Published paper:** [arXiv:2508.12104 -- "Generative Medical Event Models Improve with Scale"](https://arxiv.org/abs/2508.12104)

**How it works:**
1. A patient's medical history is formulated as a sequence of discrete medical events
2. CoMET is prompted with this history and autoregressively generates the next events
3. Multiple plausible future timelines are simulated, reflecting real-world complexity (diagnoses resolving/emerging, complications, care needs shifting)
4. These simulations are summarized into clinical insights and presented to clinicians in workflows

**Clinical capabilities (78 validated tasks):**
- Diagnosis prediction
- Disease prognosis
- 30-day readmission risk
- Extended hospitalization prediction
- ASCVD risk assessment
- Emergence of specific conditions (e.g., pancreatic cancer)
- Healthcare operational predictions

**Performance:** Generally outperforms or matches task-specific supervised models on 78 real-world tasks *without* requiring task-specific fine-tuning or few-shot examples.

**Access timeline:**
- February 2026: Cosmos AI Lab opens to researchers from participating Cosmos organizations for testing new use cases
- Built entirely within Epic Cosmos; operates under Cosmos privacy, security, and compliance standards

**Sources:**
- [arXiv paper](https://arxiv.org/html/2508.12104v1)
- [Epic Curiosity announcement](https://www.epic.com/epic/post/curiosity-a-new-medical-intelligence-for-clinical-and-operational-insights/)
- [STAT News coverage](https://www.statnews.com/2025/08/27/epics-doctor-strange-moment-ai-for-possible-patient-futures-ai-prognosis/)
- [HIT Consultant](https://hitconsultant.net/2025/09/03/epic-launches-comet-a-new-ai-platform-to-predict-patient-health-journeys/)

---

### 1.2 Epic Cosmos

**What it is:** A de-identified, aggregate clinical data platform that serves as both a research database and the training substrate for Epic's AI models.

**Data scope (as of August 2025):**
- 300+ million unique patient records
- 16.3 billion encounters
- 310+ participating health systems
- 4 countries
- De-duplicated longitudinal records (each patient's records combined across systems into a single integrated record)

**Data transmission architecture:**
- Data transmitted via encrypted HL7 Consolidated-Clinical Document Architecture (C-CDA) documents
- Uses Epic's existing Care Everywhere clinical health information exchange network
- Submissions categorized into: historical backload and event-driven triggered data

**Governance model:**
- **Governing Council:** 15 elected representatives from pediatric, academic, and community healthcare organizations
- **Rules of the Road:** Guidelines co-developed by Epic and the Governing Council governing data usage
  - Selling data is prohibited
  - Using data for advertising or market comparison is prohibited
  - Attempts to re-identify patients are strictly prohibited
  - All queries are recorded; users attest to their work every time they access data
  - The Cosmos team reviews all attested uses and follows up as needed
- **New data types:** Undergo quarterly review process evaluating re-identification risk before being added

**How health systems contribute:**
- Opt-in service for Epic EHR customers
- Organizations must agree to Cosmos guidelines
- Data contributions grant access to Cosmos at no additional cost
- Patient identifiers are removed before data leaves the contributing organization

**Sources:**
- [Epic Cosmos About](https://cosmos.epic.com/about/)
- [Cosmos Governing Council](https://cosmos.epic.com/board/governance/)
- [PMC: The Cosmos Collaborative](https://pmc.ncbi.nlm.nih.gov/articles/PMC8775787/)
- [JSCDM: Cosmos Real-World Data](https://www.jscdm.org/article/id/246/)

---

### 1.3 Epic Sepsis Model (ESM)

**What it is:** A predictive analytics model for early sepsis detection, built into Epic's EHR and deployed at hundreds of hospitals.

**Technical architecture:**
- **Model type:** Penalized logistic regression
- **Training data:** 500,000 patient encounters
- **Input variables:** ~80 demographic and clinical variables including:
  - Vital signs
  - Laboratory results
  - Comorbidities
  - Demographic factors
- **Scoring frequency:** Every 15 minutes throughout hospital admission
- **Score range:** 0-10+ (higher = greater sepsis risk)
- **Alert threshold:** Score >= 5 (determined via ROC analysis, AUC 0.834)

**Integration with Epic workflow:**
1. Model runs as a batch job, calculating scores from the Chronicles database
2. Data is sent to cloud for scoring and returned to the system
3. When threshold is met, an interruptive Best Practice Advisory (BPA) appears to physicians and nurses
4. The ESM score is treated as a "critical value" -- nurses call physicians and provide pertinent clinical information
5. BPA configured to alert clinicians about high-risk patients

**Performance (validated):**
- Sensitivity: 86.0%
- Specificity: 80.8%
- Positive predictive value: 33.8%
- Negative predictive value: 98.11%

**Known limitations:** The model has been criticized in published literature (JAMA Internal Medicine, 2021) for "poor performance" in external validation, though debate continues about appropriate evaluation metrics for screening tools.

**Sources:**
- [PMC: ESM Validation Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317482/)
- [JAMA Internal Medicine: External Validation](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307)
- [PMC: AI-Driven CDS for Sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10498958/)

---

### 1.4 Hey Epic! (Voice Assistant)

**What it is:** A voice-activated assistant within Epic Hyperspace, similar to consumer smart speakers, that allows clinicians to interact with the EHR through voice commands.

**Capabilities:**
- Place orders (medications, labs, imaging)
- Call members of a patient's care team
- Create reminders and tasks
- Navigate the EHR
- Query patient information

**Technical integration:**
- Powered by Nuance Dragon Medical Virtual Assistant technology
- Integrated into Epic Hyperspace
- Uses conversational AI for natural language understanding
- Currently used by ~20 organizations

**Sources:**
- [Healthcare IT News: Epic voice assistant](https://www.healthcareitnews.com/news/epic-debut-ambient-voice-technology-assistant-himss20)
- [Nuance/Epic integration announcement](https://www.prnewswire.com/news-releases/nuance-advances-conversational-ai-with-dragon-medical-virtual-assistant-for-hey-epic-virtual-assistant-in-epic-hyperspace-301115268.html)

---

### 1.5 Epic's Pre-Built Predictive Models

Epic offers several built-in predictive models beyond CoMET/Curiosity:

| Model | Type | Purpose | Data Source |
|-------|------|---------|-------------|
| Risk of Unplanned Readmission | Logistic regression | 30-day readmission risk | Trained on 275,000+ encounters from 26 hospitals |
| Epic Sepsis Model (ESM) | Penalized logistic regression | Early sepsis detection | 500,000 encounters, 80 variables |
| Deterioration Index | Predictive model | Patient deterioration risk | Vital signs, labs, nursing assessments |
| No-Show Prediction | Machine learning | Appointment no-show risk | Scheduling history, demographics |
| Fall Risk | Predictive model | Inpatient fall risk | Nursing assessments, patient factors |

**Cosmos-based next-generation models** (coming 2026+) will replace many task-specific models with Curiosity's simulation-based approach, which has shown comparable or superior performance without task-specific fine-tuning.

**Sources:**
- [EpicShare: Readmission Score](https://www.epicshare.org/tips-and-tricks/use-predictive-risk-score-to-reduce-readmission)
- [Healthcare IT News: Predictive models reduce readmissions](https://www.healthcareitnews.com/news/epic-integrated-predictive-models-reduce-readmissions-save-7m)

---

### 1.6 Emmie (Patient-Facing AI Assistant)

**What it is:** A 24/7 patient-facing AI chatbot integrated into the MyChart patient portal.

**Capabilities:**
- Explains lab results in plain language
- Recommends preventive screenings
- Helps with appointment scheduling
- Answers patient questions about test results
- Assists with billing: understanding bills, making payments, setting up payment plans, generating reimbursement statements
- Engages patients before appointments for pre-visit preparation
- Open-ended conversational AI
- Works with Art autonomously to streamline patient outreach tasks on behalf of providers

**Access channels:**
- MyChart web portal
- Text message (expanding)
- Conversational interface

**Adoption:** 85% of Epic's customers are live with generative AI features across Art, Emmie, and Penny.

**Sources:**
- [Epic AI for Patients](https://www.epic.com/software/ai-patients/)
- [Savva Blog: Emmie and Art](https://savva.ai/blogs/emmie-and-art-epics-ai-reshaping-healthcare)
- [EMRFinder: Emmie](https://emrfinder.com/blog/emmie-new-ai-tool-by-epic-emr-software/)

---

### 1.7 Art (AI Charting / Clinical AI Suite)

**What it is:** Epic's clinician-facing AI tool suite, encompassing ambient documentation (AI Charting), clinical insights, and In-Basket message drafting.

#### Art -- AI Charting (Ambient Documentation)

**How it works:**
1. AI Charting ambiently listens during patient visits (in-person or telehealth)
2. The system has full context of the patient's chart (medications, problem lists, prior history)
3. A draft clinical note is generated, oriented to the specific patient's context
4. Orders discussed during the visit are identified and placed in a "shopping cart" / "review cart"
5. The clinician reviews, edits, and signs off on both notes and orders

**Key features:**
- **Voice agent:** Clinicians can personalize note format through natural conversation (e.g., "format the HPI as a bulleted list")
- **Ambient Ordering:** Identifies medications, lab tests, and procedures discussed and queues them for review
- **Shopping Cart review:** Compiles notes and orders from a visit for end-of-session verification and sign-off
- **Diagnosis-aware notes:** (Coming March 2026) Links assessments and plans to specific diagnoses on the problem list

**Technical infrastructure:**
- Built on Microsoft Azure platform for HIPAA-compliant pipeline
- Uses GPT-4 via Azure OpenAI Service
- Authentication via OpenID Connect (OIDC) and OAuth 2.0
- SMART on FHIR protocols for secure data exchange between EHR and AI processing engines
- AI accesses only minimal necessary clinical context (data minimization principle)

#### Art -- Insights

**What it is:** Brings together information from across the patient chart into a concise summary for visit preparation.
- Used over 16 million times per month
- Summarizes recent chart entries, external data, and notes

#### Art -- In-Basket ART (Augmented Response Technology)

**What it is:** Generative AI that drafts responses to patient MyChart messages.
- Analyzes patient message content and clinical context
- Produces draft responses for clinician review and editing
- Clinician reviews, edits, and sends (maintaining authorship)
- 150+ healthcare systems using the technology
- Generates 1 million+ drafts per month
- Saves nurses ~30 seconds per message (Mayo Clinic data)
- Noted for generating more empathetic responses

**Sources:**
- [Epic AI Charting announcement](https://www.epic.com/epic/post/epic-ai-charting-rolls-out-alongside-an-expanding-set-of-built-in-ai-capabilities/)
- [Healthcare IT Today: AI Charting](https://www.healthcareittoday.com/2026/02/05/epic-ambient-ai-charting-released-and-more-updates-on-epics-ai-solutions/)
- [Fierce Healthcare: AI scribe](https://www.fiercehealthcare.com/ai-and-machine-learning/epic-rolls-out-ai-charting-and-more-built-automation-clinicians-and)
- [Epic AI for Clinicians](https://www.epic.com/software/ai-clinicians/)

---

### 1.8 Penny (Revenue Cycle AI)

**What it is:** AI assistant for revenue cycle management (RCM).

**Current capabilities (live):**
- Drafting appeals for denied claims
- Suggesting billing codes based on documentation
- Virtual assistant for billing code questions

**Planned capabilities (November 2026):**
- Autonomous coding (starting with ED and Radiology encounters)
- Outpatient denial appeal support
- Automated claims follow-up with payers

**Sources:**
- [CNBC: Epic UGM 2025](https://www.cnbc.com/2025/08/20/epic-ugm-2025-epic-touts-new-ai-tools.html)
- [Thoughtful.ai: Epic's Generative AI Strategy](https://www.thoughtful.ai/blog/epic-goes-all-in-on-ai)

---

### 1.9 AI Agents (Coming 2026)

Epic is building an agentic AI platform with over 160 AI projects. Planned agents include:

| Agent | Target Date | Function |
|-------|------------|----------|
| Pre-visit planning agent | 2026 | Chats with patients, identifies missing tasks (labs), helps schedule, creates visit summaries |
| Discharge planning agent | 2026 | Summarizes stay events, speeds discharge planning |
| Patient flow agent | 2026 | Manages and optimizes patient flow |
| Pre-surgical risk calculator | 2026 | Calculates surgical risk |
| MyChart virtual assistant | 2026 | Enhanced patient portal assistant |
| Patient-facing imaging results | 2026 | Explains imaging results to patients |
| Patient-reported outcomes insights | 2026 | Analyzes PRO data |

**Sources:**
- [Healthcare IT News: AI agents](https://www.healthcareitnews.com/news/epic-unveils-ai-agents-showcases-new-foundational-models)
- [Advisory.com: Epic debuts AI tools](https://www.advisory.com/daily-briefing/2025/08/28/epic-meeting)

---

## 2. Epic's Integration APIs for Third-Party AI

### 2.1 FHIR R4 APIs

**Specification:** Epic supports FHIR DSTU2, STU3, and R4. R4 is the primary version for US regulatory compliance.

**Key FHIR R4 resources available for third-party AI:**

| Category | Resources | Operations |
|----------|-----------|------------|
| **Patient/Demographics** | Patient, RelatedPerson, Practitioner, PractitionerRole, Organization | Read, Search |
| **Clinical** | Condition, Observation, DiagnosticReport, Procedure, AllergyIntolerance, Immunization | Read, Search, Create (some) |
| **Medications** | Medication, MedicationRequest, MedicationStatement, MedicationAdministration | Read, Search, Create |
| **Documents** | DocumentReference, Binary | Read, Search, Create |
| **Encounters** | Encounter, EpisodeOfCare | Read, Search |
| **Scheduling** | Appointment, Schedule, Slot | Read, Search, Create |
| **Orders** | ServiceRequest, NutritionOrder | Read, Search |
| **Care Plans** | CarePlan, CareTeam, Goal | Read, Search |
| **Imaging** | ImagingStudy, Observation (DICOM Image Characteristics) | Read, Search, Create (May 2025+) |
| **Clinical Notes** | DocumentReference (Clinical Notes) | Read, Search, Create |
| **Flags** | Flag (Patient FYI flags) | Read, Search |

**Authentication:** OAuth 2.0 with SMART App Launch Framework

**2025 updates:**
- International Patient Summary (IPS) FHIR specification support (May 2025)
- New DICOM imaging metadata resource (May 2025)
- Post-filtering search mechanism (May 2024+)

**Developer portal:** [fhir.epic.com](https://fhir.epic.com/) and [open.epic.com](https://open.epic.com/)

**Sources:**
- [Epic on FHIR Documentation](https://fhir.epic.com/Documentation)
- [Open Epic Technical Specifications](https://open.epic.com/TechnicalSpecifications)

---

### 2.2 Epic Showroom / App Market (formerly App Orchard)

**Evolution:**
- 2016: App Orchard launched
- 2021: Rebranded to App Market
- 2022: App Market shut down; replaced by Showroom ecosystem

**Current vendor programs:**

| Program | Description | Examples |
|---------|-------------|----------|
| **Cornerstone Partners** | Technology Epic uses significantly in its software | Microsoft, InterSystems |
| **Workshop** (formerly Partners/Pals) | Third-party vendors co-developing technology with Epic beyond standards | Abridge, Nuance, Ambience |
| **Toolbox** | Specific integration categories with Epic blueprints | Ambient Voice Recognition vendors |
| **Connection Hub** | Online directory where any vendor with a live Epic connection can list their product | Open to any integrated vendor |
| **Showroom** | Public-facing marketplace/directory | [showroom.epic.com](https://showroom.epic.com/) |

**Sources:**
- [Fierce Healthcare: Showroom](https://www.fiercehealthcare.com/health-tech/epic-unveils-new-app-showroom-third-party-vendors)
- [Lifebit: Epic App Store Integration Guide](https://lifebit.ai/blog/epic-app-store-integration/)

---

### 2.3 Epic "Pal" Framework / Workshop Program

**What it is:** The Pals program (now evolved into Workshop) provides a pathway for early-stage AI companies to integrate deeply with Epic's clinical workflows.

**Abridge as first Pal (case study):**
- Abridge was the first "Pal" in Epic's Partners and Pals program (August 2023)
- Integration delivers real-time, structured summaries of clinician-patient conversations
- Complete auditability of AI-generated content
- All AI output is draft-only, requiring clinician review and attestation
- Integration supports comprehensive auditing: who recorded audio, which model generated which draft, when edits were made

**Procurement benefits:**
- Standardized BAA, HIPAA risk assessment, and auditability
- Standardized launch points (not bespoke UI widgets)
- Epic-vetted integration route

**Sources:**
- [Abridge: First Pal announcement](https://www.abridge.com/press-release/abridge-becomes-epics-first-pal-bringing-generative-ai-to-more-providers-and-patients)
- [Healthcare Dive: Epic/Abridge](https://www.healthcaredive.com/news/Epic-Abridge-Partners-Pals-programs/691138/)

---

### 2.4 Epic Toolbox (Ambient Voice Recognition)

**What it is:** A specific integration category within Epic's Showroom ecosystem with standardized blueprints for ambient voice recognition tools.

**How it works:**
- Vendors achieve Toolbox designation for specific categories (e.g., "Ambient Voice Recognition")
- Integration follows Epic's Toolbox Blueprint specifications
- Tools are made available natively inside Haiku (Epic's mobile app)
- Clinicians can launch from Haiku across outpatient, ED, and inpatient settings

**Current Toolbox AI vendors:**
- Ambience Healthcare (August 2025)
- Commure Ambient Suite
- Additional vendors: NESA, Imprivata, Five9

**Ambience integration details:**
- Available natively inside Haiku
- Goes beyond ambient listening -- incorporates prior notes, diagnostics into clinical notes
- Offers downstream revenue cycle automation for coding, CDI, and prior authorization

**Sources:**
- [Fierce Healthcare: Ambience joins Toolbox](https://www.fiercehealthcare.com/ai-and-machine-learning/epic-adds-ambience-healthcares-ai-its-toolbox-program)
- [Ambience: Toolbox announcement](https://www.ambiencehealthcare.com/blog/ambience-healthcare-joins-epic-toolbox-unlocking-advanced-ai-functionality-within-haiku-for-epic-customers)

---

### 2.5 CDS Hooks

**What it is:** A RESTful API standard for invoking clinical decision support from within a clinician's EHR workflow. Epic natively supports CDS Hooks.

**Supported hooks in Epic:**

| Hook | Trigger Point | Use Case |
|------|--------------|----------|
| `patient-view` | Clinician opens a patient's record | Display risk scores, care gaps, relevant alerts |
| `order-select` | Clinician selects orders to place | Drug interaction checks, formulary alternatives, AI-recommended orders |
| `order-sign` | Immediately before order is signed | Final validation, appropriateness checks, cost alerts |

**How CDS Hooks work technically:**
1. EHR fires a hook at a specific workflow point
2. Hook sends context (patient ID, user, encounter, relevant FHIR resources) to CDS Service endpoint
3. CDS Service returns "cards" with information, suggestions, or links to launch SMART apps
4. Cards can include: information text, suggested actions, links to SMART on FHIR apps

**Integration for AI systems:**
- Third-party AI can register as a CDS Service
- Receives real-time clinical context at decision points
- Returns AI-generated recommendations as CDS cards
- Can trigger SMART on FHIR app launch for more complex interactions

**Sources:**
- [CDS Hooks specification](https://cds-hooks.org/)
- [Epic CDS Hooks documentation](https://fhir.epic.com/Documentation?docId=cds-hooks)
- [Medblocks: CDS Hooks guide](https://medblocks.com/blog/hl7-cds-hooks-a-practical-guide)

---

### 2.6 Best Practice Alert (BPA) Framework

**What it is:** Epic's mechanism for surfacing clinical decision support alerts (including AI-triggered alerts) directly to clinicians within their workflow.

**Technical architecture:**
- BPAs are configured in Epic's build environment
- Can be triggered by: predictive model scores, clinical rules, CDS Hooks responses, or external system signals
- Support both interruptive (must acknowledge) and passive (informational) modes
- Can be embedded within navigators for 2-way communication

**AI-triggered BPA workflow:**
1. Predictive model (e.g., ESM) calculates a risk score
2. Score compared against configured threshold
3. If threshold met, BPA fires in the clinician's workflow
4. BPA can appear as:
   - Interruptive alert requiring acknowledgment
   - Passive notification in a navigator sidebar
   - Dashboard entry for care team monitoring
5. Clinician can accept, override (with reason), or defer

**Key design considerations:**
- Alert fatigue management requires careful threshold tuning
- Success often requires multidisciplinary process redesign
- BPAs should include clear rationale and suggested actions

**Sources:**
- [PMC: AI-Driven CDS for Sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10498958/)
- [University of Iowa: BPA resources](https://epicsupport.sites.uiowa.edu/epic-resources/best-practice-advisories-bpas)

---

### 2.7 In-Basket Integration

**What it is:** The mechanism by which AI-drafted messages and notifications appear in clinicians' Epic In-Basket.

**In-Basket ART (Augmented Response Technology):**
- AI analyzes incoming patient MyChart messages
- Considers patient's clinical context from the chart
- Generates draft responses
- Drafts appear in In-Basket with clear "AI-drafted" status
- Clinician reviews, edits, and sends
- All edits are tracked for monitoring AI quality

**Technical pipeline:**
- Uses GPT-4 via Microsoft Azure OpenAI Service
- HIPAA-compliant pipeline
- Patient message content + chart context -> LLM -> Draft response
- Draft is never auto-sent; always requires clinician action

**Sources:**
- [Epic AI for Clinicians](https://www.epic.com/software/ai-clinicians/)
- [Nature: Systematic review of GenAI for patient messages](https://www.nature.com/articles/s44401-025-00032-5)

---

### 2.8 SMART on FHIR

**What it is:** An open standard for launching third-party applications within the EHR, with clinical context automatically passed to the app.

**Launch context in Epic:**
- Apps can be launched from within EHR with current patient, encounter, or other clinical context
- Launch context information is passed via OAuth 2.0 authorization request
- Apps appear as:
  - Main activity in a patient's chart
  - Sidebar panel
  - Custom menu button launch
  - Embedded within Epic Hyperspace

**For AI applications:**
- AI agents can use SMART on FHIR to securely fetch or update patient data
- Launch context provides the patient/encounter in scope
- OAuth 2.0 ensures secure, scoped access
- Tight workflow integration minimizes disruption to provider workflow

**Authentication flow:**
1. App registered in Epic (App Orchard or direct)
2. User clicks launch point in Epic
3. Epic redirects to app with authorization code
4. App exchanges code for access token (OAuth 2.0)
5. App uses token to make FHIR API calls in patient context

**Sources:**
- [SMART Health IT docs](https://docs.smarthealthit.org/)
- [Epic on FHIR Documentation](https://fhir.epic.com/Documentation)
- [Techno-soft: Integrating SMART App with Epic](https://techno-soft.com/integrating-smart-app-with-epic.html)

---

### 2.9 Interconnect / Open Epic

**What it is:** Epic's service-oriented architecture (SOA) platform for API access beyond FHIR.

**Technical architecture:**
- Supports SOAP/XML and REST/XML/JSON over HTTP/S
- Provides both synchronous request/response and asynchronous queued messaging
- Handles workflows not yet standardized in FHIR (billing, scheduling, in-basket messages)

**Available integration standards:**
- HL7 v2 and v3
- NCPDP (pharmacy)
- CDA (Clinical Document Architecture)
- ANSI
- DICOM (imaging)
- FHIR R4
- Custom web services

**Data layer architecture:**

| Layer | Technology | Purpose | Access |
|-------|-----------|---------|--------|
| **Chronicles** | InterSystems Cache/IRIS (proprietary non-relational) | Live operational data | Internal only |
| **Clarity** | SQL relational database | ETL from Chronicles; detailed analytical reports | SQL queries |
| **Caboodle** | Enterprise Data Warehouse (EDW) | BI-optimized view of Clarity data | Reporting, analytics |
| **Cosmos** | De-identified aggregate | Cross-system research and AI training | Research portal |
| **Cogito Cloud (Nebula)** | Azure-native lakehouse (Microsoft Fabric) | Next-gen analytics platform | Transitioning from Clarity/Caboodle |

**Migration note:** Epic is transitioning Clarity and Caboodle into medallion layers within an Azure-native lakehouse architecture using Microsoft Fabric. This will use Delta format (open-source, industry-standard) enabling new data sharing and portability use cases.

**Sources:**
- [Open Epic](https://open.epic.com/)
- [DreamFactory: Epic data access guide](https://blog.dreamfactory.com/how-to-securely-access-and-unlock-epic-epiccare-data-2025-guide-to-integration-options-for-apps-ai-and-analytics/)
- [Hakkoda: Epic Cogito to Azure](https://hakkoda.io/resources/moving-to-azure/)

---

## 3. Epic Workflow Integration Points

### 3.1 Ambulatory (Outpatient Clinic) Workflow

| Workflow Stage | AI Integration Point | Epic Module | AI Tool |
|----------------|---------------------|-------------|---------|
| **Pre-visit** | Patient preparation, care gap identification, pre-visit summary | MyChart, Scheduling | Emmie (patient), Art Insights (clinician), Pre-visit planning agent (2026) |
| **Scheduling** | No-show prediction, optimal scheduling | Cadence | Predictive models |
| **Check-in** | Patient engagement, form completion | Welcome/Kiosk | Emmie |
| **Rooming** | Vitals integration, nursing assessment | EpicCare Ambulatory | Nursing AI workflows |
| **Encounter** | Ambient documentation, note drafting, order suggestions | EpicCare Ambulatory, Haiku | Art AI Charting, Ambience, Nuance DAX |
| **Orders** | CDS Hooks, formulary checks, AI recommendations | EpicCare Ambulatory | CDS Services via order-select/order-sign hooks |
| **Checkout** | Follow-up scheduling, patient instructions | EpicCare Ambulatory | Art, automated summaries |
| **Post-visit** | Patient messaging, In-Basket response drafting | MyChart, In-Basket | Art ART, Emmie |

### 3.2 Inpatient Workflow

| Workflow Stage | AI Integration Point | Epic Module | AI Tool |
|----------------|---------------------|-------------|---------|
| **Admission** | Risk scoring (readmission, deterioration, falls) | EpicCare Inpatient | Predictive models, BPAs |
| **Rounding** | Patient summary, clinical insights | EpicCare Inpatient, Haiku | Art Insights |
| **Monitoring** | Sepsis prediction, deterioration alerts | EpicCare Inpatient | ESM (every 15 min), Deterioration Index |
| **Nursing assessments** | Ambient documentation for bedside nursing | EpicCare Inpatient | AI Charting for nursing (March 2026) |
| **Orders** | CDS Hooks, drug interaction checks | CPOE | CDS Services |
| **Documentation** | Ambient note generation, structured data | EpicCare Inpatient | Art AI Charting |
| **Discharge** | Discharge summary generation, planning | EpicCare Inpatient | AI discharge summaries, Discharge planning agent (2026) |

### 3.3 Emergency Department Workflow

| Workflow Stage | AI Integration Point | Epic Module | AI Tool |
|----------------|---------------------|-------------|---------|
| **Triage** | Acuity scoring, predictive risk | ASAP (ED module) | Predictive models |
| **Tracking** | Patient flow optimization | ASAP | Patient flow tracking |
| **Assessment** | Ambient documentation | ASAP, Haiku | Art AI Charting, Ambience (via Toolbox) |
| **Orders** | CDS Hooks, sepsis screening | ASAP | CDS Services, ESM |
| **Disposition** | Readmission risk, discharge planning | ASAP | Predictive models |
| **Coding** | Autonomous coding | Revenue Cycle | Penny (November 2026) |

### 3.4 Radiology (Epic Radiant)

| Integration Point | Description | AI Role |
|-------------------|-------------|---------|
| **Worklist prioritization** | AI modules assign priority scores to exams | Suspected stroke -> expedite; AI triage |
| **Image viewing** | PACS integration with Epic Radiant | AI-flagged studies highlighted |
| **Results integration** | FHIR Observation (DICOM Image Characteristics) | AI-generated findings filed via FHIR (May 2025+) |
| **Report generation** | Draft report text from structured findings | AI-generated draft reports for radiologist review |
| **Patient-facing results** | Imaging results explanation | Patient-facing imaging results overview (2026) |

### 3.5 Lab/Pathology (Epic Beaker)

| Integration Point | Description | AI Role |
|-------------------|-------------|---------|
| **Order management** | Outgoing Lab Instrument Orders interface | AI-driven test ordering recommendations |
| **Results integration** | Incoming Lab Instrument Results interface | AI-augmented results interpretation |
| **Digital pathology** | Image management system integration | AI-based slide analysis, case prioritization |
| **Workflow automation** | Case-level and slide-level integration | AI-driven case ordering and workflow optimization |
| **Reporting** | Native data transfer from LIS to Epic EMR | AI-assisted pathology reporting |

**Key architectural note for Beaker AI integration:** Deep integration between Beaker and digital pathology systems requires mapping clinical realities to technical architecture. Shallow integration creates fragmented workflows and manual reconciliation. The integration uses the Outgoing Lab Instrument Orders and Incoming Lab Instrument Results interfaces.

**Sources:**
- [Epic Radiant overview](https://www.healthcareitleaders.com/blog/what-is-epic-radiant/)
- [Pathology News: Beaker integration for digital pathology](https://www.pathologynews.com/industry-news/maximizing-your-epic-investment-why-deep-epic-beaker-integration-matters-for-digital-pathology/)
- [SPSoft: Epic EHR AI Trends](https://spsoft.com/tech-insights/epic-ehr-ai-trends-in-2025-reshaping-care/)

---

## 4. What Clinicians Actually See

### 4.1 AI-Triggered Best Practice Alert (BPA)

**Visual presentation:**
- Appears as a pop-up alert within the patient's chart or during an order workflow
- Contains: alert title, risk score, relevant clinical data that triggered the alert, and suggested actions
- Clinician response options: Accept recommendation, Override with reason, Defer/Snooze
- Can also appear within a custom-built "navigator" sidebar for ongoing monitoring
- Navigator supports 2-way communication (provider can respond to BPA)

**Example (Sepsis BPA):**
- Title: "Sepsis Risk Alert"
- Score displayed (e.g., ESM Score: 7/10)
- Key contributing factors listed (vital signs, lab values)
- Suggested actions: order sepsis bundle, blood cultures, lactate
- Response required: acknowledge, accept, or override with reason

### 4.2 AI-Generated Notes (Art AI Charting)

**How they appear:**
- Note appears as a draft within the encounter documentation
- Clearly marked as AI-generated / requiring review
- Clinician can:
  - Review and edit any section
  - Use voice commands to restructure (e.g., "make this a bulleted list")
  - View the "shopping cart" of suggested orders
  - Accept or reject individual orders
  - Sign the note once satisfied
- **Safety guardrails:** Epic and customers monitor metrics like the number of changes made to drafts to prevent AI hallucinations or errors

### 4.3 AI Alerts in In-Basket

**How they appear:**
- AI-drafted message responses appear as draft messages in the clinician's In-Basket
- Clinician sees the original patient message alongside the AI-drafted response
- Clear indication that the response is AI-generated
- Clinician must review, edit (if needed), and explicitly send
- Draft is never auto-sent

### 4.4 Art Insights in Patient Chart

**How they appear:**
- Concise summary panel within the patient chart
- Aggregates information from:
  - Recent chart entries
  - External data (Care Everywhere)
  - Prior notes
  - Lab trends
  - Medication changes
- Presented as an easy-to-read summary for visit preparation or shift handoff
- Used 16+ million times per month

**Sources:**
- [Epic AI Charting](https://www.epic.com/epic/post/epic-ai-charting-rolls-out-alongside-an-expanding-set-of-built-in-ai-capabilities/)
- [Becker's: AI Charting](https://www.beckershospitalreview.com/healthcare-information-technology/ehrs/epic-rolls-out-ai-charting-tool/)

---

## 5. Epic's Governance and Audit Capabilities

### 5.1 AI Trust and Assurance Suite

**What it is:** Free, open-source software suite released on GitHub for healthcare organizations to validate and monitor AI models.

**Capabilities:**
- Automated data collection and mapping
- Near real-time metrics and analysis on AI models
- Validation of any AI model (vendor-supplied or homegrown)
- Continuous monitoring for performance, bias, and unintended consequences
- Eliminates need for manual data mapping by data scientists

**Development:** Created in collaboration with the Health AI Partnership and data scientists at the University of Wisconsin and elsewhere.

**Sources:**
- [Fierce Healthcare: AI validation software](https://www.fiercehealthcare.com/ai-and-machine-learning/epic-plans-launch-ai-validation-software-healthcare-organizations-test)
- [Epic: Open-source AI validation tool](https://www.fiercehealthcare.com/ai-and-machine-learning/epic-releases-ai-validation-software-health-systems)
- [Healthcare IT News: AI validation cookbook](https://www.healthcareitnews.com/news/epics-ai-validation-cookbook-helps-health-systems-review-models-performance)

### 5.2 Audit Trail Logging

**What Epic logs for AI-assisted decisions:**
- All AI outputs (notes, suggestions, alerts) are logged
- Clinician interactions with AI outputs (accept, reject, edit, override) are tracked
- For AI scribes: who recorded audio, which model generated which draft, when edits were made
- All Cosmos data queries are recorded with user attestation
- BPA override reasons are logged
- Changes to AI-drafted notes are tracked (number of edits as quality metric)

### 5.3 Model Version Tracking

**Current capabilities:**
- The AI Trust and Assurance Suite enables tracking of model performance over time
- Organizations can test and validate specific model versions
- Continuous monitoring detects performance drift
- Specific model version information is available for Epic's own models (e.g., ESM version)

**For third-party AI:**
- Integration through SMART on FHIR and CDS Hooks includes metadata about the responding service
- Organizations can configure logging for which CDS Service (and version) provided a recommendation
- AI Charting/ambient tools track which model generated output

### 5.4 Reporting Tools for AI Performance

| Tool | Purpose |
|------|---------|
| AI Trust and Assurance Suite | Validate and continuously monitor AI model performance |
| Slicer Dicer | Ad hoc data exploration and cohort analysis |
| Caboodle/Clarity reports | Detailed analytical reports on AI-assisted outcomes |
| BPA reporting | Alert fire rates, override rates, action rates |
| AI Charting metrics | Number of edits to AI-drafted notes, acceptance rates |

**Sources:**
- [Healthcare IT News: Democratize AI validation](https://www.healthcareitnews.com/news/epic-leads-new-effort-democratize-health-ai-validation)
- [UCSD Center: AI validation](https://healthinnovation.ucsd.edu/news/epic-plans-to-launch-ai-validation-software-for-healthcare-organizations-to-test-monitor-models)

---

## 6. Other EHR AI Platforms

### 6.1 Oracle Health (Cerner)

**Status:** Oracle released a completely new EHR platform (not refurbished Cerner) built from the ground up on Oracle Cloud Infrastructure (August 2025).

**AI capabilities:**
- **Clinical AI Agent:** Combines generative AI, clinical intelligence, multimodal voice, and screen-driven assistance
  - Drafts clinical documentation
  - Proposes next steps (lab tests, follow-up visits)
  - Pulls data from notes to automate coding
- **Voice-activated navigation:** "Voice-first" design philosophy
- **Health Data Intelligence (HDI):** Access to longitudinal patient records from 300+ data sources (clinical, claims, social determinants, pharmacy)
- **Open architecture:** Allows customers to extend Oracle's AI agents, create their own, or add third-party models
- **Agentic AI:** Ramps up with features for patients, prior authorization, and clinical trials

**Current availability:** Ambulatory providers in the US (acute care functionality planned for 2026)

**Key difference from Epic:** Oracle's platform is designed as an open system for third-party AI model integration, while Epic's approach is more controlled through Showroom/Toolbox/Workshop programs.

**Sources:**
- [Advisory.com: Oracle's AI-powered EHR](https://www.advisory.com/daily-briefing/2025/11/20/oracle-ehr)
- [Healthcare Dive: Oracle new EHR](https://www.healthcaredive.com/news/oracle-new-ai-backed-ehr-2025/731398/)
- [Fierce Healthcare: Oracle AI EHR](https://www.fiercehealthcare.com/health-tech/oracle-health-will-offer-new-ehr-2025-embedded-ai-and-analytics-tools)

### 6.2 MEDITECH

**AI capabilities (Expanse platform):**
- **Ambient scribing:** Integrated ambient listening technology as digital scribe
- **No-show appointment prediction:** ML-based scheduling optimization
- **Automated patient messaging:** AI-drafted patient communications
- **Search and summarization:** Collaboration with Google for searching entire EHR (structured/unstructured data, scanned documents, faxes, legacy data, handwritten notes)
- **Discharge documentation:** Auto-generates discharge summaries (saves ~7 min per discharge)
- **MyHealth assistant:** Patient-facing chatbot
- **Clinician chatbots:** Workflow-focused conversational AI
- **Traverse Exchange:** FHIR-based data exchange network using AI for data processing and compilation
- **Agentic UX:** Moving toward agentic user experience where intelligent systems proactively support and automate tasks

**Philosophy:** "AI should support the people making care decisions -- not take away their control."

**Sources:**
- [MEDITECH: AI initiatives](https://ehr.meditech.com/news/meditech-advances-toward-agentic-user-experience-and-previews-new-ai-initiatives)
- [MEDITECH: Expanse AI](https://ehr.meditech.com/ehr-solutions/expanse-artificial-intelligence)

---

## 7. Implications for TELOS

### 7.1 Integration Architecture Summary

For TELOS governance deployed at Mercy Health and Wexner (both Epic sites), the key integration points are:

```
                          +------------------+
                          |   TELOS          |
                          |   Governance     |
                          |   Layer          |
                          +--------+---------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
    +-------v-------+    +--------v--------+    +--------v--------+
    | FHIR R4 APIs  |    | CDS Hooks       |    | Audit/Logging   |
    | (Read/Write)  |    | (patient-view,  |    | (AI Trust &     |
    |               |    |  order-select,  |    |  Assurance)     |
    +-------+-------+    |  order-sign)    |    +--------+--------+
            |            +--------+--------+             |
            |                     |                      |
    +-------v---------------------v----------------------v-------+
    |                    Epic Hyperspace                          |
    |  +----------+ +----------+ +----------+ +----------+      |
    |  | BPA      | | In-Basket| | AI       | | Chart    |      |
    |  | Alerts   | | Messages | | Charting | | Insights |      |
    |  +----------+ +----------+ +----------+ +----------+      |
    |                                                            |
    |  +----------+ +----------+ +----------+ +----------+      |
    |  | Radiant  | | Beaker   | | ASAP     | | Cadence  |      |
    |  | (Rad)    | | (Lab)    | | (ED)     | | (Sched)  |      |
    |  +----------+ +----------+ +----------+ +----------+      |
    +------------------------------------------------------------+
            |                                        |
    +-------v-------+                       +--------v--------+
    | Chronicles    |                       | Clarity/Caboodle|
    | (Operational) |                       | (Analytics/DW)  |
    +---------------+                       +-----------------+
            |                                        |
    +-------v----------------------------------------v--------+
    |                     Epic Cosmos                          |
    |              (De-identified aggregate)                   |
    |              +------------------+                        |
    |              | CoMET/Curiosity  |                        |
    |              | Foundation Model |                        |
    |              +------------------+                        |
    +----------------------------------------------------------+
```

### 7.2 TELOS-Relevant Integration Points

1. **Governance observation via FHIR:** TELOS can read AI-generated DocumentReferences, Observations, and Flags to monitor what AI systems are writing to the chart.

2. **CDS Hooks interception:** TELOS could potentially register as a CDS Service or monitor CDS Hook responses to audit AI-driven clinical decision support in real-time.

3. **BPA audit trail:** Epic's BPA logging provides data on AI alert fire rates, acceptance rates, and override reasons -- critical inputs for TELOS governance reporting.

4. **AI Trust and Assurance Suite alignment:** Epic's open-source validation toolkit is directly complementary to TELOS's governance mission. Integration could provide standardized validation metrics.

5. **Art/ART monitoring:** Epic tracks edit distances on AI-drafted notes and messages -- these metrics could feed into TELOS's AI performance monitoring.

6. **Cosmos governance model:** Epic's "Rules of the Road" governance framework for Cosmos data provides a precedent and potential alignment point for TELOS's governance approach.

### 7.3 Key Technical Constraints

- **Write access is controlled:** Not all FHIR resources support Create/Update from external systems. AI systems that need to write to the chart must work within Epic's approved integration pathways.
- **Draft-only paradigm:** All AI-generated clinical content (notes, messages, orders) must be in draft status requiring clinician review. This is enforced at the platform level.
- **Vendor program requirement:** Third-party AI systems must go through Epic's Showroom/Toolbox/Workshop programs for deep integration. SMART on FHIR provides lighter-weight access.
- **Authentication is mandatory:** All API access requires OAuth 2.0 with scoped permissions.
- **Scoring frequency varies:** ESM scores every 15 minutes; other models may run hourly or on-demand. TELOS monitoring must account for these different cadences.

---

*End of research document. All findings based on publicly available sources as of February 2026.*
