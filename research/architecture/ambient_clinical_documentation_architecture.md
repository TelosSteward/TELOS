# Ambient Clinical Documentation AI: Technical Architecture Analysis
# DAX Copilot (Microsoft/Nuance) and Abridge

**Classification:** Systems Engineering Research -- Technical Architecture Deep Dive
**Date:** 2026-02-16
**Prepared by:** Karpathy (Systems Engineer), TELOS Research Team
**Version:** 1.0
**Status:** For Internal Technical Review

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## 1. Executive Summary

This document maps the exact technical architecture, tool-calling workflows, and data flow pipelines of the two dominant ambient clinical documentation AI systems: **Microsoft DAX Copilot** (Nuance) and **Abridge**. The analysis treats these systems the way TELOS treats the Nearmap property intelligence API -- decomposing every step from input trigger to final EHR write, identifying the models involved, the structured outputs produced, the integration mechanisms used, and the known failure modes.

**Key finding:** Both systems follow a fundamentally similar 8-step action chain (trigger -> capture -> transcribe -> understand -> generate -> review -> submit -> code), but differ significantly in three architectural dimensions:

1. **Model ownership:** Abridge controls its full model stack (proprietary ASR + proprietary LLMs). DAX Copilot combines Nuance's legacy ASR with Microsoft/OpenAI GPT-4 for generation.
2. **Transparency mechanism:** Abridge provides bidirectional "Linked Evidence" (generated text -> source transcript -> source audio). DAX provides transcript-only review with no audio-to-text linking.
3. **EHR integration depth:** DAX embeds natively via Epic SmartSections and "Write My Note" integration. Abridge integrates via Epic's Pal/Workshop program with SmartPhrase mapping from Haiku to Hyperdrive.

Both systems process audio in the cloud (Azure for DAX, undisclosed for Abridge), both require explicit clinician approval before EHR submission, and both report error rates in the 1-3% range -- though even this rate has significant patient safety implications.

---

## 2. DAX Copilot (Microsoft/Nuance) -- Technical Architecture

### 2.1 System Identity

| Property | Value |
|----------|-------|
| **Full Name** | Nuance Dragon Ambient eXperience (DAX) Copilot |
| **Current Branding** | Microsoft Dragon Copilot (as of March 2025) |
| **Parent Company** | Microsoft (acquired Nuance for $19.7B, closed March 2022) |
| **Cloud Infrastructure** | Microsoft Azure (regional data residency, multi-site redundancy) |
| **Certifications** | HITRUST CSF certified, SOC 2 Type 2, FIPS-compliant |
| **Training Data** | 1B+ minutes of medical dictation annually, 10M+ real-world encounters |
| **Supported Platforms** | iOS (PowerMic Mobile 6.0.1+); Android NOT supported |
| **Price Point** | $600-800/month per provider (1-3 year contracts typical) |

### 2.2 Audio Capture Mechanism

DAX Copilot uses **manual-trigger recording**, NOT always-on ambient listening.

```
TRIGGER MECHANISM:
  1. Clinician opens PowerMic Mobile app on iPhone
  2. Clinician explicitly taps "Start Recording" button
  3. Recording indicator appears with elapsed timer
  4. App continues recording even when:
     - Screen sleeps (display off)
     - User switches to other apps
     - Phone calls arrive (recording pauses, auto-resumes)
  5. Clinician taps "Stop Recording" to end capture
  6. Upload begins when connectivity available (Wi-Fi or cellular)

CONSTRAINTS:
  - Maximum recording: 75 minutes (hard limit)
  - Optimal recording: <=45 minutes (summary generation degrades beyond this)
  - Summary text limit: 45 minutes of content even if recording continues
  - Transcript capture continues beyond 45 minutes
  - Bluetooth microphones MUST be disconnected (headsets, AirPods, hearing aids
    will prevent or interrupt recording)
  - Requires Wi-Fi or cellular to START recording
  - If connectivity drops mid-recording, audio stored locally until reconnection
```

**Critical architecture detail:** The recording device is always the clinician's personal iPhone running PowerMic Mobile. There is no room-based microphone array, no always-on ambient sensor, and no hardware appliance. This is a mobile-app-mediated capture system.

### 2.3 Processing Pipeline (Audio -> Note -> EHR)

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: AUDIO CAPTURE                                           │
│ Device: iPhone (PowerMic Mobile app)                            │
│ Format: Compressed audio stream                                 │
│ Storage: Local on device until upload                           │
│ Trigger: Manual tap by clinician                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │ Upload (encrypted HTTPS)
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: CLOUD INGESTION (Azure)                                 │
│ Audio uploaded to Azure data center                             │
│ Regional data residency enforced                                │
│ Audio file DELETED from mobile device after successful upload   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: ASR + DIARIZATION                                       │
│ Engine: Nuance proprietary medical ASR                          │
│   (trained on 1B+ minutes of medical dictation)                 │
│ Outputs:                                                        │
│   - Multi-party transcript (speaker-attributed)                 │
│   - Speaker diarization (clinician vs patient vs others)        │
│   - Medical terminology recognition                             │
│   - Timestamp alignment                                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: DE-IDENTIFICATION (Azure Health Data Services)          │
│ PHI Detection Engine:                                           │
│   - Detects 27 categories of PHI (exceeds HIPAA's 18)          │
│   - Proprietary PHI detection on unstructured text              │
│ Surrogate Generation Engine:                                    │
│   - Replaces tagged PHI with realistic pseudonyms               │
│   - Same-category substitution (name->name, date->date)         │
│   - GDPR pseudonymization compliant (Recital 26)               │
│   - Microsoft does NOT maintain original-to-pseudonym pairs     │
│ Purpose: Create de-identified copy for model improvement        │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: CLINICAL NOTE GENERATION (GPT-4)                        │
│ Model: Microsoft/OpenAI GPT-4 (via Azure OpenAI Service)        │
│ Combined with: Nuance conversational + ambient AI models        │
│ Input: Diarized transcript + encounter context                  │
│ Processing:                                                     │
│   - Specialty detection (auto-identifies clinical domain)       │
│   - Template selection (customizable SmartSection templates)     │
│   - Structured section generation (HPI, ROS, PE, A&P)          │
│   - Diagnosis evidence curation (subjective + objective)        │
│   - Coding suggestions (ICD-10, CPT)                           │
│ Output: Specialty-specific clinical note draft                  │
│ Latency: "Seconds" after recording stops (marketing claim)      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6: CLINICIAN REVIEW                                        │
│ Surfaces: PowerMic Mobile app + Dragon Medical One (desktop)    │
│ Clinician can:                                                  │
│   - Review generated summary alongside full transcript          │
│   - Edit any section of the generated note                      │
│   - Accept/reject individual SmartSections                      │
│   - Add dictation overlays via Dragon Medical One               │
│ Review window: 30 days (summaries auto-deleted after)           │
│ NO audio playback linking (transcript only)                     │
└────────────────────────┬─────────────────────────────────────────┘
                         │ "Sync to EHR" action (manual)
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 7: EHR SUBMISSION (Epic Integration)                       │
│ Integration method: SmartSections + "Write My Note"             │
│ Mechanism:                                                      │
│   - Epic note template must contain DAX Copilot SmartSections   │
│   - SmartSections map to standard note sections                 │
│   - SmartData population fills discrete data elements           │
│   - SmartLink commands trigger SmartData population              │
│   - Clinician clicks "Sync to EHR" to transfer                 │
│   - Content appears in Epic Hyperdrive for final sign-off       │
│ Available in: Epic Haiku (mobile) + Epic Hyperdrive (desktop)   │
│ NOT a direct database write -- goes through Epic's standard     │
│ note authoring workflow                                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 8: BILLING/CODING SUPPORT                                  │
│ Outputs:                                                        │
│   - ICD-10 code suggestions based on documented encounter       │
│   - CPT code suggestions for E/M level                          │
│   - HCC diagnosis capture for risk adjustment                   │
│   - Clinical evidence summaries for code support                │
│ Impact: 11% increase in wRVUs, 14% increase in documented      │
│   HCC diagnoses per encounter (Riverside Health data)           │
│ Note: These are SUGGESTIONS -- coder/clinician must validate    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.4 Model Stack

| Layer | Model | Provenance | Purpose |
|-------|-------|------------|---------|
| ASR | Nuance proprietary medical ASR | Nuance (pre-acquisition), trained on 1B+ min/year | Audio-to-text with medical vocabulary |
| Diarization | Nuance speaker identification | Nuance proprietary | Clinician vs. patient vs. third party |
| De-identification | Azure Health Data Services PHI detector | Microsoft Azure | 27-category PHI detection + surrogate generation |
| Note generation | GPT-4 (Azure OpenAI Service) | OpenAI via Microsoft | Clinical note synthesis from transcript |
| Conversational AI | Nuance ambient + conversational models | Nuance proprietary | Context understanding, intent classification |
| Template engine | Customizable template prompts | Customer-configurable | Specialty-specific note formatting |

### 2.5 Structured Outputs

```
GENERATED SECTIONS (SmartSections):
  - History of Present Illness (HPI)
  - Review of Systems (ROS)
  - Physical Examination (PE)
  - Allergies
  - Assessment and Plan (A&P)
    - Numbered problem list format
    - Bulleted sub-items under each problem
  - After-Visit Summary (patient-facing)
  - Referral Letters

CODING OUTPUTS:
  - ICD-10 code suggestions
  - CPT code suggestions (E/M level)
  - HCC diagnosis codes
  - Clinical evidence summaries supporting each code

SECONDARY OUTPUTS:
  - Full multi-party transcript
  - Encounter duration metadata
  - Speaker attribution data
```

### 2.6 Epic Integration Mechanism (Detailed)

DAX Copilot's Epic integration operates through a **SmartSection injection model**, NOT through FHIR REST APIs or HL7 messaging:

```
INTEGRATION ARCHITECTURE:

  1. PREREQUISITE: Epic note template must include DAX Copilot SmartSections
     - Can use pre-built SmartPhrases with embedded SmartSections
     - Can manually add SmartSections to personal SmartPhrases
     - Currently available: Allergies, Physical Exam, Assessment & Plan
     - Future: HPI, ROS, additional sections

  2. RECORDING CONTEXT:
     - DAX Copilot for Epic embeds directly in Haiku (mobile) and
       Hyperdrive (desktop)
     - The embedded DAX Copilot window opens within the EHR
     - Recording can be initiated from within Epic workflow

  3. NOTE TRANSFER:
     - Generated content populates SmartSection placeholders in the note
     - SmartData mapping converts AI text to discrete data elements
     - SmartLink commands trigger SmartData population with AI-extracted
       values (e.g., BP reading -> BP SmartData element -> quality dashboard)
     - "Sync to EHR" button transfers any manual edits from DAX back to Epic

  4. AUTHENTICATION:
     - OAuth 2.0 with Epic FHIR endpoints (standard)
     - Short-lived tokens with refresh
     - Scoped access per application registration
     - Epic App Orchard certification required

  5. DATA FLOW:
     - Epic -> DAX: Patient context, encounter metadata, note template
     - DAX -> Epic: Generated note sections, SmartData values, coding suggestions
     - NOT direct database writes -- all through Epic's standard note workflow
```

### 2.7 Data Retention Policy

| Data Type | Operational Retention | Product Improvement |
|-----------|----------------------|---------------------|
| Audio recordings | 90 days max | 1% sample stored 1 year |
| Transcripts | 90 days max (30 days visible to clinician) | Retained in de-identified form |
| AI-generated drafts | 90 days max | De-identified copies retained |
| PHI mappings | Not retained (no re-identification possible) | N/A |
| Audio on device | Deleted after successful upload | N/A |

---

## 3. Abridge -- Technical Architecture

### 3.1 System Identity

| Property | Value |
|----------|-------|
| **Full Name** | Abridge |
| **Founded** | 2018 (Pittsburgh Health Data Alliance -- CMU + Pitt + UPMC) |
| **Founders** | Shivdev Rao (CEO, cardiologist), Florian Metze (CSO), Sandeep Konam (CTO) |
| **AI System Name** | "Ears" (internal name for the combined ASR + note generation system) |
| **Training Data** | 1.5M+ fully consented, de-identified medical encounters |
| **Language Support** | 14+ languages (English primary, Spanish, French, German, Italian, Chinese, Japanese, Haitian Creole, others) |
| **Specialty Coverage** | 50+ specialties |
| **Epic Status** | First "Pal" in Epic's Partners and Pals program; Workshop participant |
| **Deployments** | 150+ health systems (Kaiser, Yale New Haven, Duke, Emory, UChicago, Sutter, UNC) |
| **Price Point** | ~$2,500/seat/year |
| **Funding** | $300M Series E (a16z led, 2025) |

### 3.2 Key Architectural Difference: Full-Stack Model Ownership

Abridge's fundamental technical differentiator is that it controls its **entire model pipeline** -- proprietary ASR and proprietary LLMs -- rather than depending on third-party models like GPT-4:

```
ABRIDGE MODEL STACK (Proprietary):

  ┌─────────────────────────────────────────────────┐
  │ LAYER 1: Medical ASR (Proprietary)              │
  │ Purpose: Audio -> medical transcript            │
  │ Training: 1.5M+ consented medical encounters    │
  │ Performance vs. competitors:                    │
  │   - 16% lower WER than Google Medical Conv.     │
  │   - 45% fewer errors than OpenAI Whisper v3     │
  │ Outputs:                                        │
  │   - Speaker-diarized transcript                 │
  │   - Audio-to-text timestamp alignment           │
  │   - Speaker attribution ("diarization")         │
  │   - Cross-talk handling                         │
  │   - Background noise filtering                  │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │ LAYER 2: Clinical Note Generation (Proprietary) │
  │ Purpose: Transcript -> structured clinical note │
  │ Architecture: Purpose-built healthcare LLMs     │
  │   (NOT GPT-4 or other third-party LLMs)         │
  │ Combined with:                                  │
  │   - Medical ontology classifiers                │
  │   - SNOMED CT mapping                           │
  │   - LOINC mapping                               │
  │ Outputs:                                        │
  │   - Structured SOAP note sections               │
  │   - Linked Evidence metadata                    │
  │   - Ontology-coded medical concepts             │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │ LAYER 3: Confabulation Detection (Proprietary)  │
  │ Purpose: Catch and correct hallucinations       │
  │ Architecture:                                   │
  │   - In-house classifier (NOT GPT-4o)            │
  │   - Trained on 50,000+ curated examples         │
  │   - Classification: Support x Severity axes     │
  │ Performance:                                    │
  │   - Catches 97% of confabulations               │
  │   - GPT-4o comparison: catches only 82%         │
  │   - Abridge misses 6x fewer errors than GPT-4o  │
  │ Action: Auto-correction using transcript + EHR  │
  │   context as ground truth                       │
  └─────────────────────────────────────────────────┘
```

### 3.3 Linked Evidence -- The Transparency Architecture

Abridge's "Linked Evidence" system is the technical feature that most differentiates it from DAX. It provides **bidirectional traceability** from any generated text back to both the source transcript AND source audio:

```
LINKED EVIDENCE DATA FLOW:

  Generated Note Section
       │
       │  (user clicks/highlights any word or phrase)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │ LINKED EVIDENCE ENGINE                      │
  │                                             │
  │  1. Text-to-transcript mapping:             │
  │     Generated phrase -> source transcript   │
  │     segment that substantiates it           │
  │                                             │
  │  2. Transcript-to-audio mapping:            │
  │     Transcript segment -> audio timestamp   │
  │     (via ASR alignment data)                │
  │                                             │
  │  3. Audio playback:                         │
  │     Clinician can LISTEN to the exact       │
  │     moment in the conversation that         │
  │     generated this text                     │
  └─────────────────────────────────────────────┘

  IMPLEMENTATION:
  - ASR produces timestamp-aligned transcript as a secondary output
  - Note generation model maintains provenance metadata linking
    each generated phrase to source transcript spans
  - UI renders clickable highlights in the Abridge tab within Epic
  - Clicking any region surfaces:
    a) The source transcript passage
    b) Audio playback starting at the corresponding timestamp
  - This is NOT post-hoc retrieval -- it's built into the generation
    pipeline as a first-class output
```

This is architecturally significant because it transforms clinician review from "read and hope it's right" to "click and verify against source." It also creates an audit trail that connects every word in the final note to evidence in the original conversation.

### 3.4 Confabulation Taxonomy

Abridge published a research framework classifying hallucinations ("confabulations") along two axes:

```
CONFABULATION CLASSIFICATION MATRIX:

                        LOW SEVERITY          HIGH SEVERITY
                    ┌────────────────────┬────────────────────┐
  UNSUPPORTED       │ Minor addition     │ Fabricated         │
  (not in           │ (e.g., added       │ diagnosis,         │
   transcript)      │  "mild" when not   │ medication, or     │
                    │  explicitly stated) │ exam finding       │
                    ├────────────────────┼────────────────────┤
  CONTRADICTED      │ Minor              │ Wrong medication,  │
  (contradicts      │ inconsistency      │ wrong patient      │
   transcript)      │ (e.g., left vs     │ attribution,       │
                    │  right side)        │ opposite diagnosis │
                    ├────────────────────┼────────────────────┤
  SUPPORTED         │ N/A (correct)      │ N/A (correct)      │
  (matches          │                    │                    │
   transcript)      │                    │                    │
                    └────────────────────┴────────────────────┘

  DETECTION PIPELINE:
  1. For each claim in the generated note:
     a. Classify SUPPORT level (supported / unsupported / contradicted)
     b. Classify SEVERITY (low / high based on clinical impact)
  2. Flag unsupported + high-severity as CRITICAL
  3. Auto-correct using transcript + EHR context as ground truth
  4. Surface remaining flags to clinician via Linked Evidence UI
```

### 3.5 Epic Integration Architecture (Abridge Inside)

```
INTEGRATION TIERS:

  TIER 1: "Pal" Status (August 2023)
    - First third-party AI in Epic's Partners and Pals program
    - Generative AI summaries integrated into Epic clinical workflows
    - Abridge tab appears within Epic interface
    - Linked Evidence available within the Abridge tab

  TIER 2: "Abridge Inside" (February 2024)
    - Full embedding from Haiku (mobile) to Hyperdrive (desktop)
    - Recording initiated from within Haiku
    - AI-generated notes embedded within patient chart in Hyperdrive
    - SmartPhrase and template integration for review and rapid note closure

  TIER 3: "Workshop" Co-Development (January 2025+)
    - Epic Workshop = collaborative co-development program
    - Abridge Inside for Emergency Medicine (ASAP module integration)
    - Abridge Inside for Inpatient (H&P, progress notes, consult notes)
    - Outpatient Orders (medications from conversation -> Epic orders)

  NOTE TYPE SUPPORT (Inpatient):
    - History & Physical (H&P)
    - Progress Notes
    - Consult Notes
    - Epic auto-maps generated note into selected note type

  ORDERS WORKFLOW (Pilot):
    - AI identifies medications discussed during encounter
    - Surfaces them inside Epic for one-click order placement
    - Clinician confirms/modifies before order submission
    - Currently outpatient only, via Workshop pilot

  INTEGRATION MECHANISM:
    - SmartPhrase integration for note templates
    - SmartData population for discrete data elements
    - FHIR-based API connections (standard Epic FHIR endpoints)
    - OAuth 2.0 authentication with scoped access
    - Real-time note generation (not batch)
```

### 3.6 Structured Outputs

```
CLINICAL DOCUMENTATION:
  - SOAP note sections (Subjective, Objective, Assessment, Plan)
  - Specialty-adapted formats (50+ specialties)
  - Note types: H&P, progress notes, consult notes, ED notes

CODED DATA:
  - ICD-10 code suggestions with supporting evidence
  - HCC diagnosis codes with MEAT criteria documentation
  - Visit diagnoses
  - Linked Evidence grounding each code to conversation

ONTOLOGY MAPPING:
  - SNOMED CT concept mapping
  - LOINC observation mapping
  - Deep EHR integration via coded data elements

REVENUE CYCLE:
  - Audit-ready documentation (Linked Evidence = audit trail)
  - Billable documentation generated from conversation
  - CPT-ready encounter documentation

TRANSPARENCY METADATA:
  - Linked Evidence mappings (text -> transcript -> audio)
  - Provenance chain for every generated phrase
  - Speaker attribution for all transcript content
```

---

## 4. Comparative Architecture Analysis

### 4.1 Head-to-Head Technical Comparison

| Dimension | DAX Copilot (Microsoft/Nuance) | Abridge |
|-----------|-------------------------------|---------|
| **ASR Engine** | Nuance proprietary (legacy, pre-acquisition) | Proprietary, purpose-built for medical conversations |
| **ASR Training Data** | 1B+ min/year dictation | 1.5M+ consented encounters |
| **LLM for Note Generation** | GPT-4 (Azure OpenAI Service) | Proprietary healthcare LLMs |
| **Third-party LLM Dependency** | Yes (OpenAI GPT-4) | No (full-stack proprietary) |
| **Hallucination Detection** | Not published as separate system | Dedicated classifier (97% detection, 50K training examples) |
| **Audio-to-text Linking** | No (transcript review only) | Yes (Linked Evidence -- click any text to hear source audio) |
| **Cloud Provider** | Azure (explicit) | Not publicly disclosed |
| **Epic Integration** | SmartSections + Write My Note (embedded in Haiku/Hyperdrive) | Pal + Workshop + Abridge Inside (Haiku to Hyperdrive) |
| **Orders Integration** | Not announced | Pilot via Workshop (outpatient medications) |
| **Inpatient Support** | Limited (primarily ambulatory) | Yes (H&P, progress notes, consult notes via Workshop) |
| **ED Support** | Limited | Yes (ASAP module integration) |
| **Language Support** | Multiple (number not specified) | 14+ languages |
| **Recording Device** | iPhone only (PowerMic Mobile) | Multiple (app-based) |
| **Max Recording** | 75 minutes (45 min for optimal summary) | Not publicly limited |
| **Data Retention (Audio)** | 90 days (1% sample: 1 year) | Not publicly specified |
| **Price** | $600-800/month/provider | ~$2,500/year/seat ($208/month) |
| **Specialties** | "Dozens" | 50+ |
| **Consumer App** | No | Yes (patient-facing) |

### 4.2 Architecture Diagrams -- Side by Side

```
DAX COPILOT:
  iPhone App ──(HTTPS)──> Azure ──> Nuance ASR ──> GPT-4 ──> SmartSections ──> Epic
                           │                                       ▲
                           └── Azure De-ID Service ────────────────┘
                               (PHI pseudonymization)

ABRIDGE:
  App (Haiku) ──(HTTPS)──> Cloud ──> Abridge ASR ──> Abridge LLM ──> Confab Detector ──> Epic
                                         │               │                  │
                                         └── Timestamp ──┘                  │
                                             Alignment    │                 │
                                                          └── Linked ──────┘
                                                              Evidence
                                                              Metadata
```

---

## 5. The Clinical Workflow as an Action Chain

### 5.1 Full 8-Step Action Chain (Generalized)

```
STEP 1: TRIGGER
  ├─ DAX: Clinician opens PowerMic Mobile, taps "Start Recording"
  ├─ Abridge: Clinician initiates recording from Haiku or Abridge app
  ├─ Both: Patient consent obtained (verbal or written, per org policy)
  └─ Data: encounter_id, patient_context, clinician_id, timestamp

STEP 2: AUDIO CAPTURE
  ├─ Input: Ambient room audio via device microphone
  ├─ Processing: Continuous streaming or buffered capture
  ├─ Duration: Typically 10-45 minutes (primary care: ~15 min avg)
  ├─ Participants: Clinician + patient + (optional: family, interpreter, nurse)
  ├─ Telehealth: Single mic captures both sides of virtual call
  └─ Output: Compressed audio file (.m4a or similar)

STEP 3: TRANSCRIPTION (ASR + Diarization)
  ├─ Tool: Medical ASR engine (Nuance proprietary / Abridge proprietary)
  ├─ Sub-steps:
  │   a. Speech-to-text conversion (medical vocabulary)
  │   b. Speaker diarization (who said what)
  │   c. Timestamp alignment (text <-> audio positions)
  │   d. Medical term normalization
  │   e. Cross-talk resolution
  │   f. Background noise filtering
  └─ Output: {
       "transcript": [...],
       "speakers": ["clinician", "patient", ...],
       "timestamps": [[start, end], ...],
       "confidence": [0.0-1.0 per segment]
     }

STEP 4: CLINICAL NLP (Understanding)
  ├─ Tool: LLM + clinical classifiers + ontology mappers
  ├─ Sub-steps:
  │   a. Intent classification (chief complaint, history, exam, plan)
  │   b. Medical entity extraction (conditions, medications, procedures)
  │   c. Ontology mapping (concepts -> SNOMED CT, LOINC codes)
  │   d. Diagnosis evidence curation (subjective + objective elements)
  │   e. Specialty detection (auto-identify clinical domain)
  │   f. [Abridge only] Confabulation detection pass
  └─ Output: Structured clinical entities + coded concepts

STEP 5: NOTE GENERATION
  ├─ Tool: LLM (GPT-4 for DAX / proprietary for Abridge)
  ├─ Input: Structured entities + transcript + template
  ├─ Template selection: Specialty-specific + organization-customizable
  ├─ Sections generated:
  │   - HPI (History of Present Illness)
  │   - ROS (Review of Systems)
  │   - Physical Examination
  │   - Allergies
  │   - Assessment & Plan (numbered problem list)
  │   - [Optional] After-Visit Summary
  │   - [Optional] Referral Letters
  ├─ Coding suggestions: ICD-10, CPT, HCC
  ├─ [Abridge only] Linked Evidence metadata attached
  └─ Output: Draft clinical note + coding suggestions + provenance metadata

STEP 6: CLINICIAN REVIEW
  ├─ Interface: EHR-embedded (Hyperdrive/Haiku) or companion app
  ├─ Actions available:
  │   a. Read generated note sections
  │   b. Edit any text
  │   c. Accept/reject individual sections
  │   d. [Abridge] Click any phrase -> see transcript -> hear audio
  │   e. [DAX] Review transcript alongside note
  │   f. Add dictation overlays (DAX via Dragon Medical One)
  │   g. Verify coding suggestions
  ├─ Typical review time: <60 seconds for clean notes
  └─ Required: YES -- clinician MUST review and approve before EHR entry

STEP 7: EHR SUBMISSION
  ├─ Mechanism:
  │   - DAX: "Sync to EHR" -> SmartSections populate note template
  │   - Abridge: SmartPhrase mapping -> note closure within Epic
  ├─ Data written:
  │   a. Clinical note text (narrative sections)
  │   b. SmartData discrete elements (vitals, codes, structured fields)
  │   c. Encounter metadata
  ├─ NOT direct database writes -- flows through Epic's standard
  │   note authoring and signing workflow
  ├─ Clinician signs/locks the note through normal Epic process
  └─ Note becomes part of permanent medical record

STEP 8: BILLING/CODING
  ├─ AI-suggested codes available for:
  │   a. ICD-10 diagnosis codes
  │   b. CPT procedure/E&M codes
  │   c. HCC codes for risk adjustment
  │   d. MEAT criteria documentation for HCC support
  ├─ Revenue cycle impact:
  │   - More complete diagnosis capture (14% more HCC codes)
  │   - Higher E/M level support (11% wRVU increase)
  │   - Audit-ready documentation (especially Abridge Linked Evidence)
  ├─ Clinician and/or coder must validate all suggestions
  └─ Concern: "Coding arms race" -- more complete != upcoding,
     but payers are responding with downcoding and recalibration
```

### 5.2 Action Chain as Tool Calls (TELOS-Style Mapping)

```python
# Pseudocode action chain -- how TELOS would see this workflow

action_chain = [
    {
        "step": 1,
        "tool": "encounter_initiate",
        "input": {"patient_id", "encounter_type", "consent_obtained"},
        "output": {"encounter_id", "recording_session_id"},
        "risk_tier": "always_allowed",
        "governance": "identity_verification"
    },
    {
        "step": 2,
        "tool": "audio_capture",
        "input": {"encounter_id", "device_mic", "participants"},
        "output": {"audio_blob", "duration_seconds"},
        "risk_tier": "always_allowed",
        "governance": "consent_verified, PHI_in_transit"
    },
    {
        "step": 3,
        "tool": "medical_asr_transcribe",
        "input": {"audio_blob"},
        "output": {"diarized_transcript", "timestamps", "speaker_labels"},
        "risk_tier": "always_allowed",
        "governance": "PHI_processing, cloud_upload"
    },
    {
        "step": 4,
        "tool": "clinical_nlp_extract",
        "input": {"diarized_transcript"},
        "output": {"entities", "snomed_codes", "loinc_codes", "intent_classes"},
        "risk_tier": "requires_confirmation",
        "governance": "medical_reasoning, hallucination_risk"
    },
    {
        "step": 5,
        "tool": "clinical_note_generate",
        "input": {"entities", "transcript", "note_template", "specialty"},
        "output": {"draft_note", "coding_suggestions", "linked_evidence"},
        "risk_tier": "requires_human_review",
        "governance": "CRITICAL -- generated clinical content, must not fabricate"
    },
    {
        "step": 6,
        "tool": "clinician_review_gate",
        "input": {"draft_note", "linked_evidence", "transcript"},
        "output": {"approved_note", "edits_made", "sections_rejected"},
        "risk_tier": "human_in_the_loop",
        "governance": "mandatory_human_review, edit_audit_trail"
    },
    {
        "step": 7,
        "tool": "ehr_submit",
        "input": {"approved_note", "smartdata_elements", "encounter_id"},
        "output": {"note_id_in_ehr", "submission_timestamp"},
        "risk_tier": "requires_confirmation",
        "governance": "permanent_medical_record, non_reversible"
    },
    {
        "step": 8,
        "tool": "billing_code_suggest",
        "input": {"approved_note", "encounter_data"},
        "output": {"icd10_suggestions", "cpt_suggestions", "hcc_codes"},
        "risk_tier": "requires_confirmation",
        "governance": "revenue_impact, upcoding_risk, audit_exposure"
    }
]
```

---

## 6. Data Flow Analysis -- What PHI Goes Where

### 6.1 PHI Captured in Audio

Every ambient clinical recording inherently captures:

| PHI Category | Examples | HIPAA Category |
|-------------|----------|----------------|
| Patient name | "Mrs. Johnson, how are you feeling?" | Name |
| Date of birth | "You're 67, right?" | DOB |
| Medical history | "You had that bypass in 2019" | Medical record |
| Medications | "You're still on metformin 500mg twice daily?" | Treatment |
| Diagnoses | "Your A1C is 8.2, so your diabetes isn't controlled" | Diagnosis |
| Family history | "Your mother had breast cancer at 52" | Family history |
| Social history | "I live at 425 Oak Street with my husband" | Address |
| Insurance | "Your Blue Cross plan covers this" | Health plan |
| Phone numbers | "Call my daughter at 555-0142 if anything changes" | Contact |
| Other patients | "The patient before you had something similar" | Third-party PHI |
| Sexual/reproductive | "When was your last period?" | Sensitive PHI |
| Mental health | "I've been feeling really depressed" | Sensitive PHI |
| Substance use | "I drink about a bottle of wine a night" | Sensitive PHI |

### 6.2 Data Flow Map

```
                    ┌──────────────────────────────────┐
                    │         CLINICIAN DEVICE          │
                    │  (iPhone / Workstation)           │
                    │                                   │
                    │  Audio: Captured locally           │
                    │  -> Encrypted                     │
                    │  -> Uploaded to cloud             │
                    │  -> DELETED from device           │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │         VENDOR CLOUD              │
                    │  (Azure for DAX / TBD for Abridge)│
                    │                                   │
                    │  Audio: Processed for transcript   │
                    │  Transcript: Generated             │
                    │  Note draft: Generated             │
                    │                                   │
                    │  RETENTION (DAX):                 │
                    │  - Audio: 90 days                 │
                    │  - 1% audio sample: 1 year        │
                    │  - Transcripts: 90 days           │
                    │  - De-identified copies: ongoing   │
                    │                                   │
                    │  DE-IDENTIFICATION (DAX):         │
                    │  - 27 PHI categories detected      │
                    │  - Pseudonymized copies created    │
                    │  - Used for model improvement      │
                    │  - No re-identification maintained │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │         EHR SYSTEM (Epic)         │
                    │                                   │
                    │  WHAT ENTERS THE EHR:             │
                    │  - Approved clinical note text     │
                    │  - SmartData discrete elements     │
                    │  - Coding suggestions              │
                    │  - Encounter metadata              │
                    │                                   │
                    │  WHAT STAYS IN VENDOR SYSTEM:      │
                    │  - Audio recordings (temporary)    │
                    │  - Full transcript (temporary)     │
                    │  - Linked Evidence metadata        │
                    │    (Abridge -- accessible from     │
                    │     within Epic but stored by      │
                    │     Abridge)                       │
                    │  - De-identified training data     │
                    │  - Governance/audit metadata       │
                    │                                   │
                    │  WHAT IS DISCARDED:               │
                    │  - Audio after retention window    │
                    │  - PHI-to-pseudonym mappings       │
                    │  - Session state                   │
                    └──────────────────────────────────┘
```

### 6.3 Processing Location

| Processing Step | DAX Copilot | Abridge |
|----------------|-------------|---------|
| Audio capture | On-device (iPhone) | On-device (app) |
| Audio storage (temp) | On-device until upload | On-device until upload |
| ASR/Transcription | Cloud (Azure) | Cloud (vendor-hosted) |
| Note generation | Cloud (Azure OpenAI) | Cloud (vendor-hosted) |
| Clinician review | On-device (app + desktop) | On-device (within Epic) |
| EHR submission | Epic server (on-prem or hosted) | Epic server |
| De-identification | Cloud (Azure Health Data Services) | Cloud (vendor-hosted) |
| Model training | Cloud (Azure, de-identified data) | Cloud (de-identified data) |

**No on-device ML inference** in either system for the primary pipeline. Some systems (not DAX/Abridge specifically) have explored on-device PHI redaction before cloud upload, but production systems currently process full PHI in the cloud under BAA protections.

---

## 7. Known Failure Modes

### 7.1 Published Error Taxonomy

Based on peer-reviewed literature (PMC, npj Digital Medicine, Surgical Endoscopy, NEJM Catalyst):

```
ERROR TYPE 1: HALLUCINATED MEDICAL FACTS
  - Fabricated medications not discussed in encounter
  - Documented examinations that never occurred
  - Nonexistent diagnoses inserted into note
  - Example: "Patient is on lisinopril 10mg" when lisinopril was never mentioned
  - Rate: 1-3% of generated notes contain some hallucination

ERROR TYPE 2: MISATTRIBUTION
  - Family history attributed to patient
  - Example: "Patient has diabetes" when patient said "my mother has diabetes"
  - One of the most dangerous failure modes -- creates persistent EHR errors
  - Particularly problematic for conditions that affect treatment decisions

ERROR TYPE 3: CRITICAL OMISSIONS
  - Symptoms mentioned but not documented
  - Medication changes discussed but not captured
  - Safety-critical information missed (allergies, adverse reactions)
  - Example: Patient mentions chest pain but it doesn't appear in HPI

ERROR TYPE 4: CONTEXTUAL MISINTERPRETATION
  - Left/right confusion
  - Temporal confusion (current vs. historical)
  - Negation mishandling ("no chest pain" -> "chest pain")
  - Conditional mishandling ("if pain worsens, take..." -> "take...")

ERROR TYPE 5: FABRICATED SNOMED/ICD CODES
  - LLMs can generate plausible but nonexistent codes
  - If not validated against code databases, false codes propagate
  - Downstream impact on billing, quality reporting, and population health

ERROR TYPE 6: SPEAKER DIARIZATION ERRORS
  - Misidentifying who said what in multi-party conversations
  - Particularly problematic with interpreters, family members, nurses
  - Can lead to patient statements attributed to clinician and vice versa

ERROR TYPE 7: UPCODING BIAS
  - More complete documentation leads to higher billing codes
  - Not necessarily inaccurate, but payers view as systematic upcoding
  - Risk-adjustment implications for Medicare Advantage plans
  - Published evidence: 11% wRVU increase, 14% more HCC diagnoses
```

### 7.2 Published Accuracy Data

| System | Study | Metric | Result |
|--------|-------|--------|--------|
| **DAX Copilot** | Surgical Endoscopy (2025) | Note quality score (simulated inpatient) | 46.91/50 (93.8%) |
| **DAX Copilot** | Same study | Hallucination rate | 0% in 25 simulated scenarios |
| **DAX Copilot** | Same study | Domains assessed | Accuracy, thoroughness, comprehensibility, succinctness, synthesis, internal consistency |
| **Abridge** | Internal benchmark (2025) | Confabulation detection rate | 97% (vs. GPT-4o at 82%) |
| **Abridge** | Same benchmark | Dataset size | 10,000+ clinical encounters |
| **Abridge** | ASR performance (2025) | WER vs. Google Medical | 16% lower word error rate |
| **Abridge** | ASR performance (2025) | Error reduction vs. Whisper v3 | 45% fewer errors |
| **General** | npj Digital Medicine (2025) | Overall AI scribe error rate | ~1-3% |
| **General** | PMC systematic review (2025) | Adoption rate | ~30% of physician practices |

**Critical caveat:** The DAX Copilot surgical study used 25 **simulated** encounters (authors provided voices in controlled setting), NOT real clinical encounters. Simulated environments tend to produce cleaner audio with less cross-talk, fewer interruptions, and more structured dialogue than real clinical settings. The 0% hallucination rate in simulation should not be extrapolated to production performance.

---

## 8. Clinical Population Workflow Variations

### 8.1 Primary Care (Outpatient Office Visit)

```
WORKFLOW:
  Recording: 10-20 minutes typical
  Note type: Office visit / E&M note
  Sections: HPI, ROS, PE, Allergies, Medications, A&P
  Coding: E&M level (99213-99215), ICD-10 for each diagnosis
  Special considerations:
    - Multiple problems per visit (3-5 typical)
    - Chronic disease management (longitudinal context)
    - Preventive care elements (screenings, immunizations)
    - Medication reconciliation
  AI challenges:
    - Distinguishing current problems from historical mentions
    - Tracking medication changes across visits
    - Capturing all problems discussed (often 5+ per visit)
```

### 8.2 Emergency Department

```
WORKFLOW:
  Recording: Variable (5-60 minutes, often interrupted)
  Note type: ED note (per ASAP module in Epic)
  Sections: Chief complaint, HPI, ROS, PE, MDM, Disposition
  Coding: ED E&M (99281-99285), critical care (99291)
  Special considerations:
    - Multiple interruptions (other patients, consults, procedures)
    - High acuity, fast-paced conversations
    - Handoffs between providers
    - Procedures interspersed with documentation
    - Disposition planning (admit vs. discharge)
  AI challenges:
    - Fragmented conversations (5 min here, 10 min later)
    - Multiple patients seen concurrently
    - Critical information density (allergies, medications in emergencies)
    - Recording start/stop across interrupted encounters
  Integration: Abridge Inside for Emergency Medicine integrates with
    Epic's ASAP module in Haiku and Hyperspace
```

### 8.3 Inpatient (Hospital Medicine)

```
WORKFLOW:
  Recording: 5-15 minutes per encounter (rounding conversations)
  Note types: H&P (admission), Daily Progress Notes, Consult Notes
  Sections vary by note type:
    - H&P: Full history, complete exam, medical decision making
    - Progress: Interval changes, focused exam, updated plan
    - Consult: Reason for consult, focused assessment, recommendations
  Coding: Inpatient E&M (99221-99223 initial, 99231-99233 subsequent)
  Special considerations:
    - Multiple encounters per day per patient
    - Team-based care (attending + resident + students)
    - Longitudinal narrative across admission
    - Handoff documentation
  AI challenges:
    - Tracking which information is new vs. carried forward
    - Multiple voices in team rounds
    - Distinguishing teaching discussion from clinical decisions
    - Integration with nursing notes, orders, results
  Status: Abridge Inside for Inpatient launched June 2025;
    DAX Copilot primarily ambulatory-focused
```

### 8.4 Specialist Encounters

```
CARDIOLOGY EXAMPLE:
  Recording: 15-30 minutes
  Additional sections: Cardiac history, risk factors, imaging review
  Structured data: Echo measurements, EKG interpretation, stress test results
  Challenge: Interpreting imaging/test results discussed verbally
  Challenge: Longitudinal tracking (repeat echos, trend data)

OB/GYN EXAMPLE:
  Recording: 10-20 minutes
  Additional sections: OB history (G/P), gestational age, fetal assessment
  Structured data: Fundal height, fetal heart tones, cervical status
  Challenge: Highly sensitive content (reproductive history, STIs)
  Challenge: Combined prenatal + general care in single visit

ONCOLOGY EXAMPLE:
  Recording: 20-45 minutes
  Additional sections: Cancer staging, treatment protocol, response assessment
  Challenge: Complex longitudinal narrative across many visits
  Challenge: Treatment protocol tracking (cycle numbers, dose modifications)
  Challenge: "Early ambient systems struggled to support complex or nuanced
    workflows, with oncologists needing tools that could track longitudinal
    narratives across visits" (published finding)
```

### 8.5 Telehealth Visits

```
WORKFLOW:
  Recording: Same as in-person equivalent
  Audio capture: Single microphone captures both sides of video call
  Special considerations:
    - Audio quality depends on patient's connection/hardware
    - No physical exam (limited ROS/PE sections)
    - Screen sharing of results during encounter
    - May have interpreter on three-way call
  AI challenges:
    - Variable audio quality from patient side
    - Echo and feedback artifacts
    - Difficulty with speaker diarization when audio quality is poor
    - Missing physical exam data that can't be fabricated
  Both DAX and Abridge support telehealth encounters
```

---

## 9. Reference Architecture: NVIDIA Ambient Provider Blueprint

For comparison, NVIDIA published an open-source reference architecture (Apache 2.0) that reveals the engineering pipeline explicitly:

```
NVIDIA AMBIENT PROVIDER BLUEPRINT:

  Technology Stack:
    ASR: Parakeet model (via NVIDIA NIM)
    Diarization: Included in NIM ASR service
    LLM: Llama-3.3-Nemotron-Super-49B-v1 (reasoning model)
    Deployment: Docker containers

  Pipeline:
    1. Audio input (recorded consultation)
    2. NIM ASR with diarization -> transcript + speaker IDs
    3. Reasoning LLM + SOAP template prompt -> structured note
    4. Rich text editor with citation support -> clinician review
    5. Note export

  Key architectural insights from the blueprint:
    - LLM is guided by a "prompt" (standardized note template + style guidance)
    - GPT-4 32K context window used in some implementations
    - Modular design allows swapping ASR and LLM components
    - Real-time processing with live progress updates
    - Citation support links generated text to transcript sources

  Source: github.com/NVIDIA-AI-Blueprints/ambient-provider (Apache 2.0)
```

This reference architecture confirms the general pipeline pattern used by commercial systems, though commercial systems add significant additional layers (de-identification, ontology mapping, EHR integration, compliance controls) that the blueprint omits.

---

## 10. TELOS Governance Implications

### 10.1 Where Governance Fits in the Action Chain

Every step in the ambient documentation pipeline maps to a TELOS governance evaluation point:

| Step | Tool | TELOS Risk Tier | Governance Concern |
|------|------|-----------------|--------------------|
| 1. Trigger | `encounter_initiate` | `always_allowed` | Consent verification |
| 2. Capture | `audio_capture` | `always_allowed` | PHI scope, recording boundaries |
| 3. Transcribe | `medical_asr_transcribe` | `always_allowed` | Accuracy, speaker attribution |
| 4. Understand | `clinical_nlp_extract` | `requires_confirmation` | Entity extraction accuracy, ontology mapping correctness |
| 5. Generate | `clinical_note_generate` | `requires_human_review` | Hallucination risk, omission risk, fabrication risk |
| 6. Review | `clinician_review_gate` | `human_in_the_loop` | Review thoroughness, automation bias |
| 7. Submit | `ehr_submit` | `requires_confirmation` | Permanent record, downstream clinical impact |
| 8. Code | `billing_code_suggest` | `requires_confirmation` | Upcoding risk, audit exposure, revenue integrity |

### 10.2 Critical Governance Gaps in Current Systems

1. **No continuous fidelity measurement.** Neither system measures whether the generated note drifts from the conversation's actual clinical content in embedding space. They rely on post-hoc human review.

2. **No action chain integrity verification.** There is no SCI (Semantic Chain Integrity) tracking to verify that each step's output is semantically consistent with the previous step's input.

3. **No boundary corpus enforcement.** Neither system has formalized "out of scope" boundaries -- what the AI should explicitly refuse to document or infer.

4. **No governance receipt.** There is no cryptographically signed audit trail linking the audio -> transcript -> note -> approval chain with tamper-evident properties.

5. **Review thoroughness is unmonitored.** When a clinician approves a note in under 60 seconds, there is no measurement of whether they actually reviewed the content or clicked through.

### 10.3 Primacy Attractor Mapping

For TELOS integration with ambient clinical documentation:

```
PRIMACY ATTRACTOR: "Accurately document the clinical encounter
  between [clinician] and [patient] for the purpose of continuity
  of care, clinical decision support, and compliant billing,
  without fabricating, omitting, or misattributing any clinical
  information discussed during the encounter."

TOOL PALETTE:
  - encounter_initiate (Tier: always_allowed)
  - audio_capture (Tier: always_allowed)
  - medical_asr_transcribe (Tier: always_allowed)
  - clinical_nlp_extract (Tier: requires_confirmation)
  - clinical_note_generate (Tier: requires_human_review)
  - clinician_review_gate (Tier: human_in_the_loop)
  - ehr_submit (Tier: requires_confirmation)
  - billing_code_suggest (Tier: requires_confirmation)

BOUNDARY SPECIFICATIONS:
  - MUST NOT fabricate clinical findings not discussed
  - MUST NOT attribute family history to patient
  - MUST NOT omit safety-critical information (allergies, adverse reactions)
  - MUST NOT generate codes without supporting documentation
  - MUST NOT bypass clinician review step
  - MUST flag low-confidence sections for explicit review
  - MUST maintain provenance chain (audio -> text -> note -> approval)
```

---

## 11. Sources

### Microsoft/Nuance DAX Copilot
- [DAX Copilot Data Sheet (PDF)](https://www.nuance.com/asset/en_us/collateral/healthcare/data-sheet/ds-ambient-clinical-intelligence-en-us.pdf)
- [DAX Copilot 1.7 Technical User Guide (PDF)](https://csocontent.nuance.com/DAX%20Copilot/Resources/DAX-Copilot-technical-user-guide.pdf)
- [DAX Copilot FAQs (PDF)](https://csocontent.nuance.com/Hub/DAX%20Copilot/Misc%20Resources/DAX-Copilot-FAQs.pdf)
- [DAX Copilot Customizable Templates (PDF)](https://csocontent.nuance.com/Hub/DAX%20Copilot/Copilot_for_Epic/DCE-cust-templates.pdf)
- [Microsoft Dragon Copilot Overview](https://www.microsoft.com/en-us/health-solutions/clinical-workflow/dragon-copilot)
- [Dragon Copilot De-Identification Architecture (Microsoft Tech Community)](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/how-microsoft-dragon-copilot-uses-the-azure-health-data-services-de-identificati/4450165)
- [Dragon Copilot Privacy and Data Retention (Microsoft Learn)](https://learn.microsoft.com/en-us/industry/healthcare/dragon-copilot/about/privacy-old)
- [DAX Copilot for Epic Quick Start (Microsoft Support)](https://support.microsoft.com/en-us/topic/dragon-copilot-for-epic-quick-start-guide-cbf8312b-640e-4ad9-b9fc-4ba77bc665da)
- [DAX Copilot for Epic FAQs (Microsoft Support)](https://support.microsoft.com/en-us/topic/dax-copilot-for-epic-faqs-a1726f08-b193-471d-9d1a-e35c6ecad30b)
- [DAX Copilot Embedded in Epic (PR Newswire)](https://www.prnewswire.com/news-releases/nuance-announces-general-availability-of-dax-copilot-embedded-in-epic-302037590.html)
- [DAX Copilot Surgical Resident Study (Surgical Endoscopy, 2025)](https://link.springer.com/article/10.1007/s00464-025-12404-x)
- [DAX Cohort Study (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10990544/)
- [DAX Copilot Review 2026 (TryTwofold)](https://www.trytwofold.com/compare/dax-copilot-review)

### Abridge
- [Abridge Becomes Epic's First Pal (Press Release)](https://www.abridge.com/press-release/abridge-becomes-epics-first-pal-bringing-generative-ai-to-more-providers-and-patients)
- [Abridge Inside from Haiku to Hyperdrive (Press Release)](https://www.abridge.com/press-release/abridge-inside-emory)
- [Abridge Inside for Emergency Medicine (Press Release)](https://www.abridge.com/press-release/abridge-inside-for-emergency-medicine-announcement)
- [Abridge Inside for Inpatient and Orders (Press Release)](https://www.abridge.com/press-release/abridge-inside-for-inpatient-and-outpatient-orders)
- [Abridge Science of Confabulation Elimination](https://www.abridge.com/ai/science-confabulation-hallucination-elimination)
- [Abridge AI Evaluation Science](https://www.abridge.com/ai/science-ai-evaluation)
- [Abridge Revenue Cycle](https://www.abridge.com/platform/revenue-cycle)
- [Abridge AI Technology Overview](https://www.abridge.com/ai)
- [Abridge Business Breakdown (Contrary Research)](https://research.contrary.com/company/abridge)
- [Engineering Behind Healthcare LLMs with Abridge (Out-of-Pocket)](https://www.outofpocket.health/p/the-engineering-behind-healthcare-llms-with-abridge)
- [Abridge Series E / $300M (Press Release)](https://www.abridge.com/blog/series-e)

### Academic / Peer-Reviewed
- [Ambient AI Scribes and Coding Arms Race (npj Digital Medicine / PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12738533/)
- [Beyond Human Ears: Risks of AI Scribes (npj Digital Medicine / PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12460601/)
- [AI Scribes: Balancing Potential with Responsible Integration (JMIR / PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12316405/)
- [Ambient AI Scribes Randomized Trial (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12768499/)
- [Ambient AI Scribes NEJM Catalyst Review](https://catalyst.nejm.org/doi/full/10.1056/CAT.23.0404)
- [Ambient Documentation and Burnout (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12371510/)
- [Quality of Clinical Documentation with AI (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452835/)
- [Framework for Clinical Safety and Hallucination Rates (npj Digital Medicine)](https://www.nature.com/articles/s41746-025-01670-7)
- [AMA Journal of Ethics: Ambient Listening and EHR Documentation](https://journalofethics.ama-assn.org/article/how-should-we-think-about-ambient-listening-and-transcription-technologies-influences-ehr/2025-11)

### Technical References
- [NVIDIA Ambient Provider Blueprint (GitHub)](https://github.com/NVIDIA-AI-Blueprints/ambient-provider)
- [Epic FHIR Documentation](https://fhir.epic.com/Documentation)
- [Epic Technical Specifications](https://open.epic.com/TechnicalSpecifications)
- [Epic EHR Integration Technical Guide (Orbdoc)](https://orbdoc.com/learn/epic-integration-technical-guide)

---

*Research conducted 2026-02-16 by Karpathy (Systems Engineer), TELOS Research Team*
