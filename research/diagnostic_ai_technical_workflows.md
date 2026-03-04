# Diagnostic & Radiology AI Systems: Technical Workflows, Data Flows, and Integration Points

**Prepared by:** Gebru (Data Scientist), TELOS Research Team
**Date:** 2026-02-16
**Purpose:** Engineering-level reference for understanding how hospital-deployed diagnostic AI systems actually work in production -- data flows, tool calls, integration protocols, and governance gaps.

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## 1. Viz.ai (Stroke Detection & Care Coordination)

### 1.1 Imaging Modality & Trigger

**Primary modality:** CT Angiography (CTA) of the brain, acquired in the acute/emergency setting.

The system is triggered automatically. When a CTA is completed at the scanner, DICOM images are routed to Viz.ai without any manual clinician action. The system also has FDA-cleared algorithms for:
- **Non-contrast CT head** (ASPECTS scoring for early ischemic changes, ICH detection and quantification)
- **CT Perfusion** (perfusion maps)

The original De Novo clearance (DEN170073, February 2018) was specifically for CTA-based LVO (Large Vessel Occlusion) detection -- the "ContaCT" algorithm.

### 1.2 Technical Pipeline

```
Step 1: CT scanner completes CTA acquisition
         |
Step 2: DICOM images auto-forwarded via on-premise "Image Forwarding Software"
         | (configured to interact with hospital scanner, PACS, or local DICOM router)
         | (transmits DICOM files through TLS-encrypted channel)
         |
Step 3: Images arrive at cloud-hosted "Image Processing and Analysis Software"
         |
Step 4: Pre-processing pipeline:
         |  a) Inspect DICOM metadata to identify applicable CTA series
         |  b) Verify existence of contrast (reject non-contrast studies)
         |  c) 3D registration of the brain
         |  d) Crop 3D cuboid containing ICA-T, M1, and M2 regions
         |  e) Quality checks (reject incomplete, poor quality, or inappropriate studies)
         |
Step 5: CNN inference (deep learning convolutional neural network)
         |  - Trained on tens of thousands of labeled CT scans
         |  - Binary classification: suspected LVO vs. no suspected LVO
         |
Step 6: If LVO suspected --> push notification generated
         |  - Median time-to-notification: 5 min 45 sec (from scan completion)
         |
Step 7: Alert delivered to neurovascular specialist via:
         |  - Mobile push notification (Viz.ai iOS/Android app)
         |  - Text/SMS notification
         |  - Web-based viewer notification
         |  - HIPAA-compliant, immediate, to pre-designated multidisciplinary team
         |
Step 8: Specialist opens Viz app -> views compressed preview images
         |  - Mobile images are compressed, for informational use only
         |  - NOT intended for diagnostic use beyond notification
         |  - Specialist must view non-compressed images on diagnostic display
         |
Step 9: Secure team communication channel opens within Viz platform
         |  - Chat/messaging between neurointerventionalist, ED physician, stroke nurse
         |  - Image sharing within secure channel
         |  - Transfer coordination across hospital network
         |
Step 10: Clinical decision and documentation in EHR
```

### 1.3 AI Output Details

The ContaCT algorithm produces:
- **Binary classification:** Suspected LVO present / not present (notification-only, not a diagnostic)
- **Notification with image viewer link** containing preview of relevant images
- **No explicit probability score, heatmap, or precise vessel segment localization is described in FDA clearance** -- it is a triage/notification tool, not a diagnostic tool

Per FDA classification: "Identification of suspected findings is not for diagnostic use beyond notification." The device was classified as **Radiological Computer Aided Triage and Notification Software** (new regulatory category created with this De Novo).

Additional cleared algorithms in the Viz Neuro suite include:
- **Viz ICH:** Intracranial hemorrhage detection on non-contrast CT head
- **Viz ICH Plus:** Automated identification, labeling, and volumetric quantification of segmentable brain structures on NCCT
- **Viz ASPECTS:** Automated ASPECTS scoring for early ischemic changes

### 1.4 Alert Delivery Mechanism

- **Mobile push notification** to Viz.ai app (iOS/Android) on specialist's personal device
- Notification contains compressed preview images + link to full viewer
- **Secure, unified communication channel** for the entire care team opens automatically
- The alert goes to a **pre-designated, multidisciplinary care team** (not just one specialist)
- Web-based viewer available for desktop access

### 1.5 Specialist Actions Within Viz Platform

1. View compressed preview images on mobile
2. Open full diagnostic-quality images on web viewer / PACS
3. Communicate with care team via in-platform secure messaging
4. Coordinate patient transfer if needed (especially in hub-and-spoke stroke networks)
5. Initiate treatment planning (thrombectomy evaluation)
6. The platform provides a consistent real-time view across the network regardless of which PACS systems individual facilities use

### 1.6 PACS and EHR Integration

- **PACS:** Viz connects to any imaging device/PACS using the DICOM standard. The platform is PACS-independent, meaning it works across heterogeneous hospital networks with different PACS vendors.
- **EHR:** Tailored integration with hospital EHR for worklist embedding and documentation
- **Cloud architecture:** The Viz cloud is PACS-independent, scalable, and enables access to patient data from outside hospitals not directly connected to the health system
- **Standards:** NEMA PS 3.1-3.20 (2016) DICOM compliance referenced in FDA documentation

### 1.7 FDA Clearance Status

| Clearance | Type | Date | Device | Classification |
|-----------|------|------|--------|----------------|
| DEN170073 | De Novo (Class II) | Feb 2018 | ContaCT (LVO triage) | Radiological Computer Aided Triage and Notification Software |
| K193658 | 510(k) | Mar 2020 | Viz ICH (hemorrhage detection) | Class II |
| K223042 | 510(k) | 2022 | Viz ICH Plus (ICH quantification) | Class II |
| Multiple additional | 510(k) / De Novo | 2019-2025 | Cardio, Pulmonary, Aortic, etc. | Class II |

**Total portfolio:** More than **50 FDA-cleared AI algorithms** across neurovascular, cardiovascular, pulmonary, aortic, vascular, trauma, and oncology domains as of 2025. Deployed in over **1,700 hospitals**.

### 1.8 Published Sensitivity/Specificity

**FDA submission data (DEN170073):**
- Sensitivity: **96.32%** [95% CI: 92.68-98.84%]
- Specificity: **93.83%** [95% CI: 92.83-94.75%]
- Median time-to-notification: **5 min 45 sec**

**Real-world validation studies show variable performance:**

| Study Setting | Sensitivity | Specificity | PPV | NPV | N |
|---------------|-------------|-------------|-----|-----|---|
| FDA submission | 96.3% | 93.8% | -- | -- | -- |
| Academic center (PMC7872164) | 90.6% | 87.6% | 48.3% | 98.6% | -- |
| Large integrated network (PMC9847593) | 78.2% | 97.0% | 61.0% | 99.0% | 3,851 |
| Single-institution 9-month (Viz publications) | 82.0% | 94.0% | 77.0% | 95.0% | 650 |

**Key finding:** M2 occlusions are more frequently missed. Real-world sensitivity is notably lower than FDA submission data, ranging from 78-91% depending on institution and occlusion location. This is a significant governance gap -- the difference between cleared performance and deployed performance.

**Clinical impact:** Patients with stroke get to treatment approximately **66 minutes faster** when the AI alert system is used (Viz.ai data).

### 1.9 Data Governance: Cloud vs. On-Premises

| Component | Location | Data |
|-----------|----------|------|
| Image Forwarding Software | **On-premises** (hospital network) | DICOM images originate here |
| Image Processing & Analysis | **Cloud** (inference) | DICOM images transmitted via TLS |
| Mobile App | **Clinician device** | Compressed preview images; PHI cleared on hibernation/logout/close |
| Communication Platform | **Cloud** | Secure messaging, image sharing |

- **PHI handling:** PII/PHI only exists on the mobile device when the application is in use; cleared as soon as the app goes into hibernation, login timeout expires, or app is closed
- **Encryption:** Data encrypted at every point in transit; industry-standard cryptographic algorithms
- **Certifications:** SOC 2 Type II (4th consecutive year), ISO-27001:2022 (7 ISO certificates), HIPAA compliant
- **Audit trail:** Not publicly detailed beyond SOC 2 / HIPAA compliance attestation
- **Model updates:** Not publicly documented (cloud-based architecture implies server-side model updates possible without hospital-side changes -- a governance concern)

---

## 2. Aidoc (Radiology AI Triage)

### 2.1 Conditions Detected

**Individual FDA-cleared algorithms (pre-foundation model):**

| # | Condition | Imaging Modality | FDA Status |
|---|-----------|-----------------|------------|
| 1 | Intracranial Hemorrhage (ICH) | Non-contrast head CT | 510(k) cleared |
| 2 | Large Vessel Occlusion (LVO) | Head CTA | 510(k) cleared |
| 3 | Acute C-Spine Fractures | Cervical spine CT | 510(k) cleared |
| 4 | Pulmonary Embolism (PE) | CT Pulmonary Angiography | 510(k) cleared |
| 5 | Incidental Pulmonary Embolism | CT (any) | 510(k) cleared (industry first) |
| 6 | Intra-Abdominal Free Gas | Abdomen CT | 510(k) cleared |
| 7 | Rib Fractures | Chest CT | 510(k) cleared |
| 8 | Pneumothorax | Chest X-ray | 510(k) cleared |
| 9 | Brain Aneurysm | Head CTA | 510(k) cleared |
| 10 | Thoracic/Lumbar Spine Fractures | Spine CT | 510(k) cleared |
| 11 | Aortic Dissection | CT | 510(k) cleared |
| 12 | All Vessel Occlusion | CTA | 510(k) cleared |

**CARE Foundation Model -- Comprehensive Abdomen CT Triage (January 21, 2026):**

11 **newly** cleared + 3 **previously** cleared indications in a single workflow:

*New (11):*
- Appendicitis
- Acute diverticulitis
- Abdominal-pelvic abscess
- Small bowel obstruction
- Large bowel obstruction
- Obstructive kidney stone
- Intestinal ischemia and/or pneumatosis
- Kidney injury
- Liver injury
- Spleen injury
- Pelvic fracture

*Previously cleared (3):*
- Abdominal aortic measurement
- Aortic dissection
- Intra-abdominal free air

This was the **first comprehensive foundation model AI** to receive FDA clearance in healthcare. The model can triage **14 critical findings in a single abdominal CT scan**.

### 2.2 Technical Architecture

```
Step 1: CT/X-ray scanner completes study
         |
Step 2: DICOM images routed to Aidoc AI Orchestrator
         | (on-premise VM in customer's environment)
         | (receives studies via DICOM C-STORE SCP or query/retrieve)
         |
Step 3: Orchestrator performs intelligent routing:
         |  a) Identifies eligible scan type
         |  b) Recognizes anatomy present on scan
         |  c) Determines which algorithms to apply
         |  d) DE-IDENTIFIES the study (white-list process)
         |     - Only copies specific DICOM tags from a white-list
         |     - All non-white-listed tags are DELETED
         |     - Deep learning algorithm detects and removes burned-in text in pixel data
         |     - Generates new SeriesInstanceUID + StudyInstanceUID (AidocSeriesUID / AidocStudyUID)
         |  e) Uploads de-identified study to cloud Analysis Service
         |
Step 4: Cloud-based Aidoc Analysis Service performs inference
         | (distributed, scalable cloud environment on AWS/Azure/GCP)
         | (AWS: EC2 P3 instances for ML training + inference)
         | (AWS: S3 for anonymized medical imagery storage)
         | (AWS: RDS for image metadata)
         |
Step 5: Results returned to on-premise Orchestrator
         |  a) PHI RE-ATTACHED using the unique identifier generated at de-identification
         |  b) Results pushed to:
         |     - Aidoc Widget (in-PACS overlay)
         |     - Native radiology worklist
         |     - Reporting system
         |     - PACS
         |
Step 6: Radiologist sees flagged study in worklist with priority elevation
```

**Critical data flow design:** PHI NEVER leaves the customer's environment. Only de-identified (pseudonymized) data goes to the cloud. PHI is re-attached on the return path using the unique identifier.

### 2.3 Worklist Prioritization

Instead of the traditional **first-in, first-out (FIFO)** radiology worklist, Aidoc re-prioritizes based on AI findings:
- Studies with detected acute/critical findings are elevated to the top of the worklist
- Priority is assigned as the data enters the system and is analyzed
- Radiologists see urgent cases first, regardless of when the study was acquired
- Works with existing Radiology Information Systems (RIS)

### 2.4 What the Radiologist Sees

The **Aidoc Widget** is the front-end interface:
- Installed within the radiologist's existing PACS/reporting environment
- Consolidates results from ALL algorithms into a **single unified interface**
- Displays flagged findings with visual indicators
- Provides smart filters for urgency-based prioritization
- Functions as an "always-on safety net" -- every study is analyzed, not just selected ones

**The radiologist does NOT need to leave their normal PACS workflow** -- the Widget integrates directly.

### 2.5 Handling Multiple Findings on a Single Study

With the CARE foundation model, **multiple findings on a single study are flagged simultaneously**. The system analyzes a single CT and can detect up to 14 different conditions in one pass. This is architecturally different from the previous generation of "point solutions" which ran separate algorithms independently and could generate separate, disconnected alerts.

The foundation model approach provides:
- ~10x reduction in false alerts compared to prior generation point solutions
- Unified signal rather than disconnected per-condition alerts
- Single workflow for all findings

### 2.6 Integration with PACS and EHR

**Protocols:** DICOM, HL7v2, FHIR
- **PACS:** Direct integration via DICOM; Widget overlays within PACS viewer
- **EHR:** HL7/FHIR-based integration for clinical documentation
- **RIS:** Worklist modification for priority reordering
- **VNA:** Vendor Neutral Archive connectivity
- **Scheduling/Reporting systems:** Full integration

The on-premise Orchestrator handles all integration points, routing, and protocol translation.

### 2.7 Published Performance Metrics

**CARE Foundation Model (FDA pivotal study, January 2026):**
- Mean sensitivity: **97%** (up to 98.5% in certain settings)
- Mean specificity: **98%** (up to 99.7%)
- **~10x reduction** in false alerts vs. prior generation point solutions

**Platform scale:** Over **100 million patient cases analyzed** to date across 300+ customer sites.

### 2.8 Data Governance Details

| Component | Location | Data Handled |
|-----------|----------|--------------|
| AI Orchestrator | **On-premises** (customer VM) | Full PHI -- receives, de-identifies, re-identifies |
| Analysis Service | **Cloud** (AWS/Azure/GCP) | De-identified studies ONLY |
| Widget | **On-premises** (PACS workstation) | Re-identified results with PHI |
| Metadata DB | **Cloud** (AWS RDS) | De-identified image metadata only |
| Image Storage | **Cloud** (AWS S3) | Anonymized imagery for analysis |

**De-identification process:**
- White-list approach: only explicitly listed DICOM tags are copied; all others deleted
- Deep learning-based burned-in text detection and removal from pixel data
- New unique identifiers generated for each series and study
- AES-256 encryption at rest for all uploaded data
- Two-way authentication between Analysis Service and Orchestrator
- Firewall outbound rules limited to Aidoc services only

**Security certifications:** SOC 2 Type 2, HIPAA/HITECH compliant, role-based access control

**Model updates:** The aiOS platform includes "continuous performance monitoring" and "built-in governance" -- implies server-side model updates managed by Aidoc. The CARE foundation model is deployed through aiOS, suggesting centralized model management. Specific update cadence and mechanism (OTA vs. managed rollout) not publicly documented.

**Logging:** Logging and analytics data are de-identified before being sent to the Orchestrator's database.

---

## 3. Paige.ai (Digital Pathology)

### 3.1 Slide Scanning to AI Analysis Pipeline

```
Step 1: Tissue preparation (standard pathology workflow)
         | Prostate needle biopsy -> FFPE (formalin-fixed paraffin-embedded)
         | H&E (hematoxylin & eosin) staining
         |
Step 2: Glass slide scanning on FDA-cleared scanner
         | Compatible scanners:
         |   - Philips IntelliSite Pathology Solution Ultra Fast Scanner (UFS)
         |   - Leica Aperio GT 450 DX (SVS + DICOM formats)
         |   - Hamamatsu NanoZoomer S360MD (NDPI format)
         | Output: Whole Slide Image (WSI), 1-2 GB per slide
         |
Step 3: WSI ingested into Paige Platform
         | Platform connects to:
         |   - Whole slide scanners
         |   - Laboratory Information System (LIS)
         |   - Other clinical information systems
         | WSI + case metadata (accession ID, specimen type, etc.) are married together
         |
Step 4: AI inference on Paige cloud (Microsoft Azure)
         | Deep learning model (weakly supervised CNN approach)
         | Processes entire whole slide image
         |
Step 5: Results displayed in FullFocus viewer
         | If cancer suspected: provides (X,Y) coordinates of the single location
         | with the highest likelihood of having cancer
         |
Step 6: Pathologist reviews AI-highlighted area in FullFocus
         | Makes final diagnostic determination
         | Actions communicated back to LIS
```

### 3.2 What the Pathologist Sees in Their Viewer

**FullFocus** is Paige's FDA-cleared whole-slide image viewer (510(k) cleared):
- Displays the entire digitized slide at diagnostic quality
- When Paige Prostate Detect identifies suspicious tissue, it provides **coordinates (X,Y)** pointing to the **single location on the image with the highest likelihood of having cancer**
- The pathologist navigates to that location for further review
- Includes diagnostic and collaboration tools
- Supports remote consultation and expert sharing
- "Intuitive AI visualizations" integrated directly into the viewing workflow

### 3.3 Specific Diagnoses and Tools

**The Paige Prostate Suite:**

| Tool | Function | FDA Status |
|------|----------|------------|
| **Paige Prostate Detect** | Detects foci suspicious for cancer on prostate needle biopsy WSI | **De Novo approved** (DEN200080, Sept 2021) -- FIRST AI pathology product FDA-authorized |
| **Paige Prostate Grade & Quantify** | Grades suspicious areas using Gleason scoring; quantifies overall tumor percentage and length | CE-IVD (EU/UK); not yet FDA-cleared |
| **Paige Prostate Perineural Invasion** | Identifies perineural invasion in prostate tissue | CE-IVD; not yet FDA-cleared |
| **Paige Prostate Biomarker Panel** | Detects biomarkers relevant to prostate cancer (e.g., TP53 mutations) | Research/CE-IVD |

**Additional FDA clearance (Jan 2025):** FullFocus viewer cleared for use with Leica Aperio GT 450 DX and Hamamatsu NanoZoomer S360MD scanners (expanding scanner compatibility).

**Note:** Tempus AI acquired Paige's digital pathology business for $81.25 million in August 2025, giving Tempus access to nearly 7 million digitized pathology slide images and associated clinical/molecular data.

### 3.4 FDA Clearance Details

**Historic significance:** Paige Prostate Detect (DEN200080) was the **first-ever FDA-authorized AI product in pathology**, receiving De Novo classification on September 21, 2021.

- **Regulatory pathway:** De Novo (Class II designation with special controls)
- **Generic name established:** "Software algorithm device to assist users in digital pathology"
- **Intended use:** Assist pathologists in detecting foci suspicious for cancer during review of scanned WSI from prostate needle biopsies (H&E stained FFPE tissue)
- **Requires:** Philips Ultra Fast Scanner + Paige FullFocus viewer (initial clearance); later expanded to Leica and Hamamatsu scanners
- **Classification:** Class II medical device

### 3.5 LIS Integration

Paige provides **API-based integration** with Laboratory Information Systems:
- Connection via Application Programming Interfaces (APIs)
- **Inbound data flow:** WSI + case metadata (accession ID, specimen type, etc.) sent into Paige Platform
- **Automatic case assembly:** Platform automatically marries image data with metadata to create a complete, interpretable case
- **Outbound data flow:** Actions taken on cases (diagnosis, annotations) communicated back into the LIS
- Enables end-to-end digital pathology workflow without manual case assembly

**Real-world implementation finding:** In a non-academic, non-commercial pathology laboratory, implementing AI-assisted prostate biopsy workflow reduced turnaround time by approximately **9 hours** and decreased ancillary immunohistochemical testing by roughly **one-third** after an adaptation period.

### 3.6 Published Accuracy Data

**FDA pivotal study (DEN200080):**
- Pathologists using Paige Prostate increased sensitivity from **89.5% to 96.8%** (7.3 percentage point improvement)
- **70% reduction in false negatives**
- **24% reduction in false positives**
- Dataset: 610 de-identified prostate needle biopsy WSI from 218 institutions worldwide

**Standalone algorithm performance:**
- Sensitivity: **97.7%**
- Specificity: **99.3%**
- PPV: **97.9%**
- NPV: **99.2%**

**Small tumor detection improvement:**
- Average sensitivity for tumors under 0.6mm increased from **46% to 83%** with Paige Prostate Alpha
- Greatest gains in the smallest, hardest-to-detect tumors

**Non-specialist pathologists:** Achieved sensitivity levels similar to specialist pathologists when using Paige Prostate Detect -- a key equity finding.

### 3.7 Data Governance

| Component | Location | Data |
|-----------|----------|------|
| Slide Scanner | **On-premises** (pathology lab) | Glass slide to WSI conversion |
| Paige Platform | **Cloud** (Microsoft Azure) | WSI storage, AI inference, case management |
| FullFocus Viewer | **Cloud-served** (web-based) | WSI viewing, AI results display |
| LIS Integration | **On-premises** (lab network) | Case metadata, diagnostic results |

- **Cloud provider:** Microsoft Azure (Paige's stated preference for cloud over on-premise)
- **Data scale:** Single slide = 1-2 GB; large medical centers generate up to 1.5 million slides/year
- **Storage:** Various levels of cloud storage options offered through the Platform
- **Model versioning:** Uses **lakeFS** for data lake versioning to enforce reproducibility of ML experiments in compliance with FDA regulations
- **Training data pipeline:** ~2,000-3,000 new images/day added to training data (~2-3 TB/day); ~200 dbt tables containing enhanced training datasets; 200+ ML models
- **Regulatory compliance:** lakeFS + dbt pipeline ensures traceability and reproducibility required by FDA
- **Audit trail:** Not explicitly detailed in public documentation beyond FDA compliance requirements

---

## 4. Clinical Decision Support AI (Sepsis Prediction)

### 4.1 Epic Sepsis Model (ESM)

**Architecture:**

| Property | ESM v1 | ESM v2 |
|----------|--------|--------|
| **Model type** | Penalized logistic regression | Gradient boosted tree |
| **Variables** | ~50 | Different but overlapping set |
| **Training data** | 405,000 encounters, 3 health systems, 2013-2015 | Can be localized to individual hospital data |
| **Outcome definition** | Clinical intervention indicative of sepsis | Sepsis-3 criteria |
| **Deployment** | Hundreds of hospitals | Recommended replacement for v1 |

**Data inputs (ESM v1 -- 50 variables across these categories):**
- **Demographics:** Age, sex
- **Vital signs:** Temperature, heart rate, respiratory rate, blood pressure, SpO2
- **Laboratory values:** WBC, lactate, creatinine, bilirubin, platelets, blood cultures
- **Medications:** Antibiotic orders, vasopressor use
- **Comorbidities:** Charlson comorbidity index components
- **Procedural data:** Recent procedures, surgical status
- **Radiology data:** Relevant imaging orders

**Alert workflow:**

```
Step 1: ESM runs continuously in background within Epic EHR
         | Monitors all active inpatient encounters
         | Recalculates score on any data change (new vital, lab result, order)
         |
Step 2: Score exceeds threshold (>=5 for v1; >=6 to display BPA)
         | Threshold determined by ROC analysis during implementation
         |
Step 3: Best Practice Advisory (BPA) fires
         | Interruptive alert displayed to physician AND nurse
         | Appears on chart open (nurse-facing) or within provider workflow
         | Embedded in primary workspace -- not a separate system
         |
Step 4: Provider evaluates alert
         | Reviews score, contributing factors
         | Decides whether to initiate sepsis workup
         |
Step 5: If sepsis suspected -> Sepsis bundle initiated:
         |  3-hour bundle:
         |    a) Blood cultures before antibiotics
         |    b) Serum lactate level
         |    c) Broad-spectrum antibiotics
         |  6-hour bundle:
         |    a) IV bolus 30 mL/kg crystalloid (if hypotension or lactate >=4)
         |    b) Vasopressors if fluid-refractory hypotension
         |    c) Repeat lactate if initial >2 mmol/L (reflex order in EHR)
         |
Step 6: Documentation in EHR
         | Sepsis alert acknowledgment documented
         | Bundle compliance tracked automatically
```

**Critical performance concerns (published validation):**
- ESM v1 sensitivity: **14.7%** (at threshold of 6, within 6-hour window)
- ESM v1 specificity: **95.3%**
- ESM v1 PPV: **7.6%**
- ESM v1 NPV: **97.7%**
- Median lead time: **0 minutes** (alerted at or after sepsis onset in half of cases)
- AUROC: **0.63** in external validation (JAMA Internal Medicine 2021)

**This is a major documented failure of a widely-deployed clinical AI system.** The ESM v1 missed the vast majority of actual sepsis cases while generating many false positives. This motivated Epic's overhaul to v2.

**ESM v2 improvements:**
- Gradient boosted tree model (more expressive than logistic regression)
- Can be **localized** (trained on a hospital's own data before deployment)
- Changed sepsis definition to Sepsis-3 (more commonly accepted standard)
- Reduced reliance on clinician orders (which caused label leakage in v1)
- Significant reduction in false positive alert rates: non-sepsis alerts dropped from 3.3% to 1.2%; sepsis alerts from 26.1% to 6.8%
- Retains stronger discrimination even when restricted to pre-recognition period

### 4.2 COMPOSER (UC San Diego)

**Architecture:**

COMPOSER (COnformal Multidimensional Prediction Of SEpsis Risk) is fundamentally different from the Epic Sepsis Model in both architecture and deployment.

```
Step 1: Real-time data extraction
         | AWS-hosted cloud analytics platform
         | Continuous ADT (Admit, Discharge, Transfer) HL7v2 message stream from hospital integration engine
         | FHIR API extraction at HOURLY resolution with OAuth 2.0 authentication
         |
Step 2: Feature extraction (~150 variables)
         | - Demographics: age, sex
         | - Vital signs (real-time)
         | - Laboratory results (real-time)
         | - Comorbidities
         | - Medications (concomitant)
         | - Clinical notes (COMPOSER-LLM version)
         |
Step 3: Feed-forward neural network inference
         | Outputs: sepsis risk score for onset within NEXT 4 HOURS
         | Uses conformal prediction to REJECT out-of-distribution samples
         |   (data entry errors, unfamiliar patient presentations)
         |
Step 4: Risk score + top contributing features written to EHR
         | Via HL7v2 outbound message to a flowsheet within the EHR
         |
Step 5: Nurse-facing Best Practice Advisory (BPA) triggered on chart open
         | Alert states: patient at risk of developing severe sepsis
         | Shows risk score and top features driving the recommendation
         |
Step 6: Clinical response and bundle initiation
```

**Key architectural differences from ESM:**

| Feature | Epic Sepsis Model | COMPOSER |
|---------|-------------------|----------|
| **Model type** | Logistic regression (v1) / GBT (v2) | Feed-forward neural network |
| **Hosting** | Within Epic EHR (on-premise) | AWS cloud + FHIR/HL7 integration |
| **Data access** | Internal Epic data model | FHIR API + HL7v2 message streams |
| **Prediction window** | Current risk (often concurrent with onset) | 4 hours ahead |
| **Uncertainty handling** | None | Conformal prediction rejects out-of-distribution |
| **Localization** | v2 supports local training | Designed for site-specific deployment |
| **Update mechanism** | Epic release cycle | Independent development cycle |
| **Performance (AUROC)** | 0.63 (ESM v1 external) | 0.938-0.945 (in ED settings) |

**Clinical impact (prospective study, npj Digital Medicine 2024):**
- Deployed in two UC San Diego Emergency Departments
- Demonstrated improvement in quality of care metrics and survival
- Model achieves AUROC of **0.938-0.945**

**COMPOSER-LLM (2025 advancement):**
- Integrates large language model with base COMPOSER
- For high-uncertainty predictions, the LLM extracts additional clinical context from notes to assess sepsis-mimics
- Sensitivity: **72.1%**, PPV: **52.9%**, F-1: **61.0%**
- False alarm rate: **0.0087 per patient hour** (very low)
- Outperforms standalone COMPOSER model

### 4.3 SepsisLab (Ohio State University Wexner Medical Center)

**Architecture -- four distinct components:**

```
Component 1: IMPUTATION MODEL
              | Estimates distribution (mean + std dev) of missing lab values
              | Std dev = uncertainty of imputed results
              |
Component 2: SEPSIS PREDICTION MODEL
              | Time-aware model predicts sepsis onset within coming hours
              | Generates BOTH risk score AND uncertainty simultaneously
              |
Component 3: UNCERTAINTY QUANTIFICATION
              | Propagated uncertainty = variance of prediction output
              | Dominant at beginning of hospital admissions (when data is sparse)
              | Uses uncertainty propagation methods from imputation through prediction
              |
Component 4: ACTIVE SENSING ALGORITHM
              | Recommends which LAB TESTS to order to maximally reduce uncertainty
              | Identifies the most informative missing data points
              | Tells clinicians: "ordering THIS specific lab test will most reduce
              |   the uncertainty in the sepsis prediction"
```

**What makes SepsisLab unique:** The **active sensing** component. Rather than passively consuming whatever data is available, SepsisLab actively recommends which observations (lab tests) the clinician should order to maximize prediction confidence. This is a fundamentally different paradigm -- the AI is not just predicting, it is **requesting specific data collection actions** from clinicians.

**Technical details:**
- Produces new prediction every hour after new patient data is added
- Predicts sepsis risk within 4-hour window
- Visual interface shows clinicians how specific missing information would affect the risk prediction
- Validated on MIMIC-III, AmsterdamUMCdb (public), and OSUWMC data (proprietary)

**Performance:**
- Adding just **8% of the recommended data** improved sepsis prediction accuracy by **11%**
- Active sensing algorithm outperforms state-of-the-art active sensing methods
- Open source: https://github.com/yinchangchang/sepsislab

### 4.4 EHR Data Feeding Real-Time Sepsis Models

All sepsis prediction systems consume some subset of these real-time EHR data streams:

| Data Category | Specific Elements | Update Frequency |
|---------------|-------------------|------------------|
| **Vital signs** | HR, BP (systolic/diastolic/MAP), RR, SpO2, temperature | Every 1-4 hours (nursing assessment) or continuous (telemetry) |
| **Laboratory values** | WBC, lactate, creatinine, bilirubin, platelets, BUN, glucose, blood gas, procalcitonin, blood cultures | As ordered (varies) |
| **Medications** | Antibiotics, vasopressors, fluids, antipyretics | Real-time on order/administration |
| **Nursing assessments** | Mental status (GCS), skin assessment, urine output, fluid balance | Every 2-4 hours |
| **Demographics** | Age, sex, admission source, surgical status | Static (on admission) |
| **Comorbidities** | Problem list, Charlson comorbidity index, immunosuppression status | Semi-static |
| **ADT events** | Admit, discharge, transfer messages | Real-time |
| **Clinical notes** | Physician/nursing narrative (COMPOSER-LLM only) | As documented |

### 4.5 Alert-to-Action Workflow

When a sepsis prediction triggers an alert:

```
Step 1: BPA/Alert fires in EHR (interruptive or passive, configurable)
         |
Step 2: Primary RN assesses patient (vital signs, clinical status)
         |
Step 3: If clinical concern confirmed:
         |  a) RN contacts primary physician/APP
         |  b) Some institutions activate SERT (Sepsis Emergency Response Team)
         |     - SERT RN, primary RN, pharmacist, physician huddle at bedside
         |
Step 4: Physician evaluates clinical scenario
         |  a) Reviews contributing factors shown by AI
         |  b) Determines if sepsis workup is indicated
         |
Step 5: If sepsis workup initiated (within 1 hour target):
         |  a) Blood cultures drawn (before antibiotics)
         |  b) Serum lactate ordered
         |  c) Broad-spectrum antibiotics administered
         |  d) IV fluid bolus (30 mL/kg crystalloid if hypotensive or lactate >=4)
         |
Step 6: Ongoing monitoring
         |  a) Reflex lactate if initial >2 mmol/L (EHR auto-order)
         |  b) Vasopressors if fluid-refractory hypotension
         |  c) Source control assessment
         |  d) De-escalation when culture results available
         |
Step 7: Bundle compliance documented in EHR
         | Automated timestamp tracking for each bundle element
```

**Known challenge:** Alert fatigue. High false positive rates (especially ESM v1) lead clinicians to ignore alerts. Studies show nurses and physicians develop "alarm fatigue" -- the single greatest barrier to effective sepsis AI deployment.

---

## 5. The Action Chain for Diagnostic AI

### 5.1 Radiology AI (Viz.ai / Aidoc Pattern)

```
Step 1: IMAGE ACQUISITION
        CT/MRI/X-ray scanner completes study
        Output: DICOM image series with metadata (patient demographics,
        study type, body part, contrast, scanner model, acquisition parameters)
        |
Step 2: DICOM ROUTING TO AI
        Automatic routing via:
        - Hospital DICOM router (configured routing rules by study type)
        - Direct scanner -> AI system DICOM send
        - PACS auto-forward rules
        Protocol: DICOM C-STORE, DICOM TLS for encryption
        Standards: NEMA PS 3.1-3.20
        |
Step 3: AI PRE-PROCESSING
        Viz.ai: CTA series identification via DICOM metadata, contrast verification,
                3D brain registration, ICA-T/M1/M2 region cropping
        Aidoc:  Study eligibility check, anatomy recognition, algorithm selection,
                PHI de-identification (white-list), upload to cloud
        |
Step 4: AI INFERENCE
        Viz.ai: Cloud-hosted CNN on CTA images -> binary LVO classification
        Aidoc:  Cloud-hosted models (EC2 P3) -> multi-condition detection
        Latency: 1-6 minutes from acquisition to result
        |
Step 5: RESULT GENERATION
        Viz.ai: Notification flag (suspected LVO) + image viewer link
        Aidoc:  Priority flags per condition + worklist re-ordering
        Format: DICOM Secondary Capture (SC), DICOM Structured Report (SR),
                proprietary notification via platform APIs
        IHE Profiles: AI Results (AIR), AI Workflow for Imaging (AIW-I),
                      Prioritization of Worklists for Reporting
        |
Step 6: ALERT / NOTIFICATION TO CLINICIAN
        Viz.ai: Mobile push notification to specialist team + secure chat
        Aidoc:  Worklist priority elevation + Widget flag in PACS
        Both:   EHR notification integration available
        Standards: HL7 FHIR Observation resources for non-imaging systems,
                   HL7 FHIRcast for synchronizing clinical applications
        |
Step 7: CLINICIAN REVIEW IN VIEWER
        Viz.ai: Compressed preview on mobile -> full PACS review on diagnostic display
        Aidoc:  Widget within existing PACS workflow, no context switch needed
        Paige:  FullFocus viewer with (X,Y) coordinates highlighting suspicious areas
        |
Step 8: CLINICAL DECISION
        Radiologist/specialist determines:
        - True positive -> escalate care
        - False positive -> dismiss, continue standard workflow
        - Correlate with clinical context, prior studies, patient history
        |
Step 9: DOCUMENTATION IN EHR
        - Radiology report dictated/finalized
        - AI-assisted finding acknowledged or dismissed
        - Critical result communication documented
        - Downstream orders placed (e.g., neuro consult, thrombectomy evaluation,
          anticoagulation for PE)
        |
Step 10: DOWNSTREAM ORDERS / ACTIONS
         - Interventional procedures scheduled (thrombectomy, embolectomy)
         - Specialist consultations
         - Patient transfers (spoke -> hub in stroke networks)
         - Follow-up imaging orders
         - Treatment protocols initiated
```

### 5.2 Digital Pathology AI (Paige Pattern)

```
Step 1: TISSUE ACQUISITION & PREPARATION
        Biopsy performed -> tissue fixed (FFPE) -> sectioned -> stained (H&E)
        -> glass slide prepared (standard histopathology workflow)
        |
Step 2: SLIDE SCANNING
        Glass slide placed on FDA-cleared scanner
        (Philips UFS / Leica GT 450 DX / Hamamatsu S360MD)
        Output: Whole Slide Image (1-2 GB per slide)
        Format: SVS, NDPI, or DICOM
        |
Step 3: CASE ASSEMBLY IN AI PLATFORM
        WSI + LIS metadata (accession ID, specimen type) ingested via API
        Platform auto-assembles complete case record
        |
Step 4: AI INFERENCE (cloud -- Azure)
        Weakly supervised CNN processes full WSI
        Output: Suspicion flag + (X,Y) coordinates of highest-likelihood area
        |
Step 5: RESULT DISPLAY
        Pathologist opens case in FullFocus viewer
        AI overlay shows suspicious area coordinates
        |
Step 6: PATHOLOGIST REVIEW
        Navigates to AI-flagged location
        Reviews tissue at diagnostic magnification
        Compares with surrounding tissue, clinical context
        |
Step 7: DIAGNOSTIC DECISION
        Pathologist renders final diagnosis
        AI assists but does not replace pathologist judgment
        |
Step 8: DOCUMENTATION IN LIS
        Diagnosis entered into Laboratory Information System
        Results communicated to ordering clinician
        |
Step 9: DOWNSTREAM CLINICAL ACTIONS
        Treatment planning (surgery, radiation, active surveillance)
        Molecular testing ordered if indicated
        Tumor board discussion
```

### 5.3 Sepsis Prediction AI (COMPOSER Pattern)

```
Step 1: CONTINUOUS DATA ACQUISITION
        Vital signs, lab values, medications, nursing assessments
        Flow via HL7v2 ADT messages + FHIR API at hourly resolution
        |
Step 2: DATA ROUTING TO AI
        AWS-hosted platform receives continuous HL7v2 stream
        FHIR API extracts latest data with OAuth 2.0 authentication
        |
Step 3: FEATURE EXTRACTION
        ~150 variables extracted from EHR data
        Missing values handled (SepsisLab: imputation + uncertainty quantification)
        |
Step 4: AI INFERENCE
        Neural network / GBT / LLM produces risk score
        Prediction horizon: 4 hours (COMPOSER) / variable (ESM)
        Conformal prediction rejects out-of-distribution inputs
        |
Step 5: RESULT WRITTEN TO EHR
        Risk score + top contributing features written to EHR flowsheet
        Via HL7v2 outbound message
        |
Step 6: BPA/ALERT FIRES
        Nurse-facing Best Practice Advisory on chart open
        Configured threshold (e.g., score >= 6)
        |
Step 7: CLINICAL ASSESSMENT
        Nurse evaluates patient
        Physician consulted if concern warranted
        |
Step 8: BUNDLE INITIATION (if sepsis suspected)
        Blood cultures, lactate, antibiotics within 1 hour
        IV fluids as indicated
        |
Step 9: DOCUMENTATION
        Alert response documented
        Bundle compliance tracked
        |
Step 10: ONGOING MONITORING
         Repeat predictions, lactate trending, antibiotic de-escalation
```

---

## 6. Data Governance Concerns

### 6.1 Summary Matrix

| System | PHI Accessed | Inference Location | Data Retained by Vendor | Model Update Mechanism | Audit Trail |
|--------|-------------|-------------------|------------------------|----------------------|-------------|
| **Viz.ai** | Full DICOM images (including patient demographics in DICOM headers) | **Cloud** (images transmitted via TLS) | Not publicly disclosed; PHI cleared from mobile on close | Cloud-based (server-side updates possible without hospital action) | SOC 2 Type II + HIPAA attestation; specific AI decision logging not detailed |
| **Aidoc** | Full DICOM studies on-premise; **only de-identified data** goes to cloud | **Cloud** (AWS EC2 P3) for inference; on-premise for PHI handling | De-identified images + metadata in cloud (S3/RDS); anonymized | aiOS "continuous performance monitoring"; centralized via aiOS platform | Role-based access; de-identification logging; analytics data de-identified |
| **Paige.ai** | Whole slide images (patient tissue), LIS case metadata | **Cloud** (Microsoft Azure) | WSI stored in Azure cloud (petabytes scale); training data grows ~2-3 TB/day | lakeFS + dbt pipeline for versioned, FDA-compliant model development; 200+ models | lakeFS versioning for reproducibility; FDA compliance-driven audit trail |
| **Epic Sepsis Model** | Comprehensive EHR data (vitals, labs, meds, demographics, comorbidities) | **On-premises** (within Epic EHR) | Epic retains model parameters; hospital data stays on-premise | Epic software release cycle; v2 supports local retraining | Within Epic audit trail; BPA acknowledgment logging |
| **COMPOSER** | EHR data via FHIR + HL7v2 (vitals, labs, meds, notes, demographics) | **Cloud** (AWS) | Cloud platform retains extracted feature data; unclear on retention | Independent research development cycle; COMPOSER-LLM adds capability | FHIR API OAuth 2.0 access logging; EHR flowsheet documentation |
| **SepsisLab** | EHR data (vitals, labs, nursing assessments) | **Research/on-premise** | Research data retained per IRB protocols | Research-driven updates (open source) | Research audit per institutional IRB |

### 6.2 Critical Governance Gaps Identified

**1. Cloud PHI Transmission (Viz.ai):**
Viz.ai transmits full DICOM images (including PHI in headers) to cloud for inference. While encrypted in transit and the platform is HIPAA/SOC 2 compliant, this represents a fundamentally different data governance posture than Aidoc's approach of de-identifying before cloud transmission. The DICOM header alone contains: patient name, DOB, MRN, accession number, referring physician, study date/time -- all 18 HIPAA identifiers may be present.

**2. Model Update Opacity:**
None of these systems provide public documentation on:
- How frequently models are updated
- What validation is performed before deployment of updated models
- Whether hospitals are notified of model updates
- Whether hospitals can opt out of or delay model updates
- Whether post-update performance is monitored per-site

This is a significant governance blind spot. A model update could change the sensitivity/specificity profile that was the basis for the hospital's adoption decision, without the hospital's knowledge.

**3. Real-World vs. Cleared Performance (Viz.ai):**
The FDA-cleared sensitivity of 96.3% vs. real-world sensitivity as low as 78.2% represents an 18-percentage-point gap. There is no mandated mechanism for post-market performance surveillance reporting by site. Hospitals may not know their local sensitivity differs from the cleared specification.

**4. Vendor Data Retention:**
- Aidoc retains de-identified images in AWS S3 -- for how long? Under what data deletion/retention policy?
- Paige.ai stores petabytes of WSI data on Azure and adds ~2-3 TB/day to training sets -- is hospital tissue data used for model training without explicit per-case consent?
- Viz.ai's data retention policy is not publicly documented

**5. Epic Sepsis Model Performance Crisis:**
The ESM v1 was deployed at hundreds of hospitals with a sensitivity of 14.7% -- meaning it missed 85.3% of actual sepsis cases. This was a proprietary, non-auditable model. The failure was only documented through independent external validation studies (JAMA Internal Medicine 2021). This is the canonical example of why clinical AI governance frameworks like TELOS are needed.

**6. Absence of Standardized AI Decision Audit Trails:**
While IHE profiles (AI Results, AI Workflow for Imaging) define standards for encoding AI results, there is no standardized, mandated audit trail format across vendors for:
- What the AI recommended
- What the clinician decided
- Whether the AI was correct (outcome tracking)
- Time from AI alert to clinical action
- Whether the AI result influenced the clinical decision

**7. Continuous Learning / Data Feedback Loops:**
The Pre-determined Change Control Plan (PCCP) pathway (FDA guidance) allows some AI/ML devices to be updated without new 510(k) submissions -- but the specifics of which systems use PCCP, how updates are validated, and what hospitals are told about changes remain opaque.

---

## 7. Interoperability Standards Reference

| Standard | Role in Diagnostic AI |
|----------|----------------------|
| **DICOM** | Image format, metadata, routing (C-STORE, C-FIND, C-MOVE) |
| **DICOM SR** | Structured Report for encoding AI findings |
| **DICOM SC** | Secondary Capture for AI overlay images |
| **HL7v2** | ADT messages, lab results, orders, flowsheet writes |
| **HL7 FHIR** | REST API for real-time data access, AI results as Observations |
| **FHIR CDS Hooks** | Clinical decision support triggering within EHR workflow |
| **FHIRcast** | Synchronizing clinical apps (EHR, PACS, AI tools) to same patient/study |
| **IHE AIR** | AI Results profile -- standard transactions for communicating AI findings |
| **IHE AIW-I** | AI Workflow for Imaging -- manages AI processing workflow |
| **IHE PWR** | Prioritization of Worklists for Reporting -- AI-driven worklist modification |
| **OAuth 2.0** | API authentication for FHIR-based data access |
| **DICOM TLS** | Transport Layer Security for DICOM communications |

---

## Sources

### Viz.ai
- [FDA De Novo Decision Summary DEN170073](https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170073.pdf)
- [FDA 510(k) K193658 (Viz ICH)](https://www.accessdata.fda.gov/cdrh_docs/pdf19/K193658.pdf)
- [FDA Permits Marketing of ContaCT](https://www.fda.gov/news-events/press-announcements/fda-permits-marketing-clinical-decision-support-software-alerting-providers-potential-stroke)
- [Viz.ai LVO Detection Publication](https://www.viz.ai/publications/ai-detection-lvo-stroke-on-ct-angiography)
- [Real World Experience with Automated LVO Detection](https://www.viz.ai/publications/experience-with-automated-lvo-detection)
- [Viz.ai Implementation in Comprehensive Stroke Center (AJNR)](https://www.ajnr.org/content/44/1/47)
- [LVO Detection in Integrated Stroke Network (PMC9847593)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9847593/)
- [Evaluation of AI LVO Identification (PMC7872164)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7872164/)
- [Viz.ai SOC 2 + HIPAA Announcement](https://www.viz.ai/news/viz-ai-announces-successful-completion-of-soc-2-type-ii-hipaa-audits-for-viz-ai-one-platform)
- [Viz.ai Cloud Architecture Blog](https://www.viz.ai/blog/cloud-powered-agility-to-healthcare-with-ai)
- [Current Stroke Solutions Using AI (PMC11674960)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11674960/)

### Aidoc
- [Aidoc CARE Foundation Model FDA Clearance (Jan 2026)](https://www.aidoc.com/about/news/aidoc-secures-fda-clearance-for-healthcares-first-comprehensive-foundation-model-ai/)
- [Aidoc Comprehensive Abdomen CT Triage](https://www.aidoc.com/comprehensive-abdomen-ct-triage/)
- [FDA Clears Aidoc Multi-Condition Tool (STAT)](https://www.statnews.com/2026/01/21/fda-clears-aidoc-tool-detect-multiple-conditions-from-ct-scan/)
- [Aidoc aiOS Platform](https://www.aidoc.com/platform/aios/)
- [Aidoc Systems Integration](https://www.aidoc.com/platform/systems-integrations/)
- [Aidoc VA Security Overview](https://www.oit.va.gov/Services/TRM/files/V1_1_AidocSolutionSecurityOverview_US.pdf)
- [Aidoc DICOM De-identification Process](https://www.oit.va.gov/Services/TRM/files/AidocDICOMde_identification_process_vF_Jan_2020.pdf)
- [Aidoc AWS Case Study](https://aws.amazon.com/solutions/case-studies/aidoc-case-study/)
- [Aidoc Data Privacy Framework](https://www.aidoc.com/data-privacy-framework-notice/)
- [FDA Clears 11 New Indications (Fierce Healthcare)](https://www.fiercehealthcare.com/ai-and-machine-learning/fda-clears-11-new-indications-aidocs-triage-solution)

### Paige.ai
- [FDA De Novo Decision Summary DEN200080](https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN200080.pdf)
- [FDA Authorization Letter DEN200080](https://www.accessdata.fda.gov/cdrh_docs/pdf20/DEN200080.pdf)
- [Paige Prostate Suite (NCBI Bookshelf)](https://www.ncbi.nlm.nih.gov/books/NBK608438/)
- [Clinical Validation of AI-Augmented Pathology (Paige Publications)](https://www.paige.ai/publications/clinical-validation-of-artificial-intelligenceaugmented-pathology-diagnosis-demonstrates-significant-gains-in-diagnostic-accuracy-in-prostate-cancer-detection)
- [FullFocus FDA Clearance (Jan 2025)](https://www.paige.ai/press-releases/paige-adds-to-regulatory-portfolio-with-new-fda-clearance)
- [Paige LIS Integration Blog](https://paige.ai/blog/building-integrated-workflows-with-the-paige-platform-our-approach-to-laboratory-information-system-integration/)
- [Practical AI Implementation in Pathology Lab (Histopathology 2025)](https://onlinelibrary.wiley.com/doi/10.1111/his.15481)
- [Paige.ai Azure Customer Story (Microsoft)](https://www.microsoft.com/en/customers/story/1731604994973070357-paigeai-azure-healthcare-en-united-states)
- [Paige + lakeFS + dbt Case Study](https://lakefs.io/case-studies/paige-ai/)
- [Tempus Acquires Paige ($81.25M)](https://www.medtechdive.com/news/tempus-paige-buyout-acquisition-ai/758589/)
- [NICE Clinical Evidence Review](https://www.nice.org.uk/advice/mib280/chapter/Clinical-and-technical-evidence)

### Sepsis Prediction Systems
- [COMPOSER Impact on Quality and Survival (npj Digital Medicine 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10805720/)
- [COMPOSER-LLM Development and Implementation (npj Digital Medicine 2025)](https://www.nature.com/articles/s41746-025-01689-w)
- [UC San Diego AI Reduces Sepsis Mortality](https://health.ucsd.edu/news/press-releases/2024-01-23-study-ai-surveillance-tool-successfully-helps-to-predict-sepsis-saves-lives/)
- [SepsisLab: Active Sensing (arXiv)](https://arxiv.org/abs/2407.16999)
- [SepsisLab (PMC11470769)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11470769/)
- [SepsisLab GitHub Repository](https://github.com/yinchangchang/sepsislab)
- [ESM External Validation (JAMA Internal Medicine 2021)](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307)
- [ESM Inpatient Validation Study (PMC10317482)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317482/)
- [ESM End User Experience (PMC11458550)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11458550/)
- [Epic Overhauls Sepsis Algorithm (STAT 2022)](https://www.statnews.com/2022/10/03/epic-sepsis-algorithm-revamp-training/)
- [ESM in County EDs (JAMIA Open 2024)](https://academic.oup.com/jamiaopen/article/7/4/ooae133/7900014)

### Standards & Integration
- [Integrating AI in Radiology Workflow (Radiology, RSNA)](https://pubs.rsna.org/doi/full/10.1148/radiol.232653)
- [CDC Hospital Sepsis Program Core Elements](https://www.cdc.gov/sepsis/hcp/core-elements/index.html)
- [Surviving Sepsis Campaign Guidelines 2021 (PMC8486643)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8486643/)
- [AI in Radiology 2025 Trends (IntuitionLabs)](https://intuitionlabs.ai/articles/ai-radiology-trends-2025)

---

## TELOS Governance Relevance

This research directly informs TELOS's value proposition for healthcare AI governance:

1. **Action chain monitoring:** TELOS's action chain tracking (SCI) maps directly to the multi-step diagnostic AI workflows documented above. Each step in the radiology AI pipeline is an action that should be governed.

2. **Drift detection across the pipeline:** Model performance degradation (Viz.ai's 96% cleared vs. 78% real-world sensitivity) is exactly the kind of drift that continuous governance monitoring should detect.

3. **Tool selection governance:** When systems like Aidoc's CARE model must select which algorithms to apply to a given scan, this is a tool selection decision that TELOS's tool_selection_gate could govern.

4. **Alert fatigue as a governance signal:** The Epic Sepsis Model's 14.7% sensitivity and resulting alert fatigue is a measurable governance failure that a fidelity monitoring system should detect and flag.

5. **Audit trail generation:** The absence of standardized AI decision audit trails across these systems is a gap that TELOS's governance_trace and governance_protocol components are designed to fill.

6. **Data governance for cloud-based inference:** The different approaches (Viz.ai sending full DICOM with PHI to cloud vs. Aidoc's de-identification-before-cloud approach) represent different risk postures that governance frameworks should evaluate and monitor.
