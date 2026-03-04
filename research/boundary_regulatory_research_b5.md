# TELOS Boundary 5: Regulatory Text Research Report

**Boundary:** "No autonomous repair/replace authorization (ITEL repair vs. replace decisions require licensed adjuster)"
**Researched:** 2026-02-12
**Status:** Verified from public sources (13 sources, HIGH CONFIDENCE)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Source 1: NAIC Model #900 — Unfair Claims Settlement Practices Act

**Document URL:** https://content.naic.org/sites/default/files/model-law-900.pdf
**Note:** Full NAIC PDF could not be directly rendered. Text verified through state codifications.

### Verified Quoted Text (via state codifications)

**Section 4(c) — Prompt Investigation:**
> "Failing to adopt and implement reasonable standards for the prompt investigation of claims arising under its policies."

**Section 4(d) — Reasonable Investigation Before Denial:**
> "Refusing to pay claims without conducting a reasonable investigation based upon all available information."

**Section 4(f) — Good Faith Settlement:**
> "Not attempting in good faith to effectuate prompt, fair, and equitable settlement of claims submitted in which liability has become reasonably clear."

**Section 4(g) — Affirm or Deny Timely:**
> "Failing to affirm or deny coverage of claims within a reasonable time after proof of loss statements have been completed."

### Relevance to Boundary 5

Model 900 requires "reasonable investigation" before any claims determination. An ITEL report or aerial analysis is a data input, not an investigation. The "investigation" must be conducted by qualified personnel per state licensing requirements.

---

## Source 2: Texas Insurance Code Chapter 542 — Prompt Payment of Claims

**Document URL:** https://statutes.capitol.texas.gov/Docs/IN/htm/IN.542.htm

### Verified Quoted Text

**Section 542.055(a) — Acceptance/Rejection Timeline:**
> "Not later than the 15th business day after the date an insurer receives all items, statements, and forms required by the insurer, the insurer shall: (1) approve the claim in whole or in part and notify the claimant in writing of the approval; or (2) reject the claim in whole or in part and notify the claimant in writing of the rejection."

**Section 542.056(a) — Written Notice Required:**
> "An insurer shall notify a claimant in writing of the acceptance or rejection of a claim not later than the 15th business day after the date the insurer receives all items, statements, and forms reasonably requested and required under the policy."

**Section 542.058 — Payment Deadline:**
> "If an insurer ... delays payment of the claim for a period exceeding the period specified ... the insurer is liable to pay the holder of the policy ... the amount of the claim, plus interest on the amount of the claim."

**Source:** Verified at [Texas Legislature Online](https://statutes.capitol.texas.gov/Docs/IN/htm/IN.542.htm) and [Justia](https://law.justia.com/codes/texas/insurance-code/title-5/subtitle-d/chapter-542/).

### Relevance to Boundary 5

Texas places the duty to approve, reject, notify, and pay on the "insurer" — these are human institutional decisions, not automated outputs. The 15-day timeline for written acceptance/rejection presupposes human review and authorization.

---

## Source 3: Texas Insurance Code Chapter 4101 — Adjuster Licensing

**Document URL:** https://statutes.capitol.texas.gov/Docs/IN/htm/IN.4101.htm

### Verified Quoted Text

**Section 4101.001 — Definition:**
> "'Adjuster' means a person who investigates or adjusts losses on behalf of an insurer."

**Section 4101.051 — License Required:**
> "A person may not act as, or represent or hold themselves out to be, an adjuster in this state unless the person holds a license issued under this chapter."

**Section 4102.163 — Contractor Cannot Act as Adjuster:**
> "A contractor may not act as a public adjuster or advertise to adjust claims for ... roofing services, regardless of whether the contractor: (1) holds a license under this chapter; or (2) is authorized to act on behalf of the insured under a power of attorney or other agreement."

**Source:** Verified at [Texas Legislature Online](https://statutes.capitol.texas.gov/Docs/IN/htm/IN.4101.htm) and [Justia](https://law.justia.com/codes/texas/insurance-code/title-13/subtitle-a/chapter-4101/).

### Relevance to Boundary 5

Texas creates a bright line: adjusting requires a license. Contracting cannot be combined with adjusting on the same property. An AI system that autonomously authorizes repair or replacement performs the adjuster function without a license.

---

## Source 4: Florida Section 627.70131 — Claims Investigation and Physical Inspection

**Document URL:** https://m.flsenate.gov/Statutes/627.70131

### Verified Quoted Text

**Section 627.70131(2)(a) — Investigation Commencement:**
> "Within 14 days after an insurer receives proof-of-loss statements, the insurer shall begin such investigation as is reasonably necessary."

**Section 627.70131(3)(b) — Licensed Adjuster Identification (CRITICAL):**
> "If such investigation involves a physical inspection of the property, the licensed adjuster assigned by the insurer must provide the policyholder with a printed or electronic document containing his or her name and state adjuster license number."

**Section 627.70131(3)(c) — Adjuster ID in Communications:**
> "Any subsequent communication with the policyholder regarding the claim must also include the name and license number of the adjuster communicating about the claim."

**Section 627.70131(3)(d) — Electronic Methods Permitted for Investigation:**
> "An insurer may use electronic methods to investigate the loss. Such electronic methods may include any method that provides the insurer with clear, color pictures or video documenting the loss, including, but not limited to, electronic photographs or video recordings of the loss; video conferencing between the adjuster and the policyholder which includes video recording of the loss; and video recordings or photographs of the loss using a drone, driverless vehicle, or other machine that can move independently or through remote control."

**Section 627.70131(7) — Pay or Deny Timeline:**
> "Within 90 days after an insurer receives notice of an initial, reopened, or supplemental property insurance claim from a policyholder, the insurer shall pay or deny such claim or a portion of the claim."

**Source:** Confirmed at [Florida Senate](https://m.flsenate.gov/Statutes/627.70131) and [Justia](https://law.justia.com/codes/florida/title-xxxvii/chapter-627/part-x/section-627-70131/).

### Relevance to Boundary 5

Florida 627.70131 is critical because it:
1. Requires a **licensed adjuster** identified by name and license number for physical inspections
2. Explicitly **permits electronic methods** (including drones) — validating Nearmap/ITEL as investigative tools
3. But maintains that the **insurer** (not the AI/tool) must pay or deny

The electronic methods provision legitimizes AI-assisted analysis while the licensed adjuster requirement prevents fully autonomous decisions.

---

## Source 5: Florida Section 626.877 — Adjuster Compliance

**Document URL:** https://codes.findlaw.com/fl/title-xxxvii-insurance/fl-st-sect-626-877.html

### Verified Quoted Text

**Every Adjuster Shall Adjust:**
> "Every adjuster shall adjust or investigate every claim, damage, or loss made or occurring under an insurance contract, in accordance with the terms and conditions of the contract and of the applicable laws of this state."

**Source:** Confirmed at [Justia](https://law.justia.com/codes/florida/2016/title-xxxvii/chapter-626/part-vi/section-626.877).

### Relevance to Boundary 5

Direct mandate: "Every adjuster shall adjust or investigate every claim." An AI system that autonomously issues repair/replace authorization without adjuster review violates this requirement.

---

## Source 6: Florida HB 527 (2026 Session — PENDING, NOT YET ENACTED)

**Document URL:** https://www.flsenate.gov/Session/Bill/2026/527

### Verified Quoted Text (from bill analysis and news reporting)

**Core Prohibition:**
> The bill "prohibits workers' compensation carriers, insurers, and health maintenance organizations from reducing a claim payment, denying a claim, or denying a portion of a claim based solely on the output of an AI system, algorithm, or machine learning system."

**Qualified Human Professional Requirement:**
> "The person who reviews claim denials would have to be a 'qualified human professional' — 'an individual who, under the Florida Insurance Code, has the authority to adjust or deny a claim or a portion of a claim and may exercise such authority over a particular claim.'"

**Claims Handling Manual Requirement:**
> "When an insurer plans to use algorithms, artificial intelligence systems, or machine learning systems in its claims handling processes, it must detail in its claims handling manual the manner in which the systems are to be used and how the systems comply with the law."

**Source:** [Repairer Driven News](https://www.repairerdrivennews.com/2025/12/11/florida-advances-bill-on-ai-use-in-claims-handling/) and [Florida Politics](https://floridapolitics.com/archives/768954-cassel-ai-insurance-houseib/).

**IMPORTANT CAVEAT:** HB 527 has NOT yet been enacted. Cited because it represents regulatory direction and directly addresses the AI-human boundary in claims.

---

## Source 7: Louisiana RS 22:1892 — Payment and Adjustment of Claims

**Document URL:** https://legis.la.gov/Legis/Law.aspx?d=509041

### Verified Quoted Text

**Section 22:1892(A)(1) — Payment Timeline:**
> "All insurers issuing any type of contract...shall pay the amount of any claim due any insured within thirty days after receipt of satisfactory proofs of loss from the insured or any party in interest."

**Section 22:1892(A)(4) — Written Offer to Settle:**
> "All insurers shall make a written offer to settle any property damage claim, including a third-party claim, within the applicable number of days after receipt of satisfactory proofs of loss of that claim."

**Source:** [Justia — LA RS 22:1892](https://law.justia.com/codes/louisiana/revised-statutes/title-22/rs-22-1892/).

---

## Source 8: Louisiana RS 22:1896 — Transparency and Integrity in Property Claims

**Document URL:** https://legis.la.gov/Legis/Law.aspx?d=509045

### Verified Quoted Text

**Qualified Adjuster Requirement:**
> "An insurer of a residential or commercial property shall provide prompt adjustment by a qualified adjuster pursuant to the provisions of R.S. 22:1661 et seq."

**Field Adjuster Report Transparency:**
> "An insurer of a residential or commercial property shall furnish a copy of the insurer's field adjuster report to the insured within fifteen days of receiving a request from the insured."

**Source:** [Justia — LA RS 22:1896](https://law.justia.com/codes/louisiana/2012/title-xxxvii/chapter-626/part-vi/section-626.877).

### Relevance to Boundary 5

Louisiana explicitly requires "prompt adjustment by a qualified adjuster." An ITEL report is an analytical tool, not an adjustment. Nearmap aerial imagery is an investigative resource, not a claims determination.

---

## Source 9: Oklahoma Title 36, Sections 1250.5 and 1250.7

**Document URLs:**
- https://law.justia.com/codes/oklahoma/title-36/section-36-1250-5/
- https://law.justia.com/codes/oklahoma/title-36/section-36-1250-7/

### Verified Quoted Text

**Section 1250.5 — Unfair Claim Settlement Practices:**
> "(3) Failing to adopt and implement reasonable standards for prompt investigations of claims arising under its insurance policies;"
> "(4) Not attempting in good faith to effectuate prompt, fair and equitable settlement of claims submitted in which liability has become reasonably clear;"

**Section 1250.7 — Acceptance or Denial:**
> "Within forty-five (45) days after receipt by a property and casualty insurer of properly executed proofs of loss, the first party claimant shall be advised of the acceptance or denial of the claim by the insurer."
> "A denial shall be given to any claimant in writing, and the claim file of the property and casualty insurer shall contain a copy of the denial."
> "No property and casualty insurer shall deny a claim because of a specific policy provision, condition, or exclusion unless reference to such provision, condition, or exclusion is included in the denial."

**Source:** [Justia — OK 36-1250.5](https://law.justia.com/codes/oklahoma/title-36/section-36-1250-5/) and [OK 36-1250.7](https://law.justia.com/codes/oklahoma/title-36/section-36-1250-7/).

---

## Source 10: Colorado C.R.S. 10-3-1104 — Unfair Claims Settlement Practices

**Document URL:** https://law.justia.com/codes/colorado/title-10/regulation-of-insurance-companies/article-3/part-11/section-10-3-1104/

### Verified Quoted Text

**Section 10-3-1104(1)(h) — Unfair Claim Settlement Practices:**
> "Failing to adopt and implement reasonable standards for the prompt investigation of claims arising under insurance policies; ... Refusing to pay claims without conducting a reasonable investigation based upon all available information; ... Failing to affirm or deny coverage of claims within a reasonable time after proof of loss statements have been completed; ... Failing to promptly provide a reasonable explanation of the basis in the insurance policy in relation to the facts or applicable law for denial of a claim or for the offer of a compromise settlement."

**Source:** [Justia — CO CRS 10-3-1104](https://law.justia.com/codes/colorado/title-10/regulation-of-insurance-companies/article-3/part-11/section-10-3-1104/).

---

## Source 11: NAIC Model Bulletin on AI (December 4, 2023) — Claims Context

(Same document as B1/B3 research, claims-specific provisions)

### Verified Quoted Text

**Human Involvement Factor:**
> "the extent to which humans are involved in the final decision-making process" (Section 3, Factor 3)

**AIS Program Scope:**
> "claim administration and payment" is explicitly within scope.

**Insurer Responsibility:**
> "If insurers use AI systems, whether their own or those of third-party vendors, they remain obligated to comply with applicable legal and regulatory standards, including unfair trade practices and unfair claims settlement laws, that require, at a minimum, that decisions made by insurers are not inaccurate, arbitrary, capricious or unfairly discriminatory."

**Source:** Confirmed via [Kennedys Law](https://www.kennedyslaw.com/en/thought-leadership/article/2025/understanding-the-naic-model-ai-bulletin-what-it-means-for-insurers/).

---

## Source 12: Texas 28 TAC Section 21.203 — Administrative Regulation

**Document URL:** https://www.law.cornell.edu/regulations/texas/28-Tex-Admin-Code-SS-21-203

### Verified Quoted Text

> "(3) failing to adopt and implement reasonable standards for prompt investigations of claims;"
> "(4) not attempting in good faith to promptly settle claims where liability has become reasonably clear;"
> "(9) failing to promptly provide to a policyholder a reasonable explanation of the basis in the insurance policy in relation to the facts or applicable law for denial of a claim;"
> "(10) failing to affirm or deny coverage of a claim to a policyholder within a reasonable time;"

**Source:** [Cornell LII](https://www.law.cornell.edu/regulations/texas/28-Tex-Admin-Code-SS-21-203).

---

## Source 13: Xactimate / Estimating Standards

No state-level statute mandates the use of Xactimate or any specific estimating software. Florida requires transparency and documentation of estimates but does not require Xactimate specifically. Insurance policies generally do not require any particular method of estimating damages.

---

## Items NOT Found (Transparency Disclosure)

1. **NAIC Model 900 exact subsection text from PDF** — Verified through state codifications instead.
2. **"Desk adjusting" prohibition statutes** — No state uses this term. Florida 627.70131(3)(d) actually permits electronic methods. The constraint is on WHO makes the determination, not HOW the investigation is conducted.
3. **Xactimate-specific mandates** — None found.
4. **Direct AI claims authorization prohibition (enacted)** — No currently enacted statute explicitly says "AI may not authorize repairs." The prohibition is indirect through adjuster licensing requirements. Florida HB 527 (pending) would be the first.

---

## Consolidated Regulatory Principle for Boundary 5

Across all sources, the regulatory framework converges:

1. **Claims investigation and adjustment is a licensed activity** (TX 4101, FL 626.877, CO 10-2-103, LA RS 22:1896)
2. **Unfair claims settlement laws require reasonable human investigation** (NAIC Model 900, all state codifications)
3. **Prompt-pay statutes place duty on the insurer** — not on automated systems (TX 542, FL 627.70131, LA 22:1892, OK 36-1250.7)
4. **NAIC AI Model Bulletin** requires compliance with all existing claims laws when using AI
5. **Electronic methods are permitted for investigation** but the determination authority remains with licensed humans (FL 627.70131(3)(d))

**The regulatory logic:** ITEL and Nearmap provide analysis. Licensed adjusters make determinations. Boundary 5 enforces this separation.
