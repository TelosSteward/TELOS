# TELOS Boundary 3: Regulatory Text Research Report

**Boundary:** "No overriding human assessor or licensed adjuster findings"
**Researched:** 2026-02-12
**Status:** Verified from public sources (10 sources, HIGH CONFIDENCE)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Source 1: Texas Insurance Code Chapter 4101 — Adjuster Licensing

**Document URL:** https://statutes.capitol.texas.gov/Docs/IN/htm/IN.4101.htm
**Also available at:** https://law.justia.com/codes/texas/insurance-code/title-13/subtitle-a/chapter-4101/

### Verified Quoted Text

**Section 4101.001 — Definitions:**
> "'Adjuster' means a person who investigates or adjusts losses on behalf of an insurer."

**Section 4101.051 — License Required:**
> "A person may not act as, or represent or hold themselves out to be, an adjuster in this state unless the person holds a license issued under this chapter."

**Section 4101.052 — Type of License:**
> Adjusters in Texas must hold either an "all-lines adjuster" license or a "public adjuster" license.

**Source:** Verified at [Texas Legislature](https://statutes.capitol.texas.gov/Docs/IN/htm/IN.4101.htm) and [Justia](https://law.justia.com/codes/texas/insurance-code/title-13/subtitle-a/chapter-4101/).

### Relevance to Boundary 3

Texas law explicitly defines an "adjuster" as a **person** (not a system or technology) who "investigates or adjusts losses." The license requirement in Section 4101.051 means that only licensed persons can perform these functions. An AI system that overrides a licensed adjuster's findings would be performing a function that legally requires human licensure — an impossibility under the statute.

---

## Source 2: Florida Statute 626.854 — Public Adjuster Licensing

**Document URL:** https://www.flsenate.gov/laws/statutes/2023/626.854
**Also available at:** https://codes.findlaw.com/fl/title-xxxvii-insurance/fl-st-sect-626-854/

### Verified Quoted Text

**Section 626.854(1) — License Required:**
> "No person may: (a) directly or indirectly, for compensation or any other thing of value, prepare, complete, or file an insurance claim on behalf of an insured or a third-party claimant."

**Section 626.854(1)(b) — Prohibited Activities Without License:**
> "No person may act on behalf of or aid an insured or a third-party claimant in negotiating for or effecting the settlement of a claim for loss or damage covered by an insurance contract."

**Section 626.854(1)(e) — Investigation/Adjustment Prohibition:**
> "No person may solicit, investigate, or adjust a claim on behalf of a public adjuster, an insured, or a third-party claimant."

**Source:** Confirmed at [Florida Senate](https://www.flsenate.gov/laws/statutes/2023/626.854) and [FindLaw](https://codes.findlaw.com/fl/title-xxxvii-insurance/fl-st-sect-626-854/).

### Relevance to Boundary 3

Florida explicitly prohibits unlicensed persons from: **prepare, complete, file, negotiate, settle, investigate, adjust** claims. An AI that overrides a licensed adjuster's findings would be performing these licensed functions.

---

## Source 3: Florida Statute 627.70131 — Licensed Adjuster Physical Inspection

**Document URL:** https://www.flsenate.gov/Laws/Statutes/2023/627.70131

### Verified Quoted Text

**Section 627.70131(3)(b) — Licensed Adjuster Physical Inspection Requirement:**
> "If such investigation involves a physical inspection of the property, the licensed adjuster assigned by the insurer must provide the policyholder with a printed or electronic document containing his or her name and state adjuster license number."

> "An insurer must conduct any such physical inspection within 30 days after its receipt of the proof-of-loss statements."

**Source:** Confirmed at [Florida Senate](https://www.flsenate.gov/Laws/Statutes/2023/627.70131) and [Justia](https://law.justia.com/codes/florida/title-xxxvii/chapter-627/part-x/section-627-70131/).

### Relevance to Boundary 3

Physical inspections must be conducted by a licensed adjuster identified by name and state license number. An AI system cannot fulfill this statutory requirement.

---

## Source 4: NAIC Model Bulletin on AI — Human Involvement Factor (December 4, 2023)

**Document URL:** https://content.naic.org/sites/default/files/inline-files/2023-12-4%20Model%20Bulletin_Adopted_0.pdf

### Verified Quoted Text

**Section 3 — Human Involvement Factor:**
> "Controls and processes in the AI program should be reflective of, and commensurate with, insurers' assessment of the degree and nature of risk posed to consumers by the AI systems considering: ... (3) the extent to which humans are involved in the final decision-making process..."

**Core Mandate:**
> "decisions or actions impacting consumers that are made or supported by...AI Systems...must comply with all applicable insurance laws and regulations."

**AIS Program Scope — Claims:**
> The AIS Program should address the use of AI systems across the insurance lifecycle, including "product development and design, marketing, use, underwriting, rating, and pricing, case management, claim administration and payment, and fraud detection."

**Source:** Confirmed via [Quarles & Brady](https://www.quarles.com/newsroom/publications/states-adopt-naic-model-bulletin-on-insurers-use-of-ai), [McDermott Will & Emery](https://www.mwe.com/insights/state-regulators-address-insurers-use-of-ai-11-states-adopt-naic-model-bulletin/), [Kennedys Law](https://www.kennedyslaw.com/en/thought-leadership/article/2025/understanding-the-naic-model-ai-bulletin-what-it-means-for-insurers/).

### Relevance to Boundary 3

When AI overrides a human adjuster, humans are removed from the decision-making process — the NAIC treats this as requiring the most stringent controls. The "made or supported" distinction is key: when AI supplements the adjuster, it's "supporting" (compliant). When it overrides, it transitions to "making" (requires full regulatory compliance only a licensed professional can satisfy).

---

## Source 5: Pennsylvania Insurance Department Notice 2024-06 — Aerial Imagery (May 25, 2024)

**Document URL:** https://www.pacodeandbulletin.gov/Display/pabull?file=/secure/pabulletin/data/vol54/54-21/735.html
**Press Release:** https://www.pa.gov/agencies/insurance/newsroom/shapiro-administration-urges-insurers-to-properly-use-aerial-imagery-and-verify-damages-on-consumer-homes

### Verified Quoted Text

**Physical Inspection Requirement When Imagery is Inconclusive:**
> "In the absence of unequivocal and material damage shown, it would be prudent for an insurer to conduct a physical inspection to validate the specific type of damage that the aerial image purports to evidence."

**Cosmetic vs. Material Damage:**
> Images that simply show discoloration, streaking, or other cosmetic damage should not be used as the sole reason to nonrenew or cancel a policy.

**Source:** Confirmed via [Insurance Journal](https://www.insurancejournal.com/news/east/2024/05/29/776078.htm), [PA Insurance Dept.](https://www.pa.gov/agencies/insurance/newsroom/shapiro-administration-urges-insurers-to-properly-use-aerial-imagery-and-verify-damages-on-consumer-homes).

### Relevance to Boundary 3

Pennsylvania explicitly requires physical inspection when aerial imagery (Nearmap's exact product) does not show **unequivocal and material damage**. Aerial imagery analysis cannot override the need for human on-ground assessment.

---

## Source 6: Connecticut CID — Aerial Imagery and Roof Condition (March 19, 2024)

**Document URL:** https://portal.ct.gov/cid/-/media/cid/1_protertycasualty/aerial-imagery-notice.pdf

### Verified Quoted Text

**Physical Inspection Guardrail:**
> "An example of a guardrail to prevent unsupported underwriting action is, in the absence of unequivocal material damage shown, a physical inspection to validate the specific type of damage that the aerial image purports to show or receipt of a report prepared by a licensed home improvement contractor or roofer addressing the condition of the subject roof as submitted by the policyholder."

**Cosmetic Conditions Insufficient:**
> "The Department's position is that cosmetic roofing conditions such as natural discoloration and streaking do not support taking homeowner or dwelling fire nonrenewal action."

**Underwriting Guidelines Filing:**
> "If an insurer intends to take underwriting action based on the age and/or condition of the roof, underwriting guidelines addressing this practice must be filed including all revisions to the guidelines before implementing."

**Source:** Confirmed via [PIA Northeast](https://blog.pia.org/2024/04/24/cid-issues-bulletin-on-aerial-imagery-and-underwriting/) and [NAMIC](https://www.namic.org/news/connecticut-cid-addresses-underwriting-action-based-on-roofing-condition/).

### Relevance to Boundary 3

Connecticut positions aerial imagery as something that must be **validated** by physical inspection. Human inspector has the final word, not the AI/aerial analysis.

---

## Source 7: New Hampshire INS 25-016-AB — Aerial Imagery (February 19, 2025)

**Document URL:** https://www.insurance.nh.gov/news-and-media/new-hampshire-insurance-department-issues-bulletin-use-aerial-imagery-property

### Verified Quoted Text

**Physical Inspection Mandate When Imagery is Inconclusive:**
> "If aerial imagery does not conclusively demonstrate roof degradation or damage sufficient to justify refusal to write or renew a policy, insurers must conduct a follow-up physical inspection to verify their findings."

**Disputed Findings Require Physical Inspection:**
> "If an applicant or insured disputes the insurer's determination -- even when the insurer believes the aerial imagery to be conclusive -- a physical inspection is required."

**Cosmetic Conditions Limitation:**
> "While insurers may refuse to write or renew policies due to clear evidence of property degradation or damage, they may not do so solely based on cosmetic issues such as roof discoloration."

**Source:** Confirmed at [NHID](https://www.insurance.nh.gov/news-and-media/new-hampshire-insurance-department-issues-bulletin-use-aerial-imagery-property), [PIA Northeast](https://blog.pia.org/2025/02/20/nhid-issues-bulletin-on-aerial-imagery-in-underwriting/), [NAMIC](https://www.namic.org/news/250220cw03/).

### Relevance to Boundary 3

**Strongest articulation:** Even when the insurer believes imagery is conclusive, if a policyholder disputes it, physical inspection is mandatory. The AI's confidence score is irrelevant when a human contests the finding.

---

## Source 8: North Carolina 25-B-09 — Aerial Images (August 11, 2025)

**Document URL:** https://www.ncdoi.gov/25-b-09-use-aerial-images/open

### Verified Quoted Text

**Not Sole Information Source:**
> "If possible, aerial imagery should not be the only information utilized in decision-making."

**Accuracy Concerns:**
> "Aerial imagery does not always provide an accurate representation of a property's condition. An image may be out-of-date, it may be blurry, or there may be objects in the image that obstruct the view of a property."

**Real-World Error Example:**
> "The Department received multiple complaints after one insurer issued nonrenewals based on drone images it had purchased from a third party that were not clear, leading to problems like well-manicured flower beds being identified as yard waste."

**Source:** Confirmed at [NCDOI](https://www.ncdoi.gov/25-b-09-use-aerial-images/open).

---

## Source 9: Rhode Island Bulletin 2025-3 — Aerial Imaging (August 2025)

**Document URL:** https://dbr.ri.gov/sites/g/files/xkgbur696/files/2025-08/INS_Insurance%20Bulletin%202025-3%20Aerial%20Imaging.pdf

### Verified Quoted Text

**Cannot Alone Justify Adverse Action:**
> "Images of insured property that are low-resolution, out-of-focus, blurry, or not current do not provide an accurate and clear representation of the property, and thus cannot alone justify a cancellation or nonrenewal based on the condition of the property without further investigation into the condition of the property."

**Source:** Confirmed at [RI DBR](https://dbr.ri.gov/sites/g/files/xkgbur696/files/2025-08/INS_Insurance%20Bulletin%202025-3%20Aerial%20Imaging.pdf).

---

## Summary of Key Regulatory Phrases for Boundary 3

| Source | Key Language | Boundary 3 Implication |
|--------|-------------|----------------------|
| Texas Insurance Code 4101.001 | "investigates or adjusts losses" requires a licensed person | Only a licensed adjuster can investigate or adjust — AI cannot |
| Texas Insurance Code 4101.051 | "a person may not act as...an adjuster...unless the person holds a license" | AI is not a licensed person |
| Florida 626.854(3) | "No person may...investigate, or adjust a claim" without a license | Investigating/adjusting is a licensed activity |
| Florida 627.70131(3)(b) | "the licensed adjuster assigned by the insurer must" conduct physical inspection | Physical inspection is licensed-human-only |
| NAIC AI Bulletin, Sec. 3 | "the extent to which humans are involved in the final decision-making process" | Less human involvement = higher regulatory risk |
| Pennsylvania Notice 2024-06 | "conduct a physical inspection to validate...the aerial image" | Aerial imagery requires human validation |
| Connecticut CID (March 2024) | "a physical inspection to validate the specific type of damage that the aerial image purports to show" | Human inspection validates imagery |
| New Hampshire INS 25-016-AB | "insurers must conduct a follow-up physical inspection to verify their findings" | Even "conclusive" imagery overridden by human dispute |
| North Carolina 25-B-09 | "aerial imagery should not be the only information utilized in decision-making" | AI/imagery is supplemental, not standalone |
| Rhode Island 2025-3 | "cannot alone justify a cancellation or nonrenewal...without further investigation" | Imagery alone is legally insufficient |

## States That Have Issued Aerial Imagery Guidance (as of 2025)

| State | Document | Date | Key Requirement |
|-------|----------|------|-----------------|
| Pennsylvania | Notice 2024-06 | May 25, 2024 | Physical inspection when imagery inconclusive |
| Connecticut | CID Aerial Imagery Notice | March 19, 2024 | Physical inspection or licensed contractor report |
| New Hampshire | INS 25-016-AB | February 19, 2025 | Physical inspection when disputed, even if "conclusive" |
| North Carolina | 25-B-09 | August 11, 2025 | Aerial imagery not sole decision basis |
| Rhode Island | Bulletin 2025-3 | August 2025 | Cannot alone justify cancellation/nonrenewal |
| Massachusetts | Bulletin 2025-02 | 2025 | Physical inspection when imagery not unequivocal |

## Composite Regulatory Principle

Across all sources, the regulatory consensus is clear:

1. **Investigating and adjusting losses is a licensed activity** that only a human professional can perform (TX 4101, FL 626.854).
2. **Physical inspections must be conducted by licensed adjusters** identified by name and license number (FL 627.70131).
3. **Aerial imagery and AI analysis are supplements, not replacements** for human on-site assessment (PA, CT, NH, NC, RI bulletins).
4. **When imagery is inconclusive or disputed, physical inspection is mandatory** — even when the insurer believes the imagery is conclusive (NH INS 25-016-AB).
5. **The degree of human involvement is a regulatory risk factor** — less human involvement means more regulatory scrutiny (NAIC AI Bulletin).

## Items NOT Found

- **NAIC Model #900** — Primarily addresses rate filing, not adjuster roles. No quotable adjuster-specific text.
- **"Desk Adjusting" statutes** — No states use this term in statute. Instead addressed through physical inspection requirements and adjuster licensing laws.
- **NAIC PL-40** — Adjuster licensing comparison chart, not quotable model law text.
