# TELOS Boundary 1: Regulatory Text Research Report

**Boundary:** "No binding underwriting decisions (decision support only, per NAIC Model Bulletin)"
**Researched:** 2026-02-12
**Status:** Verified from public sources

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Source 1: NAIC Model Bulletin on the Use of Artificial Intelligence Systems by Insurers (Adopted December 4, 2023)

**Document URL:** https://content.naic.org/sites/default/files/inline-files/2023-12-4%20Model%20Bulletin_Adopted_0.pdf

### Verified Quoted Text

**Core Mandate (Opening Statement):**
> "This bulletin is issued by the [] (Department) to remind all Insurers that hold certificates of authority to do business in the state that decisions or actions impacting consumers that are made or supported by advanced analytical and computational technologies, including Artificial Intelligence (AI) Systems (as defined below), must comply with all applicable insurance laws and regulations."

**Source:** Confirmed via multiple law firm analyses including [McDermott Will & Emery](https://www.mwe.com/insights/state-regulators-address-insurers-use-of-ai-11-states-adopt-naic-model-bulletin/) and [Sullivan & Cromwell](https://www.sullcrom.com/SullivanCromwell/_Assets/PDFs/Memos/NAIC-Model-Bulletin-Use-AI-Insurers.pdf). The phrase "made or supported by" is the critical language — it establishes AI as being in either a decision-making ("made") or decision-support ("supported by") role, and requires compliance in both cases.

**Regulatory Oversight Section:**
> "The Department's regulatory oversight of Insurers includes oversight of an Insurer's conduct in the state, including its use of AI Systems to make or support decisions that impact consumers."

**Investigation Expectations:**
> "Regardless of the existence or scope of a written AIS Program, in the context of an investigation or market conduct action, an Insurer can expect to be asked about its development, deployment, and use of AI Systems, or any specific Predictive Model, AI System or application and its outcomes (including Adverse Consumer Outcomes) from the use of those AI Systems, as well as any other information or documentation deemed relevant by the Department."

**Insurer Responsibility for Rates:**
> "An Insurer is responsible for assuring that rates, rating rules, and rating plans that are developed using AI techniques and predictive models that rely on data and machine learning do not result in excessive, inadequate, or unfairly discriminatory insurance rates."

**Risk Management Controls — Human Involvement Factor (Section 3):**
> "Controls and processes in the AI program should be reflective of, and commensurate with, insurers' assessment of the degree and nature of risk posed to consumers by the AI systems considering: (1) the nature of the decisions being made, informed, or supported using the AI system; (2) the type and degree of potential harm to consumers resulting from the use of AI systems; (3) the extent to which humans are involved in the final decision-making process; (4) the transparency and explainability of outcomes to the impacted consumer; and (5) the extent and scope of the insurer's use or reliance on data, predictive models, and AI systems from third parties."

**Source:** Directly quoted from [Quarles & Brady analysis](https://www.quarles.com/newsroom/publications/states-adopt-naic-model-bulletin-on-insurers-use-of-ai).

Factor (3) — "the extent to which humans are involved in the final decision-making process" — is the critical language for Boundary 1. It establishes a regulatory expectation that humans are involved in final decisions, and that the degree of human involvement is a key risk factor that must be assessed. Less human involvement = higher risk = more stringent controls required.

**AIS Program Purpose:**
> "AIS programs are designed to assure that decisions impacting consumers which are made or supported by AI systems are accurate and do not violate unfair trade practice laws or other applicable legal standards."

**Source:** Quoted from [DLA Piper analysis](https://www.dlapiper.com/en/insights/publications/ai-outlook/2023/the-naic-model-bulletin-on-algorithms-and-predictive-models-in-insurance).

---

## Source 2: NAIC Model #880 — Unfair Trade Practices Act

**Document URL:** https://content.naic.org/sites/default/files/model-law-880.pdf

### Verified Quoted Text

**Section 1 — Purpose:**
> "The purpose of this Act is to regulate trade practices in the business of insurance in accordance with the intent of Congress as expressed in the Act of Congress of March 9, 1945 (Public Law 15, 79th Congress) and the Gramm-Leach-Bliley Act (Public Law 106-102, 106th Congress), by defining, or providing for the determination of, all such practices in this state that constitute unfair methods of competition or unfair or deceptive acts or practices and by prohibiting the trade practices so defined or determined."

**Section 4G — Unfair Discrimination (Life Insurance):**
> "Making or permitting any unfair discrimination between individuals of the same class and equal expectation of life in the rates charged for any life insurance policy or annuity or in the dividends or other benefits payable thereon, or in any other of the terms and conditions of such policy."

**Section 4G — Unfair Discrimination (Health Insurance):**
> "Making or permitting any unfair discrimination between individuals of the same class and essentially same hazard in any policy or contract of health [insurance]."

**Section 4H — Unfair Discrimination (Property/Casualty — Geographic):**
> "Making or permitting any unfair discrimination between individuals or risks of the same class and of essentially the same hazard by refusing to insure, refusing to renew, canceling or limiting the amount of insurance coverage on a property or casualty risk solely because of the geographic location of the risk, unless such action is the result of the application of sound underwriting and actuarial principles related to actual or reasonably anticipated loss experience."

**Section 4H — Unfair Discrimination (Protected Classes):**
> "Refusing to insure, or refusing to continue to insure, or limiting the amount of coverage available to an individual because of the sex, marital status, race, religion or national origin of the individual."

**Section 4H — Disability Discrimination:**
> "Refusing to insure, refusing to continue to insure, or limiting the amount, extent or kind of coverage available to an individual, or charging a different rate for the same coverage solely because of a physical or mental impairment, except where the refusal, limitation or rate differential is based on sound actuarial principles or is related to actual or reasonably anticipated experience."

### Relevance to Boundary 1

Model 880 does not explicitly address AI. However, it creates the underlying legal framework that the NAIC AI Model Bulletin references. The key phrase "Making or permitting" establishes that an insurer can violate these provisions both by actively discriminating and by "permitting" discrimination through its systems (including AI). An AI system that makes autonomous underwriting decisions without human review could "permit" unfair discrimination, exposing the insurer to liability under Section 4.

---

## Source 3: Colorado C.R.S. Section 10-3-1104.9 (SB 21-169)

**Document URL:** https://law.justia.com/codes/colorado/title-10/regulation-of-insurance-companies/article-3/part-11/section-10-3-1104-9/
**DORA Page:** https://doi.colorado.gov/for-consumers/sb21-169-protecting-consumers-from-unfair-discrimination-in-insurance-practices

### Verified Quoted Text (Full Statutory Text from Justia)

**Section 10-3-1104.9(1) — Core Prohibition:**
> **(1)** In addition to the methods and practices prohibited pursuant to section 10-3-1104 (1)(f), an insurer shall not, with regard to any insurance practice:
>
> **(a)** Unfairly discriminate based on race, color, national or ethnic origin, religion, sex, sexual orientation, disability, gender identity, or gender expression; or
>
> **(b)** Pursuant to rules adopted by the commissioner, use any external consumer data and information sources, as well as any algorithms or predictive models that use external consumer data and information sources, in a way that unfairly discriminates based on race, color, national or ethnic origin, religion, sex, sexual orientation, disability, gender identity, or gender expression.

**Section 10-3-1104.9(3)(b) — Governance Framework Requirements:**
> **(3)(b)** Rules adopted pursuant to this section must require each insurer to:
>
> **(I)** Provide information to the commissioner concerning the external consumer data and information sources used by the insurer in the development and implementation of algorithms and predictive models for a particular type of insurance and insurance practice;
>
> **(II)** Provide an explanation of the manner in which the insurer uses external consumer data and information sources, as well as algorithms and predictive models using external consumer data and information sources, for the particular type of insurance and insurance practice;
>
> **(III)** Establish and maintain a risk management framework or similar processes or procedures that are reasonably designed to determine, to the extent practicable, whether the insurer's use of external consumer data and information sources, as well as algorithms and predictive models using external consumer data and information sources, unfairly discriminates based on race, color, national or ethnic origin, religion, sex, sexual orientation, disability, gender identity, or gender expression;
>
> **(IV)** Provide an assessment of the results of the risk management framework or similar processes or procedures and actions taken to minimize the risk of unfair discrimination, including ongoing monitoring; and
>
> **(V)** Provide an attestation by one or more officers that the insurer has implemented the risk management framework or similar processes or procedures appropriately on a continuous basis.

**Section 10-3-1104.9(8)(a) — Definition of "Algorithm" (CRITICAL):**
> **(8)(a)** "Algorithm" means a computational or machine learning process that **informs human decision making** in insurance practices.

**This definition is the single most important piece of text for Boundary 1.** Colorado's statute **defines** an algorithm as something that "informs human decision making" — not something that makes decisions autonomously. The algorithm is, by statutory definition, a decision-support tool.

**Section 10-3-1104.9(8)(c) — Definition of "Insurance Practice":**
> **(8)(c)** "Insurance practice" means marketing, underwriting, pricing, utilization management, reimbursement methodologies, and claims management in the transaction of insurance.

**Section 10-3-1104.9(8)(e) — Definition of "Unfair Discrimination":**
> **(8)(e)** "Unfairly discriminate" and "unfair discrimination" include the use of one or more external consumer data and information sources, as well as algorithms or predictive models using external consumer data and information sources, that have a correlation to race, color, national or ethnic origin, religion, sex, sexual orientation, disability, gender identity, or gender expression, and that use results in a disproportionately negative outcome for such classification or classifications, which negative outcome exceeds the reasonable correlation to the underlying insurance practice, including losses and costs for underwriting.

---

## Source 4: New York DFS Insurance Circular Letter No. 7 (2024)

**Document URL:** https://www.dfs.ny.gov/industry-guidance/circular-letters/cl2024-07
**Date:** July 11, 2024

### Verified Quoted Text (Full Text Confirmed from DFS Website)

**Section I, Paragraph 3 — Compliance Expectation:**
> "The Department expects that insurers' use of emerging technologies, such as AIS and ECDIS, will be conducted in a manner that complies with all applicable federal and state laws and regulations."

**Section I, Paragraph 4 — Risk of Autonomous AI:**
> "ECDIS may reflect systemic biases and its use can reinforce and exacerbate inequality. This raises significant concerns about the potential for unfair adverse effects or discriminatory decision-making. ECDIS also may have variable accuracy and reliability and may come from entities that are not subject to regulatory oversight and consumer protections. Furthermore, the self-learning behavior that may be present in AIS increases the risks of inaccurate, arbitrary, capricious, or unfairly discriminatory outcomes that may disproportionately affect vulnerable communities and individuals or otherwise undermine the insurance marketplace in New York. It is critical that insurers that utilize such technologies establish a proper governance and risk management framework to mitigate the potential harm to consumers and comply with all relevant legal obligations."

**Section II, Paragraph 10 — Anti-Discrimination:**
> "An insurer should not use ECDIS or AIS for underwriting or pricing purposes unless the insurer can establish that the data source or model, as applicable, does not use and is not based in any way on any class protected pursuant to Insurance Law Article 26. Moreover, an insurer should not use ECDIS or AIS for underwriting or pricing purposes if such use would result in or permit any unfair discrimination or otherwise violate the Insurance Law or any regulations promulgated thereunder."

**Section II, Paragraph 11 — Insurer Retains Responsibility (CRITICAL):**
> "When using ECDIS or AIS as part of their insurance business, insurers are responsible for complying with these anti-discrimination laws irrespective of whether they themselves are collecting data and directly underwriting consumers, or relying on ECDIS or AIS of external vendors that are intended to be partial or full substitutes for direct underwriting or pricing. An insurer may not use ECDIS or AIS to collect or use information that the insurer would otherwise be prohibited from collecting or using directly. **An insurer may not rely solely on a third-party's claim of non-discrimination or a proprietary third-party process to determine compliance with anti-discrimination laws. The responsibility to comply with anti-discrimination laws remains with the insurer at all times.**"

**Section III.A, Paragraph 22 — Senior Management Responsibility:**
> "Senior management is responsible for day-to-day implementation of the insurer's development and management of ECDIS and AIS, consistent with the board's or other governing body's strategic vision and risk appetite."

**Section III.D, Paragraph 37 — Third-Party Vendor Responsibility:**
> "Insurers retain responsibility for understanding any tools, ECDIS, or AIS used in underwriting and pricing for insurance that were developed or deployed by third-party vendors and ensuring such tools, ECDIS, or AIS comply with all applicable laws, rules, and regulations."

**Section IV.E, Paragraph 43 — Proprietary Defense Rejected:**
> "An insurer may not rely on the proprietary nature of a third-party vendor's algorithmic processes to justify the lack of specificity related to an adverse underwriting or pricing action."

### Relevance to Boundary 1

The NY DFS circular letter repeatedly places the compliance responsibility on the **insurer** (meaning human decision-makers), not on the AI system. Paragraph 11's statement that "The responsibility to comply with anti-discrimination laws remains with the insurer at all times" means that an AI system cannot bear the responsibility for a binding underwriting decision — only the human insurer can.

---

## Source 5: Connecticut Insurance Department Bulletin MC-25 (February 26, 2024)

**Document URL:** https://portal.ct.gov/cid/-/media/cid/1_bulletins/bulletin-mc-25.pdf

### Verified Quoted Text

**AIS Program Requirement:**
> "All Insurers authorized to do business in the state are expected to develop, implement, and maintain a written program (an 'AIS Program') for the responsible use of AI Systems that make or support decisions related to regulated insurance practices and products."

**Definition of AI Systems:**
> "AI Systems means machine-based systems designed to simulate human intelligence to perform tasks, such as analysis and decision-making, given a set of human-defined objectives."

**Certification Requirements (Due September 1, 2024 and annually):**
Connecticut requires insurers to certify either:
(a) the insurer is not currently using any "AI Systems"; or
(b) the insurer's use of AI is substantially consistent with the guidance

---

## Summary of Key Regulatory Phrases for Boundary 1

| Source | Key Language | Boundary 1 Implication |
|--------|-------------|----------------------|
| NAIC Model Bulletin | "decisions or actions...made or supported by...AI Systems" | AI is either making or supporting — when supporting, humans make final call |
| NAIC Model Bulletin, Section 3 | "the extent to which humans are involved in the final decision-making process" | Less human involvement = higher risk requiring more controls |
| Colorado CRS 10-3-1104.9(8)(a) | Algorithm = "a computational or machine learning process that **informs human decision making**" | By definition, algorithms inform — they do not substitute for — human decisions |
| NY DFS CL No. 7, Para. 11 | "The responsibility to comply with anti-discrimination laws remains with the insurer at all times" | Compliance responsibility cannot be delegated to AI |
| NY DFS CL No. 7, Para. 43 | Insurer may not rely on "proprietary nature of a third-party vendor's algorithmic processes" | Cannot hide behind AI opacity for underwriting decisions |
| NAIC Model 880, Sec. 4G | "Making or **permitting** any unfair discrimination" | Insurer is liable for permitting AI-driven discrimination |

## Verification Notes

1. **NAIC Model Bulletin full PDF text** — Could not directly render the NAIC PDF document. Quoted text verified through multiple independent legal analyses that directly quote the bulletin, and through search engine snippets. High confidence in accuracy. Verify against PDF at: https://content.naic.org/sites/default/files/inline-files/2023-12-4%20Model%20Bulletin_Adopted_0.pdf

2. **NAIC Model 880 full Section 4 text** — PDF not directly renderable. Key subsection text verified through search engines and law firm analyses. Full text at: https://content.naic.org/sites/default/files/model-law-880.pdf

3. **Colorado CRS 10-3-1104.9** — Full statutory text verified directly from Justia (reliable legal database rendering of 2024 Colorado Revised Statutes). HIGH CONFIDENCE.

4. **NY DFS Circular Letter No. 7** — Full text verified directly from the DFS website. HIGH CONFIDENCE.

5. **Connecticut Bulletin MC-25** — Key excerpts verified. Full PDF at: https://portal.ct.gov/cid/-/media/cid/1_bulletins/bulletin-mc-25.pdf

## States That Have Adopted the NAIC Model Bulletin (as of 2024)

| State | Date Adopted |
|-------|-------------|
| Alaska | February 1, 2024 |
| Connecticut | February 26, 2024 |
| Illinois | March 13, 2024 |
| Kentucky | April 16, 2024 |
| Maryland | April 22, 2024 |
| Nevada | February 23, 2024 |
| New Hampshire | February 20, 2024 |
| Pennsylvania | April 6, 2024 |
| Rhode Island | March 15, 2024 |
| Vermont | March 12, 2024 |
| Washington | April 22, 2024 |
