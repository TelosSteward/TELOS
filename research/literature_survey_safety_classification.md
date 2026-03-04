# TELOS Literature Survey: Safety Classification Beyond Cosine Similarity
## Deep Web Research Report -- Nell Watson, Research Methodologist
## Date: 2026-02-16

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## EXECUTIVE SUMMARY

This survey covers 40+ papers and empirical studies from 2022-2026 addressing the core TELOS problem: cosine similarity measures TOPICAL proximity, not LOGICAL RELATIONSHIP, causing failures when compliant and violating actions share identical vocabulary. The literature provides strong evidence that:

1. **The problem is well-documented**: Multiple 2024-2025 papers confirm that text embeddings are "negation-blind" -- cosine similarity treats "Document the allergies" and "Skip the allergies" as nearly identical.
2. **Cross-encoders empirically outperform bi-encoders on negation-sensitive tasks**: The NevIR benchmark shows bi-encoders achieve ~50% accuracy on negation pairs (random chance) while cross-encoders reach ~75%.
3. **NLI-based classification is the most promising approach**: DeBERTa-v3 NLI models achieve 83%+ F1 on compliance tasks, are ONNX-compatible, and can run in <200ms on CPU.
4. **Few-shot methods work with limited data**: SetFit and FastFit achieve competitive accuracy with 8 examples per class.
5. **The vocabulary-overlap problem in safety contexts is an identified research gap**: No paper directly addresses the specific scenario of policy compliance where violations share identical vocabulary with compliant actions.

---

## SECTION 1: THE SIMILARITY-VS-ENTAILMENT GAP

### Paper 1.1: "Unpacking the Suitcase of Semantic Similarity"

- **CITATION**: OpenReview submission (2024/2025). Available at https://openreview.net/forum?id=OCVIGEitkg
- **WHAT**: Empirically decomposes "semantic similarity" into distinct measurable components and tests how they factor into cosine similarity of text embeddings.
- **METHOD**: Derived analytic expressions for semantic entailment similarity on concept, predicate, and proposition levels. Trained linear projections from embeddings of 15 state-of-the-art embedding models to semantic entailment space. Compared cosine similarity estimates against ground truth.
- **KEY RESULTS**: **The majority of variation in cosine similarity of text embeddings is due to contextual (topical) similarity as opposed to entailment.** The authors propose using "contextual similarity" instead of "semantic similarity" when referring to cosine similarity from embedding models.
- **RELEVANCE TO TELOS**: **HIGH** -- This is the formal articulation of the exact TELOS failure mode. Cosine similarity measures whether two texts are about the same topic, not whether one logically follows from or contradicts the other. "Document the allergies" and "Skip the allergies" score high on contextual similarity but have opposite entailment relationships to an allergy documentation rule.
- **LIMITATIONS**: Benchmark dataset construction methodology not yet fully validated; linear projection assumption may oversimplify.

### Paper 1.2: "Is Cosine-Similarity of Embeddings Really About Similarity?"

- **CITATION**: Steck, H. et al. (Netflix Research). Companion Proceedings of ACM Web Conference 2024; arXiv:2403.05440.
- **WHAT**: Demonstrates that cosine similarity of learned embeddings can yield arbitrary results.
- **METHOD**: Formal mathematical analysis of embedding degrees of freedom and their impact on cosine similarity.
- **KEY RESULTS**: Learned embeddings have degrees of freedom that can render arbitrary cosine-similarities even though unnormalized dot-products are well-defined. Items that should be similar can have very different embeddings, and vice versa.
- **RELEVANCE TO TELOS**: **HIGH** -- Provides formal proof that the metric TELOS relies on (cosine similarity of learned embeddings) is inherently unreliable for semantic judgment. Proposes training models directly with cosine similarity or applying normalization during training.
- **LIMITATIONS**: Analysis focuses on recommendation systems; direct NLP safety implications are inferred.

### Paper 1.3: "Problems with Cosine as a Measure of Embedding Similarity for High Frequency Words"

- **CITATION**: Zhou, K., Ethayarajh, K., Card, D., Jurafsky, D. ACL 2022 (Short Papers). https://aclanthology.org/2022.acl-short.45/
- **WHAT**: Shows cosine similarity systematically underestimates similarity for high-frequency words.
- **METHOD**: Compared BERT embedding cosine similarities against human judgments, controlling for polysemy.
- **KEY RESULTS**: Relative to human judgements, cosine similarity underestimates the similarity of frequent words. Effect traced to training data frequency and representational geometry.
- **RELEVANCE TO TELOS**: **MEDIUM** -- Clinical vocabulary contains high-frequency terms ("document," "check," "review") where this underestimation compounds the entailment-blindness problem.
- **LIMITATIONS**: Focused on word-level, not sentence-level embeddings.

### Paper 1.4: "Magnitude Matters: a Superior Class of Similarity Metrics for Holistic Semantic Understanding"

- **CITATION**: arXiv:2509.19323 (September 2025)
- **WHAT**: Proposes magnitude-aware similarity metrics (Overlap Similarity, Hyperbolic Tangent Similarity) that outperform cosine similarity.
- **METHOD**: Evaluated with 4 sentence embedding models across 8 NLP benchmarks including STS-B, SICK, SNLI, MultiNLI.
- **KEY RESULTS**: Proposed metrics incorporate vector magnitude in a controlled way, which helps better separate broader categories in entailment tasks.
- **RELEVANCE TO TELOS**: **MEDIUM** -- Offers a metric-level improvement but does not fundamentally solve the entailment-vs-similarity distinction. May improve at margins.
- **LIMITATIONS**: Does not address negation specifically; evaluated on standard benchmarks, not safety/compliance tasks.

---

## SECTION 2: NEGATION BLINDNESS IN EMBEDDINGS

### Paper 2.1: "Semantic Adapter for Universal Text Embeddings: Diagnosing and Mitigating Negation Blindness"

- **CITATION**: Cao, H. arXiv:2504.00584 (April 2025). Revised title: "Semantic Adapter for Universal Text Embeddings: Diagnosing and Mitigating Negation Blindness to Enhance Universality"
- **WHAT**: Diagnoses and quantifies the "negation blindness" problem in state-of-the-art text embeddings.
- **METHOD**: Proposed a data-efficient, computation-efficient embedding re-weighting method (negation adapter) that amplifies negation-sensitive dimensions without fine-tuning the embedding model.
- **KEY RESULTS**: Universal text embeddings interpret negated text pairs as roughly similar in meaning. The proposed adapter improves negation awareness by 4.68% for BERT-based embeddings and 3.83% for contextual embeddings on both simple and complex negation understanding tasks.
- **RELEVANCE TO TELOS**: **CRITICAL** -- This paper directly diagnoses the failure TELOS experiences. "Document the allergies" and "Skip the allergies" are treated as semantically similar by current embeddings. However, the 4.68% improvement is modest and may not be sufficient for safety-critical applications.
- **LIMITATIONS**: Improvement is incremental. The adapter approach modifies the metric, not the fundamental representation. Complex negation (negation + antonym) remains challenging. Not evaluated on safety/compliance-specific tasks.

### Paper 2.2: NevIR: Negation in Neural Information Retrieval

- **CITATION**: Weller, O. et al. EACL 2024 (Long Papers). https://aclanthology.org/2024.eacl-long.139/
- **WHAT**: Benchmark that asks IR models to rank two documents differing only by negation.
- **METHOD**: Pairwise accuracy evaluation across lexical, bi-encoder, cross-encoder, late interaction, and transformer models.
- **KEY RESULTS**:
  - **Most IR models (including state-of-the-art) perform at or below random chance (50%) on negation pairs.**
  - Cross-encoders perform best, followed by late-interaction, with bi-encoders and sparse architectures in last place.
  - After fine-tuning on negation data: bi-encoders and late interaction reach ~50%, **cross-encoders reach ~75%**.
  - Negation comprehension does not transfer between different negation scenarios.
- **RELEVANCE TO TELOS**: **CRITICAL** -- This is the most directly relevant benchmark to the TELOS problem. It proves empirically that bi-encoder architectures (what TELOS uses) fundamentally cannot handle negation at scale. Cross-encoders are significantly better but still not perfect.
- **LIMITATIONS**: Evaluated on information retrieval, not classification. 75% accuracy for cross-encoders still leaves a 25% error rate that is unacceptable for safety-critical applications.

### Paper 2.3: "Reproducing NevIR: Negation in Neural Information Retrieval"

- **CITATION**: SIGIR 2025 Proceedings. arXiv:2502.13506.
- **WHAT**: Reproduction study of NevIR adding new model categories.
- **METHOD**: Extended evaluation to listwise LLM re-rankers and additional model architectures.
- **KEY RESULTS**: Listwise LLM re-rankers achieved 20% gains over previous approaches. **Only cross-encoders and listwise LLM re-rankers achieve reasonable performance.** Performance on one negation benchmark does not transfer to other negation scenarios.
- **RELEVANCE TO TELOS**: **HIGH** -- Confirms bi-encoder limitations are robust across reproductions. Shows negation handling does not generalize -- models must be specifically trained for the types of negation they will encounter.
- **LIMITATIONS**: Does not evaluate on safety/compliance domains.

### Paper 2.4: "A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers"

- **CITATION**: Petcu et al. arXiv:2507.22337 (October 2025). University of Amsterdam.
- **WHAT**: Comprehensive taxonomy of negation types and their impact on neural models.
- **METHOD**: Created taxonomy from philosophical, linguistic, and logical definitions. Generated two benchmark datasets. Proposed logic-based classification mechanism.
- **KEY RESULTS**: Dense neural models still underperform on negation queries. The taxonomy produces balanced data distributions leading to faster convergence in training. Studies negation across bi-encoder, cross-encoder, late interaction, NLI, and NTP models.
- **RELEVANCE TO TELOS**: **HIGH** -- Provides the theoretical framework for understanding what types of negation TELOS must handle ("Skip" as lexical negation of "Document" in the context of an obligation).
- **LIMITATIONS**: Very recent; results not yet widely validated.

### Paper 2.5: "Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning"

- **CITATION**: Sun, Y. et al. arXiv:2506.08354 (June 2025).
- **WHAT**: Position paper arguing embeddings must move beyond surface meaning to capture pragmatics, speaker intent, and sociocultural context.
- **METHOD**: Pilot study benchmarking state-of-the-art models on implicit semantics tasks.
- **KEY RESULTS**: Even state-of-the-art models perform only marginally better than simplistic baselines on implicit semantics tasks. Calls for paradigm shift in embedding research.
- **RELEVANCE TO TELOS**: **HIGH** -- The distinction between "Document the allergies" (compliance intent) and "Skip the allergies" (violation intent) is precisely the kind of implicit semantic distinction that current embeddings fail to capture.
- **LIMITATIONS**: Position paper with pilot study, not full empirical evaluation.

---

## SECTION 3: NLI FOR POLICY/SAFETY COMPLIANCE

### Paper 3.1: "Lessons from the Use of Natural Language Inference (NLI) in Requirements Engineering Tasks"

- **CITATION**: Fazelnia, M. et al. IEEE RE 2024. arXiv:2405.05135.
- **WHAT**: Evaluates NLI for requirements classification, defect detection, and conflict detection.
- **METHOD**: Compared NLI with prompt-based models, transfer learning, LLM chatbots, and probabilistic models across zero-shot and conventional learning settings.
- **KEY RESULTS**:
  - NLI achieves **overall F1 of 83%** for requirements classification.
  - **F1 > 80%** for all specification defects.
  - **100% F1** on Uncertain and Directive defect classes.
  - NLI surpassed all other methods including LLMs in requirements specification analysis.
  - **Label verbalization is key**: well-defined, descriptive hypothesis labels significantly improve NLI performance.
- **RELEVANCE TO TELOS**: **HIGH** -- Requirements compliance is structurally similar to TELOS boundary compliance. The 83% F1 with proper label verbalization is a realistic baseline for what NLI could achieve on TELOS boundaries. The label verbalization finding is directly actionable.
- **LIMITATIONS**: Requirements text is more formulaic than clinical conversation; real-world clinical text is noisier.

### Paper 3.2: "Explainable Compliance Detection with Multi-Hop NLI on Assurance Case Structure" (EXCLAIM)

- **CITATION**: arXiv:2506.08713 (July 2025).
- **WHAT**: Formulates compliance detection against GDPR requirements as multi-hop NLI.
- **METHOD**: Uses assurance case claim-argument-evidence structure. Generates synthetic assurance cases with LLMs. Analyzes faithfulness of interpretation methods (gradient, Integrated Gradients, LIME, SHAP) on NLI models.
- **KEY RESULTS**: Demonstrates effectiveness on GDPR compliance with explainable predictions. Limited assurance case data addressed through LLM-generated synthetic data.
- **RELEVANCE TO TELOS**: **HIGH** -- TELOS boundaries are structurally similar to regulatory requirements. The multi-hop approach could handle complex boundaries where a single entailment check is insufficient. Synthetic data generation with LLMs addresses the limited training data problem.
- **LIMITATIONS**: GDPR text is formal/legal; clinical text is informal/conversational. Exact accuracy metrics not publicly available.

### Paper 3.3: "Building Efficient Universal Classifiers with Natural Language Inference"

- **CITATION**: Laurer, M., van Atteveldt, W., Casas, A., Welbers, K. arXiv:2312.17543 (December 2023, updated March 2024).
- **WHAT**: Systematic guide for building NLI-based universal classifiers with empirical evaluation.
- **METHOD**: Trained classifiers on 33 datasets with 389 diverse classes using DeBERTa-v3 and RoBERTa. Mixed NLI and non-NLI training data.
- **KEY RESULTS**:
  - Models trained with mixed NLI + non-NLI data achieve **+9.4% improvement** over NLI-only models.
  - DeBERTa-v3 performs clearly better than RoBERTa (but slower).
  - Smaller BERT-like models can learn universal classification tasks.
  - Enables zero-shot and few-shot classification.
- **RELEVANCE TO TELOS**: **CRITICAL** -- This is the practical foundation for the NLI approach. The DeBERTa-v3 models are already available in ONNX format (protectai/deberta-v3-base-zeroshot-v1-onnx). The architecture is proven, the models are deployable, and the methodology for adapting to new domains is documented.
- **LIMITATIONS**: Zero-shot performance varies by domain; healthcare/clinical compliance not specifically evaluated.

### Paper 3.4: "Bonafide at LegalLens 2024: DeBERTa-Based Encoder for Legal Violation Detection"

- **CITATION**: LegalLens Shared Task, NLLP Workshop 2024. arXiv:2410.22977.
- **WHAT**: Lightweight DeBERTa-v3 encoder for legal violation detection and resolution.
- **METHOD**: DeBERTa-v3-small NLI model fine-tuned on augmented data (312 rows expanded to 936 via Mixtral paraphrases). NLI formulation for matching violations to legal complaints.
- **KEY RESULTS**:
  - NLI system achieved **F1 of 84.73%** on violation resolution.
  - **Data augmentation boosted F1 by 7.65%** (critical finding for limited data scenarios).
  - Lightweight DeBERTa outperformed LLM baselines.
- **RELEVANCE TO TELOS**: **HIGH** -- Demonstrates that NLI with DeBERTa-v3-small works for violation detection with limited data. The 312-row starting dataset is comparable to TELOS's ~240 examples. Data augmentation via paraphrasing provides a concrete path forward.
- **LIMITATIONS**: Legal violations are topically distinct; the vocabulary-overlap problem specific to TELOS is not addressed.

---

## SECTION 4: SAFETY CLASSIFIER BENCHMARKS AND COMPARISONS

### Paper 4.1: "Evaluating the Robustness of LLM Safety Guardrails Against Adversarial Attacks"

- **CITATION**: arXiv:2511.22047 (November 2025).
- **WHAT**: Comprehensive evaluation of 10 guardrail models from Meta, Google, IBM, NVIDIA, Alibaba, Allen AI.
- **METHOD**: 1,445 test prompts spanning 21 attack categories. Separated public benchmark prompts from novel adversarial prompts.
- **KEY RESULTS**:
  - **Qwen3Guard-8B**: 85.3% overall, 91.0% on public prompts, **33.8% on novel attacks** (57.2pp generalization gap)
  - **WildGuard-7B**: 82.8% overall
  - **Granite-Guardian-3.3-8B**: 81.0% overall, **best generalization** (only 6.5% gap)
  - **LlamaGuard models**: 97-99% benign accuracy but **only 4.5-21.8% harmful detection** (safety-usability tradeoff)
  - 2.2x performance gap between best (85.3%) and worst (38.0%) models.
- **RELEVANCE TO TELOS**: **MEDIUM** -- These are general safety classifiers for jailbreaks/toxicity, not policy compliance. However, the generalization gap finding is crucial: TELOS needs a model that generalizes to novel violation patterns, not one overfit to public benchmarks. Granite-Guardian's generalization strength is noteworthy.
- **LIMITATIONS**: Evaluates prompt-level safety (is this prompt harmful?) not policy compliance (does this action violate this specific rule?). Different problem structure from TELOS.

### Paper 4.2: WildGuard

- **CITATION**: Allen AI, 2024. https://huggingface.co/allenai/wildguard
- **WHAT**: 7B safety classifier for prompt harmfulness, response harmfulness, and refusal detection.
- **METHOD**: Mistral-7b-v0.3 base, trained on WildGuardTrain dataset.
- **KEY RESULTS**: Outperforms LlamaGuard2 and Aegis-Guard by up to 25.3% on refusal detection. Matches GPT-4 across tasks, surpasses GPT-4 by 4.8% on adversarial prompt harmfulness.
- **RELEVANCE TO TELOS**: **LOW** -- 7B parameter model is too large for <200ms CPU inference. General safety focus, not policy-specific.
- **LIMITATIONS**: Not designed for custom policy compliance. Too large for edge deployment.

### Paper 4.3: R2-Guard (ICLR 2025)

- **CITATION**: Proceedings ICLR 2025. arXiv:2407.05557.
- **WHAT**: Reasoning-enabled guardrail combining category-specific learning with knowledge-enhanced logical reasoning.
- **METHOD**: Two components: data-driven category classifier + probabilistic graphical model (MLN/PC) for logical inference. Category-specific unsafety probabilities fed into PGM for final prediction.
- **KEY RESULTS**: Surpasses LlamaGuard by **12.6% on standard moderation** and by **59.9% against jailbreaking attacks**. Adapts to new safety categories by editing the PGM reasoning graph.
- **RELEVANCE TO TELOS**: **MEDIUM-HIGH** -- The architecture of combining a classifier with logical reasoning mirrors what TELOS might need: classify the action, then reason about whether it complies with a specific rule. The PGM component for encoding safety rules is conceptually similar to encoding PA boundaries. However, it requires an LLM backbone.
- **LIMITATIONS**: Requires LLM-scale model (8B+). Not tested on fine-grained policy compliance.

### Paper 4.4: GSPR -- Aligning LLM Safeguards as Generalizable Safety Policy Reasoners

- **CITATION**: arXiv:2509.24418 (September 2025).
- **WHAT**: Guardrail that generalizes across different safety taxonomies using GRPO training.
- **METHOD**: Group Relative Policy Optimization with varied safety taxonomies. Cold-start SFT strategy with category rewards.
- **KEY RESULTS**: Improves reasoning capabilities for both safety and category prediction. Demonstrates powerful safety generalization across novel taxonomies. Achieves the least inference token costs with explanations.
- **RELEVANCE TO TELOS**: **MEDIUM** -- The ability to generalize to new taxonomies is relevant because TELOS boundaries are custom per Principal Alignment. However, requires LLM-scale model.
- **LIMITATIONS**: LLM-scale inference cost. Not evaluated on subtle vocabulary-overlap violations.

### Paper 4.5: GuardBench (EMNLP 2024)

- **CITATION**: EMNLP 2024 Main Conference. https://aclanthology.org/2024.emnlp-main.1022/
- **WHAT**: First large-scale standardized benchmark for guardrail models (40 safety evaluation datasets).
- **METHOD**: Automated evaluation pipeline across multiple safety dimensions.
- **KEY RESULTS**: IBM Granite Guardian holds 6 of top 10 spots on leaderboard. General-purpose instruction-following models achieve competitive results without specific fine-tuning.
- **RELEVANCE TO TELOS**: **MEDIUM** -- Provides standardized evaluation methodology. The finding that general-purpose models are competitive suggests TELOS might benefit from a well-prompted general model rather than a specialized safety classifier.
- **LIMITATIONS**: Focused on general safety, not domain-specific compliance.

### Paper 4.6: SG-Bench (NeurIPS 2024)

- **CITATION**: NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2410.21965.
- **WHAT**: Benchmark for LLM safety generalization across diverse tasks and prompt types.
- **KEY RESULTS**:
  - Most LLMs perform worse on **discriminative** tasks than generative ones.
  - LLMs are highly susceptible to prompts, indicating poor safety generalization.
  - Few-shot demonstrations can **induce** harmful responses.
  - Chain-of-thought prompting **harms** safety on discrimination tasks.
- **RELEVANCE TO TELOS**: **MEDIUM** -- The finding that LLMs struggle with discriminative safety tasks (binary: safe/unsafe) supports using specialized classifiers rather than general LLMs for TELOS boundary detection.
- **LIMITATIONS**: Evaluates LLMs, not specialized classifiers.

### Paper 4.7: "Safeguarding Large Language Models: A Survey"

- **CITATION**: Springer AI Review 2025; PMC:12532640.
- **WHAT**: Comprehensive systematic review of LLM safeguarding mechanisms.
- **METHOD**: Systematic literature review covering attack/defense methodologies, evaluation challenges, and implementation landscape.
- **KEY RESULTS**: Taxonomy of safeguarding techniques. Notes that defensive measures have progressed from blacklists to context-aware classifiers. Identifies Llama Guard and NeMo as most widely adopted. Healthcare providers integrating NeMo and TruLens.
- **RELEVANCE TO TELOS**: **MEDIUM** -- Provides context for where TELOS fits in the broader landscape.
- **LIMITATIONS**: Survey breadth sacrifices domain-specific depth.

---

## SECTION 5: CLINICAL NLP AND DOCUMENTATION SAFETY

### Paper 5.1: "A Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation"

- **CITATION**: npj Digital Medicine 2025. Nature. PMC:12075489.
- **WHAT**: Framework for evaluating LLM hallucinations in clinical notes.
- **METHOD**: Error taxonomy (CREOLA framework) for classifying LLM outputs. Clinical safety assessment with clinician annotation.
- **KEY RESULTS**: **1.47% hallucination rate** (44% rated "major") and **3.45% omission rate** (17% "major") in clinically-annotated medical notes. Categories: Hallucination, Inference, Misunderstanding, No Factual Error.
- **RELEVANCE TO TELOS**: **HIGH** -- Directly relevant to TELOS healthcare application. The error taxonomy (hallucination vs. omission vs. misunderstanding) maps to TELOS boundary violation types. The 1.47% hallucination rate and 3.45% omission rate are benchmarks for what TELOS should aim to detect.
- **LIMITATIONS**: Evaluates generation quality, not boundary compliance. Detection methods not specified.

### Paper 5.2: "Benchmarking and Datasets for Ambient Clinical Documentation"

- **CITATION**: medRxiv 2025. doi:10.1101/2025.01.29.25320859.
- **WHAT**: Scoping review of evaluation frameworks for AI-assisted medical note generation.
- **KEY RESULTS**: Wide diversity of evaluation metrics makes cross-study comparison challenging. Critical gaps include limited integration of clinical relevance in automated metrics and lack of standardized approaches for hallucination measurement. ROUGE and BERTScore are common but insufficient.
- **RELEVANCE TO TELOS**: **HIGH** -- Identifies the exact evaluation gap TELOS operates in. No standardized method exists for detecting boundary violations in clinical documentation.
- **LIMITATIONS**: Review, not primary research.

### Paper 5.3: "An Evaluation Framework for Ambient Digital Scribing Tools" (SCRIBE)

- **CITATION**: npj Digital Medicine 2025. Nature.
- **WHAT**: Comprehensive evaluation framework for ambient scribe tools.
- **METHOD**: Framework encompassing transcription accuracy, diarization performance, note quality, simulation-based robustness tests.
- **KEY RESULTS**: Mean percent error across platforms: **26.3%**. Only **35.8%** of correctly reported elements consistently correct across platforms. Average **3.0 errors per case** with potential for moderate-to-severe harm.
- **RELEVANCE TO TELOS**: **CRITICAL** -- These error rates demonstrate the urgent need for TELOS boundary enforcement in clinical documentation. If ambient scribes have 26.3% error rates and 3 harmful errors per case, boundary detection is not theoretical -- it is a patient safety necessity.
- **LIMITATIONS**: Evaluates overall quality, not specific boundary violations.

### Paper 5.4: "Beyond Negation Detection: Comprehensive Assertion Detection Models for Clinical NLP"

- **CITATION**: John Snow Labs, arXiv:2503.17425 (March 2025).
- **WHAT**: State-of-the-art assertion detection for clinical text (Present, Absent, Hypothetical, Conditional, Possible, Associated with Someone Else).
- **METHOD**: Compared fine-tuned LLMs, transformer classifiers, few-shot classifiers, deep learning, GPT-4o, commercial APIs, and NegEx on i2b2 2010 dataset.
- **KEY RESULTS**:
  - Fine-tuned LLM: **accuracy 0.962** (highest)
  - GPT-4o: **0.901**
  - **Few-shot classifier: 0.929** (lightweight, competitive)
  - Fine-tuned LLM excels on Absent assertions (+8.4% vs GPT-4o)
  - Hypothetical detection: +23.4% over GPT-4o
- **RELEVANCE TO TELOS**: **CRITICAL** -- The "Absent" assertion category maps directly to TELOS's negation problem (allergies absent = allergies not documented). The few-shot classifier achieving 0.929 is highly relevant given TELOS's limited training data. The i2b2 assertion framework could inform TELOS boundary labels.
- **LIMITATIONS**: Assertion detection operates on clinical entities, not policy compliance. Needs adaptation to map assertion labels to compliance/violation labels.

---

## SECTION 6: FEW-SHOT AND LIMITED-DATA APPROACHES

### Paper 6.1: SetFit

- **CITATION**: Tunstall, L. et al. Hugging Face, 2022. https://huggingface.co/blog/setfit
- **WHAT**: Sentence Transformer fine-tuning for few-shot classification without prompts.
- **METHOD**: Contrastive learning on sentence pairs to fine-tune a sentence transformer, then trains a classification head.
- **KEY RESULTS**:
  - With **8 labeled examples per class**, competitive with full-dataset fine-tuning (e.g., on Customer Reviews).
  - Outperforms GPT-3 on RAFT benchmark (0.669 vs 0.627).
  - Outperforms human baseline on 7/11 RAFT tasks.
  - 110M parameters (small, deployable).
  - With ModernBERT: **92.7% accuracy with 8 samples** on IMDB (near full-data upper bound of 25k samples).
- **RELEVANCE TO TELOS**: **HIGH** -- With ~30 ground truth examples and ~210 disagreement examples, TELOS has enough data for SetFit. The 8-per-class requirement is achievable. However, SetFit uses contrastive learning on a bi-encoder, which inherits the negation blindness problem.
- **LIMITATIONS**: Bi-encoder backbone means negation sensitivity issues persist. Not tested on safety/compliance tasks specifically.

### Paper 6.2: FastFit (NAACL 2024)

- **CITATION**: IBM Research. NAACL 2024 Demo. arXiv:2404.12365.
- **WHAT**: Fast and effective few-shot classification using batch contrastive learning with token-level similarity.
- **METHOD**: Novel integration of batch contrastive learning and token-level similarity scoring.
- **KEY RESULTS**:
  - **3-20x faster training** than SetFit.
  - FastFit-small comparable to SetFit-large.
  - Consistently outperforms SetFit across 6 languages in 5/10-shot settings.
  - Token-level similarity helps distinguish fine-grained differences.
- **RELEVANCE TO TELOS**: **HIGH** -- Token-level similarity is particularly promising for the TELOS problem because it can potentially attend to the specific tokens that differentiate compliance from violation ("Document" vs "Skip"). The speed improvement is valuable for iterative development.
- **LIMITATIONS**: Not tested on negation-sensitive tasks. Token-level similarity effectiveness on safety/compliance not validated.

### Paper 6.3: "Intent Detection in the Age of LLMs" (EMNLP 2024 Industry)

- **CITATION**: Arora, G., Jain, S., Merugu, S. EMNLP 2024 Industry Track. arXiv:2410.01627. Amazon Science.
- **WHAT**: Compares SetFit, LLMs with adaptive in-context learning, and hybrid approaches for intent detection.
- **METHOD**: Evaluation on real-world task-oriented dialogue systems with overlapping intents, imbalanced data, and multi-label classification.
- **KEY RESULTS**:
  - Hybrid system (uncertainty-based routing + negative data augmentation) achieves **within 2% of native LLM accuracy with 50% less latency**.
  - Two-step approach using internal LLM representations improves OOS detection by **>5% F1**.
  - LLM OOS detection influenced by scope of intent labels and label space size.
- **RELEVANCE TO TELOS**: **MEDIUM-HIGH** -- The hybrid routing approach (use lightweight classifier first, escalate uncertain cases to LLM) is architecturally compatible with TELOS. The negative data augmentation finding is directly relevant.
- **LIMITATIONS**: Intent detection is simpler than policy compliance; no vocabulary-overlap evaluation.

---

## SECTION 7: NEGATION DETECTION IN CLINICAL NLP

### Paper 7.1: LLM-based Negation Detection in Radiology Reports

- **CITATION**: PMC:12092861 (2024/2025).
- **WHAT**: Using LLMs and BERT variants for negation detection in radiology reports.
- **KEY RESULTS**:
  - Rule-based (NegEx): F1 = 0.932
  - BERT-based: F1 = **0.961**
  - RadBERT (domain-specific): most robust performance
- **RELEVANCE TO TELOS**: **MEDIUM** -- Demonstrates that domain-specific fine-tuning significantly improves negation detection. TELOS would benefit from clinical domain adaptation.
- **LIMITATIONS**: Radiology-specific; negation of clinical findings is simpler than negation of compliance actions.

### Paper 7.2: "Improving Negation Detection with Negation-Focused Pre-training"

- **CITATION**: NAACL 2022 Main. arXiv:2205.04012.
- **WHAT**: Targeted data augmentation and negation masking for pre-training.
- **METHOD**: Two strategies: (1) augmenting pre-training data with negation examples, (2) masking negation tokens during pre-training to force the model to learn from context.
- **KEY RESULTS**: Both strategies improve downstream negation handling. Negation-focused pre-training is particularly effective for models that will encounter negation in deployment.
- **RELEVANCE TO TELOS**: **MEDIUM** -- Provides methodology for creating negation-aware models. Could be applied to create a TELOS-specific pre-training augmentation.
- **LIMITATIONS**: Pre-training is expensive. May be overkill if fine-tuning alone is sufficient.

---

## SECTION 8: CONTRASTIVE LEARNING FOR SAFETY

### Paper 8.1: "Improving LLM Safety with Contrastive Representation Learning" (EMNLP 2025)

- **CITATION**: Simko, S., Sachan, M., Scholkopf, B., Jin, Z. EMNLP 2025. arXiv:2506.11938.
- **WHAT**: Using triplet-based contrastive learning with adversarial hard negative mining for LLM safety.
- **METHOD**: Triplet loss that pulls harmful representations together and pushes them away from harmless representations. Adversarial hard negative mining.
- **KEY RESULTS**: Reduces attack success rate on Llama 3 8B from **29% to 5%** against embedding attacks and from **14% to 0%** against REINFORCE-GCG input attacks.
- **RELEVANCE TO TELOS**: **MEDIUM-HIGH** -- The triplet loss approach with hard negative mining is directly applicable to the TELOS problem. Compliance and violation examples sharing the same vocabulary are natural hard negatives. However, this operates at the LLM representation level, not at the classifier level.
- **LIMITATIONS**: Applied to LLM internal representations, not external classifiers. Requires LLM-scale models.

### Paper 8.2: "Composition-Contrastive Learning for Sentence Embeddings" (ACL 2023)

- **CITATION**: Chanchani, S., Huang, R. ACL 2023 Long Papers. https://aclanthology.org/2023.acl-long.882/
- **WHAT**: Contrastive learning that maximizes alignment between texts and compositions of their phrasal constituents.
- **METHOD**: Parameter-efficient, no auxiliary objectives or additional parameters.
- **KEY RESULTS**: Improvements on semantic textual similarity comparable with state-of-the-art.
- **RELEVANCE TO TELOS**: **MEDIUM** -- Compositional understanding is relevant (understanding that "skip the allergies" is composed differently than "document the allergies" at the phrasal level). But not tested on negation or safety tasks.
- **LIMITATIONS**: STS-focused, not safety/compliance.

---

## SECTION 9: SYNTHETIC DATA AND TRAINING DATA STRATEGIES

### Paper 9.1: "Synthetic Data Generation Using Large Language Models" (Survey)

- **CITATION**: arXiv:2503.14023 (March 2025).
- **WHAT**: Comprehensive survey on using LLMs to generate synthetic training data.
- **KEY RESULTS**: Safety moderation identified as important application where synthetic data addresses class imbalance. LLM-generated data addresses ethical risks of human exposure to harmful content. Model performance on synthetic data comparable to real data.
- **RELEVANCE TO TELOS**: **HIGH** -- TELOS can use LLMs to generate synthetic compliance/violation pairs that preserve vocabulary overlap but differ in compliance direction. This directly addresses the limited training data problem.
- **LIMITATIONS**: Synthetic data quality varies. Risk of model hallucinating unrealistic violation patterns.

### Paper 9.2: Bonafide Data Augmentation Results (LegalLens 2024)

- **CITATION**: (See Paper 3.4 above)
- **KEY RESULTS**: Augmenting 312 NLI examples to 936 via paraphrasing boosted F1 by **7.65%**. Used Mixtral 8x7b for paraphrase generation.
- **RELEVANCE TO TELOS**: **CRITICAL** -- This is the closest analog to TELOS's data situation (312 examples --> 936 via augmentation = +7.65% F1). TELOS has ~240 examples. Same augmentation strategy is directly applicable.

---

## SECTION 10: TEXT CLASSIFICATION IN THE LLM ERA

### Paper 10.1: "Text Classification in the LLM Era: Where Do We Stand?"

- **CITATION**: arXiv:2502.11830 (February 2025).
- **WHAT**: Comprehensive comparison of encoder models vs. LLMs for text classification.
- **KEY RESULTS**:
  - **Fine-tuned transformers (BERT, RoBERTa, DeBERTa) remain state-of-the-art** for text classification.
  - In-context learning with LLMs does **not** outperform fine-tuned smaller models, even with advanced prompting.
  - Transformer encoders particularly suited for structured prediction with low inference overhead.
- **RELEVANCE TO TELOS**: **HIGH** -- Validates the approach of using fine-tuned encoder models rather than LLMs for TELOS boundary classification. Encoder models are faster, cheaper, and more accurate for this task type.
- **LIMITATIONS**: General classification benchmarks; safety/compliance domains not specifically evaluated.

### Paper 10.2: "Cost-Aware Model Selection for Text Classification"

- **CITATION**: arXiv:2602.06370 (February 2025).
- **WHAT**: Multi-objective comparison of fine-tuned encoders vs LLM prompting for production deployment.
- **METHOD**: Evaluated BERT, RoBERTa, DistilBERT against GPT-4o and Claude Sonnet across IMDB, SST-2, AG News, DBPedia. Measured accuracy, latency, and cost.
- **KEY RESULTS**: Fine-tuned smaller models achieve **competitive accuracy at 40-200x lower cost** than API calls. Fine-tuning is competitive due to high accuracy and low cost. Encoder models deployed as stateless inference services.
- **RELEVANCE TO TELOS**: **HIGH** -- Directly supports TELOS's constraint of local inference with <200ms latency. Confirms that encoder-based approaches are the right architectural choice.
- **LIMITATIONS**: Standard benchmarks, not safety-specific.

---

## SECTION 11: AVAILABLE MODELS AND DEPLOYMENT

### Model 11.1: cross-encoder/nli-deberta-v3-base

- **SOURCE**: Hugging Face. https://huggingface.co/cross-encoder/nli-deberta-v3-base
- **ARCHITECTURE**: DeBERTa-v3-base cross-encoder trained on SNLI + MultiNLI.
- **OUTPUT**: Entailment / Contradiction / Neutral probabilities.
- **ONNX**: Available as quantized ONNX (model_qint8_avx512_vnni.onnx).
- **RELEVANCE TO TELOS**: **CRITICAL** -- This is likely the most deployable model for TELOS. It's a cross-encoder (handles negation better than bi-encoders per NevIR), uses NLI (directly formulates compliance as entailment), is available in ONNX, and is quantizable for CPU inference.

### Model 11.2: MoritzLaurer/deberta-v3-base-zeroshot-v2.0

- **SOURCE**: Hugging Face. https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0
- **ARCHITECTURE**: DeBERTa-v3-base trained on NLI + 28 diverse classification datasets.
- **KEY ADVANTAGE**: Zero-shot capable; formulates any classification as NLI. +9.4% over NLI-only models.
- **ONNX**: Available via protectai/deberta-v3-base-zeroshot-v1-onnx.
- **RELEVANCE TO TELOS**: **CRITICAL** -- Can be used immediately without fine-tuning by formulating TELOS boundaries as NLI hypotheses. E.g., premise = user action, hypothesis = "This action complies with the allergy documentation requirement."

### Model 11.3: cross-encoder/nli-deberta-v3-xsmall

- **SOURCE**: Hugging Face. https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall
- **ARCHITECTURE**: DeBERTa-v3-xsmall (22M params) cross-encoder.
- **RELEVANCE TO TELOS**: **HIGH** -- Smallest cross-encoder NLI model. Best chance of meeting <200ms CPU latency requirement. May sacrifice some accuracy.

---

## RANKED APPROACHES BY EMPIRICAL EVIDENCE STRENGTH

| Rank | Approach | Evidence Strength | Accuracy Range | Data Needed | TELOS Fit |
|------|----------|-------------------|----------------|-------------|-----------|
| 1 | **NLI Cross-Encoder (DeBERTa-v3)** | Strong (NevIR, LegalLens, RE tasks) | 75-85% F1 | Zero-shot possible; few-shot improves | Excellent |
| 2 | **SetFit/FastFit Few-Shot** | Strong (RAFT, multiple benchmarks) | 85-93% with 8 examples/class | 8-64 examples/class | Good, but bi-encoder backbone |
| 3 | **NLI Zero-Shot (Laurer models)** | Strong (33 datasets, 389 classes) | 70-83% zero-shot | None (zero-shot) | Excellent for bootstrap |
| 4 | **Assertion Detection (Clinical)** | Strong (i2b2 benchmark) | 92.9-96.2% | Domain-specific fine-tuning | High for clinical domain |
| 5 | **Contrastive Learning with Hard Negatives** | Moderate (EMNLP 2025) | Task-specific | Requires violation/compliance pairs | High potential, unproven for compliance |
| 6 | **Hybrid NLI + Routing** | Moderate (EMNLP 2024 Industry) | Within 2% of LLM | Some labeled data | Good architecture fit |
| 7 | **R2-Guard/GSPR Reasoning Guardrails** | Strong (ICLR 2025) | 12-60% over baselines | Substantial training | Too large for TELOS constraints |
| 8 | **Negation Adapter for Embeddings** | Moderate (arXiv 2025) | +3.8-4.7% improvement | Minimal | Quick fix, insufficient alone |
| 9 | **Synthetic Data Augmentation** | Moderate (LegalLens) | +7.65% F1 improvement | Seed data + LLM | Complementary to any approach |
| 10 | **General Safety Classifiers (WildGuard, etc.)** | Strong for general safety | 80-85% general safety | Pre-trained | Wrong problem; too large |

---

## ASSESSMENT: MOST PROMISING APPROACHES FOR TELOS

Given TELOS constraints: **local inference, <200ms latency, limited training data (~240 examples), ONNX compatibility**:

### Tier 1: Implement First (Highest confidence, immediately deployable)

**1. NLI Cross-Encoder with DeBERTa-v3-base or v3-xsmall**

- **Why**: Cross-encoders handle negation significantly better than bi-encoders (NevIR: 75% vs 50%). NLI formulation naturally captures "does this action comply with this rule?" as an entailment question. ONNX-quantized versions exist. The architecture is the single most empirically-validated approach for the exact problem TELOS faces.
- **How**: Formulate each boundary as an NLI hypothesis. E.g., boundary = "Document patient allergies" --> hypothesis = "The action involves properly documenting patient allergies." Test action = "Skip the allergies section" as premise. The model outputs entailment (compliant) or contradiction (violation).
- **Expected performance**: 75-85% accuracy on vocabulary-overlap cases based on NevIR and LegalLens results.
- **Latency**: DeBERTa-v3-xsmall (22M params) quantized ONNX should achieve <100ms on CPU. DeBERTa-v3-base (86M params) quantized may be 100-200ms.

**2. Zero-Shot NLI Bootstrap with Label Verbalization**

- **Why**: The Laurer models (deberta-v3-base-zeroshot-v2.0) achieve +9.4% over NLI-only models and require zero training data. The RE paper shows label verbalization is key. Can be deployed immediately.
- **How**: Convert each TELOS boundary into a well-verbalized NLI hypothesis. Test on the ~30 ground truth examples. Iterate on hypothesis wording.

### Tier 2: Implement Next (Strong evidence, requires some data preparation)

**3. Few-Shot Fine-Tuning with Synthetic Hard Negatives**

- **Why**: SetFit/FastFit achieve competitive accuracy with 8 examples per class. TELOS has ~30 ground truth examples. LegalLens shows that paraphrase augmentation (312 --> 936 rows) provides +7.65% F1.
- **How**: Generate synthetic hard negatives by systematically negating compliance actions. "Document allergies" --> "Skip allergies", "Omit allergies", "Ignore allergies". Use an LLM to generate diverse paraphrases of both compliance and violation actions for each boundary. Fine-tune FastFit or SetFit on the augmented dataset.
- **Critical insight**: The hard negatives must share vocabulary with the positives -- this is the whole point.

**4. Clinical Assertion Detection Adaptation**

- **Why**: The i2b2 assertion detection framework achieves 92.9% with a few-shot classifier on clinical text. The "Absent" category maps to negation of clinical actions.
- **How**: Adapt the assertion detection framework to classify actions as "Present" (compliance demonstrated), "Absent" (compliance negated/omitted), "Hypothetical" (conditional compliance), etc.

### Tier 3: Research Directions (Promising but requires more work)

**5. Contrastive Learning with Hard Negatives (triplet loss)**

- **Why**: EMNLP 2025 shows triplet-based contrastive learning with hard negative mining significantly improves safety representation separation. The TELOS problem naturally generates hard negatives (compliance/violation pairs from the same boundary).
- **Risk**: Not tested on small encoders or policy compliance tasks.

**6. Negation Adapter + Bi-Encoder (Quick Fix)**

- **Why**: If replacing the bi-encoder is too disruptive, the negation adapter from Cao (2025) can provide incremental improvement (+3-5%) without model changes.
- **Risk**: Insufficient alone for safety-critical applications.

---

## IDENTIFIED GAPS IN THE LITERATURE

### Gap 1: Policy Compliance Classification with Vocabulary Overlap
**No paper directly addresses the specific TELOS scenario**: classifying whether an action complies with or violates a specific rule when both the action and the rule use identical vocabulary. The closest work (NevIR, negation blindness papers) addresses this at the information retrieval level, not the policy compliance level. This is a genuine research gap.

### Gap 2: Small-Model Safety Classification for Clinical Domains
Safety classifier research focuses on large models (7B+). There is very limited work on deploying safety classifiers with <100M parameters for domain-specific compliance, particularly in clinical settings. The few-shot clinical assertion work (Paper 5.4) is the closest, but it targets clinical entities, not policy compliance.

### Gap 3: Benchmark for Boundary Violation Detection in AI Governance
No standardized benchmark exists for the TELOS-style problem: given a set of rules (boundaries/principal alignment), classify whether a specific action or piece of generated text violates those rules. The ambient scribe evaluation frameworks (Paper 5.2) note this gap explicitly.

### Gap 4: Cross-Encoder NLI on Safety-Specific Tasks
While cross-encoders demonstrably outperform bi-encoders on negation (NevIR), this has not been validated specifically on safety classification or policy compliance tasks. The transfer from IR to safety is theoretically sound but empirically unconfirmed.

### Gap 5: Few-Shot NLI for Custom Policy Domains
The Laurer models are evaluated on general classification tasks. Their performance on domain-specific policy compliance (where policies use technical/clinical vocabulary and violations are semantically subtle) has not been benchmarked. This gap is directly relevant to TELOS.

### Gap 6: Latency-Constrained Safety Classification
Most safety classifier papers do not report inference latency. The real-world constraint of <200ms on CPU is rarely addressed in the academic literature, which assumes GPU inference.

---

## METHODOLOGY AND DATA QUALITY NOTE

This survey was conducted via systematic web search of Google Scholar, arXiv, ACL Anthology, Hugging Face, PubMed/PMC, and conference proceedings (EMNLP, NAACL, ACL, ICLR, NeurIPS) for the period 2022-2026. Where exact accuracy numbers were available from paper abstracts, model cards, or blog posts, they are cited. Some results are approximated from descriptions when exact figures were behind paywalls. All papers cited are publicly available via the URLs listed.

---

## KEY SOURCES

1. [Unpacking the Suitcase of Semantic Similarity](https://openreview.net/forum?id=OCVIGEitkg) -- OpenReview
2. [Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440) -- Netflix/ACM Web 2024
3. [Problems with Cosine as a Measure of Embedding Similarity](https://aclanthology.org/2022.acl-short.45/) -- ACL 2022
4. [Negation Blindness in Text Embeddings](https://arxiv.org/abs/2504.00584) -- arXiv 2025
5. [NevIR: Negation in Neural Information Retrieval](https://aclanthology.org/2024.eacl-long.139/) -- EACL 2024
6. [Reproducing NevIR](https://arxiv.org/abs/2502.13506) -- SIGIR 2025
7. [Comprehensive Taxonomy of Negation](https://arxiv.org/abs/2507.22337) -- arXiv 2025
8. [Text Embeddings Should Capture Implicit Semantics](https://arxiv.org/abs/2506.08354) -- arXiv 2025
9. [NLI in Requirements Engineering](https://arxiv.org/abs/2405.05135) -- IEEE RE 2024
10. [Explainable Compliance Detection (EXCLAIM)](https://arxiv.org/abs/2506.08713) -- arXiv 2025
11. [Building Efficient Universal Classifiers with NLI](https://arxiv.org/abs/2312.17543) -- Laurer et al. 2024
12. [Bonafide/LegalLens Legal Violation Detection](https://arxiv.org/abs/2410.22977) -- NLLP 2024
13. [Evaluating Robustness of LLM Safety Guardrails](https://arxiv.org/abs/2511.22047) -- arXiv 2025
14. [R2-Guard](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a07e87ecfa8a651d62257571669b0150-Abstract-Conference.html) -- ICLR 2025
15. [GSPR: Generalizable Safety Policy Reasoners](https://arxiv.org/abs/2509.24418) -- arXiv 2025
16. [GuardBench](https://aclanthology.org/2024.emnlp-main.1022/) -- EMNLP 2024
17. [SG-Bench](https://arxiv.org/abs/2410.21965) -- NeurIPS 2024
18. [Clinical Safety and Hallucination Framework](https://www.nature.com/articles/s41746-025-01670-7) -- npj Digital Medicine 2025
19. [SCRIBE Framework for Ambient Scribes](https://www.nature.com/articles/s41746-025-01622-1) -- npj Digital Medicine 2025
20. [Beyond Negation Detection: Assertion Detection for Clinical NLP](https://arxiv.org/abs/2503.17425) -- arXiv 2025
21. [SetFit: Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit) -- Hugging Face 2022
22. [FastFit: Few-Shot Classification](https://aclanthology.org/2024.naacl-demo.18/) -- NAACL 2024
23. [Intent Detection in the Age of LLMs](https://aclanthology.org/2024.emnlp-industry.114/) -- EMNLP 2024 Industry
24. [Improving LLM Safety with Contrastive Representation Learning](https://aclanthology.org/2025.emnlp-main.1430/) -- EMNLP 2025
25. [Safeguarding LLMs: A Survey](https://link.springer.com/article/10.1007/s10462-025-11389-2) -- AI Review 2025
26. [Text Classification in the LLM Era](https://arxiv.org/abs/2502.11830) -- arXiv 2025
27. [Cost-Aware Model Selection for Text Classification](https://arxiv.org/abs/2602.06370) -- arXiv 2025
28. [Magnitude Matters: Superior Similarity Metrics](https://arxiv.org/abs/2509.19323) -- arXiv 2025
29. [WildGuard](https://huggingface.co/allenai/wildguard) -- Allen AI 2024
30. [Qwen3Guard Technical Report](https://arxiv.org/abs/2510.14276) -- arXiv 2025
31. [Granite Guardian Benchmark Results](https://research.ibm.com/blog/granite-guardian-tops-guardbench) -- IBM Research 2025
32. [Patronus AI: Llama Guard is Off Duty](https://www.patronus.ai/blog/llama-guard-is-off-duty) -- Patronus AI 2024
33. [protectai/deberta-v3-base-zeroshot-v1-onnx](https://huggingface.co/protectai/deberta-v3-base-zeroshot-v1-onnx) -- ProtectAI
34. [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) -- Sentence Transformers
35. [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) -- Sentence Transformers
36. [Ambient Scribe Benchmarking Review](https://www.medrxiv.org/content/10.1101/2025.01.29.25320859v1) -- medRxiv 2025
37. [Ambient Scribe Quality and Safety Evaluation](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1691499/full) -- Frontiers AI 2025
38. [Negation-Focused Pre-training](https://arxiv.org/abs/2205.04012) -- NAACL 2022
39. [Composition-Contrastive Learning for Sentence Embeddings](https://aclanthology.org/2023.acl-long.882/) -- ACL 2023
40. [SmartShot: Fine-Tuning Zero-Shot NLI Models](https://medium.com/@igafni21/smartshot-fine-tuning-zero-shot-classification-models-with-nli-a990f5478b4f) -- 2024
