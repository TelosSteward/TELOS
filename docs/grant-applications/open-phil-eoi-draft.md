# Open Philanthropy Technical AI Safety RFP - Expression of Interest

**Research Areas:** Applications of white-box techniques (*), White-box estimation of rare misbehavior (*), Control evaluations (*)

**Project Title:** TELOS: Runtime Governance via Geometric Alignment in Embedding Spaces

## Proposal Summary (300 words)

We propose to develop and validate TELOS (Telically Entrained Linguistic Operational Substrate), a mathematically rigorous runtime governance system that monitors AI behavior through geometric properties of embedding spaces. TELOS addresses three starred research priorities in your RFP.

**Core Innovation:** TELOS treats alignment as a geometric problem. By representing user intent and AI responses as vectors in embedding space, we compute a "primacy attractor" — a mathematically defined center point with a basin of attraction. AI responses are continuously measured against this attractor using distance metrics, enabling real-time detection of semantic drift with O(n) complexity. This provides quantifiable, continuous measurement of alignment fidelity during live interactions.

**Applications to RFP Priorities:**

*White-box techniques:* TELOS leverages activation space geometry to monitor behavior. Our functional tests demonstrate zero-error (0.0000000000) mathematical formula implementation for computing attractor centers and basin radii. Unlike black-box methods, TELOS uses principled geometric properties of the embedding space itself.

*Low-probability estimation:* TELOS's basin radius calculations enable estimation of rare misbehavior probability without exhaustive input-space search. By modeling the activation distribution geometrically, we can identify when responses approach basin boundaries — precisely where dangerous behaviors become likely.

*Control evaluations:* TELOS serves as a runtime monitor in red-team/blue-team scenarios. It can detect when models pursue goals orthogonal to user intent, even when those goals are subtly encoded. The system provides continuous oversight without requiring human evaluation of every output.

**Current Status:** Functional prototype with validated mathematics, open-source codebase, and passing validation tests. We seek funding to: (1) scale to frontier models, (2) conduct adversarial evaluations against jailbreaking attempts, (3) benchmark against black-box baselines, and (4) develop theoretical guarantees for superintelligent systems.

**PI:** [Your name and credentials]
**Institution:** TELOS Labs
**Requested funding:** $50K-500K (discrete project, 12-18 months)
