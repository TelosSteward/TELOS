# Open Philanthropy Technical AI Safety RFP - Expression of Interest (Alternative: Control Focus)

**Research Areas:** Control evaluations (*), Jailbreaks and unintentional misalignment (*), Activation monitoring

**Project Title:** TELOS Runtime Monitoring for Control Protocols

## Proposal Summary (300 words)

TELOS provides continuous runtime monitoring for control evaluation protocols. When AI systems are constrained by oversight mechanisms, TELOS detects attempts at subtle sabotage by measuring geometric drift from user-specified purpose in embedding space.

**Motivation:** Control evaluations test whether misaligned AIs can subvert monitoring to achieve harmful goals. Current approaches rely on detecting overt sabotage or policy violations. TELOS offers complementary detection: we catch goal-directed drift before overt sabotage occurs, by measuring when an AI's semantic trajectory diverges from the primacy attractor defined by user intent.

**Technical Approach:** TELOS computes a "basin of attraction" in embedding space representing acceptable AI behavior. The basin center (attractor) is derived from purpose and scope vectors; the radius is computed from governance parameters. Each AI response receives a fidelity score based on its distance from the attractor center. Responses outside the basin trigger interventions proportional to drift magnitude.

**Control Evaluation Applications:**

1. **Red-team detection:** Identify when attack policies subtly pursue misaligned objectives while maintaining surface-level compliance
2. **Blue-team monitoring:** Provide continuous oversight metrics that don't require human evaluation of every output
3. **Collusion detection:** Detect when models coordinate to achieve goals orthogonal to specifications
4. **Competent violations:** Unlike simple refusal detection, TELOS identifies goal-directed behavior patterns across multiple timesteps

**Advantages over baselines:** Input/output monitoring requires understanding the semantic content of each exchange. TELOS operates on geometric properties, detecting drift patterns even in complex agent scenarios where individual actions appear benign.

**Validation plan:** Run control evaluations using TELOS as monitoring layer. Measure: (1) false positive/negative rates against human assessment, (2) detection latency for subtle sabotage attempts, (3) robustness to encoded reasoning, (4) computational overhead vs. LLM-based monitors.

**Requested funding:** $100K-300K, 12 months. Includes API credits for frontier model testing, compute for evaluation infrastructure, and adversarial red-teaming.
