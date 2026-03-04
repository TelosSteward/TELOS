/**
 * TELOS Governance Plugin for OpenClaw
 *
 * Hooks into OpenClaw's before_tool_call event to score every tool call
 * through the TELOS 4-layer governance cascade. Blocks or allows actions
 * based on the governance verdict.
 *
 * Hook lifecycle:
 *   gateway_start  -> Connect to TELOS daemon
 *   session_start  -> Reset action chain
 *   before_tool_call -> Score action, block/allow
 *   message_sending -> Score outbound messages
 *   session_end    -> Log session summary
 *   gateway_stop   -> Disconnect from daemon
 *
 * Integration:
 *   Installed at ~/.openclaw/plugins/telos-governance/
 *   Enabled via: openclaw hooks enable telos-governance
 *   Managed via: telos agent init --detect
 *
 * Regulatory compliance (this is the enforcement point):
 *   - EU AI Act Art. 14: before_tool_call hook implements human oversight
 *     by blocking execution before it runs (SAAI claim TELOS-SAAI-009)
 *   - EU AI Act Art. 72: Continuous scoring of every tool call
 *   - OWASP ASI01 (Agent Goal Hijack): Real-time PA comparison on every action
 *   - OWASP ASI02 (Tool Misuse): 4-layer cascade catches unsafe tool use
 *   See: research/openclaw_regulatory_mapping.md
 */
/**
 * OpenClaw plugin export.
 *
 * Uses the register(api) pattern expected by OpenClaw's plugin loader.
 * The loader calls resolvePluginModuleExport() which looks for
 * .register ?? .activate on the default export object.
 *
 * register() is called SYNCHRONOUSLY by the loader — async setup
 * (like bridge.connect()) is deferred to the gateway_start hook.
 */
declare const _default: {
    id: string;
    name: string;
    version: string;
    description: string;
    register(api: any): void;
};
export default _default;
//# sourceMappingURL=index.d.ts.map