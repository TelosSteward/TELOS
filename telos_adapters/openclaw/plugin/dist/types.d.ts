/**
 * TELOS Governance Types for OpenClaw Plugin
 *
 * Type definitions for the IPC protocol between the TypeScript plugin
 * and the Python governance daemon.
 */
/** Governance decisions from the TELOS engine */
export type GovernanceDecision = "execute" | "clarify" | "suggest" | "inert" | "escalate";
/** Risk tier for tool groups */
export type RiskTier = "critical" | "high" | "medium" | "low";
/** Governance presets */
export type GovernancePreset = "strict" | "balanced" | "permissive" | "custom";
/** Message types sent to the Python daemon */
export interface IPCScoreMessage {
    type: "score";
    request_id: string;
    tool_name: string;
    action_text: string;
    args?: Record<string, unknown>;
    timestamp: number;
}
export interface IPCHealthMessage {
    type: "health";
    request_id: string;
}
export interface IPCResetChainMessage {
    type: "reset_chain";
    request_id: string;
}
export interface IPCShutdownMessage {
    type: "shutdown";
    request_id: string;
}
export interface IPCAcknowledgeDriftMessage {
    type: "acknowledge_drift";
    request_id: string;
    args?: {
        reason?: string;
    };
}
export interface IPCGetDriftStatusMessage {
    type: "get_drift_status";
    request_id: string;
}
export interface IPCGateTransitionMessage {
    type: "gate_transition";
    request_id: string;
    args?: {
        gate_file_path?: string;
    };
}
export type IPCMessage = IPCScoreMessage | IPCHealthMessage | IPCResetChainMessage | IPCShutdownMessage | IPCAcknowledgeDriftMessage | IPCGetDriftStatusMessage | IPCGateTransitionMessage;
/** Governance verdict returned by the Python daemon */
export interface GovernanceVerdict {
    allowed: boolean;
    decision: GovernanceDecision;
    fidelity: number;
    tool_group: string;
    telos_tool_name: string;
    risk_tier: RiskTier;
    is_cross_group: boolean;
    purpose_fidelity: number;
    scope_fidelity: number;
    boundary_violation: number;
    tool_fidelity: number;
    chain_continuity: number;
    boundary_triggered: boolean;
    human_required: boolean;
    latency_ms: number;
    cascade_layers: string[];
    explanation: string;
    governance_preset: GovernancePreset;
    cusum_alert: boolean;
    cusum_tool_group: string;
    drift_level: "NORMAL" | "WARNING" | "RESTRICT" | "BLOCK";
    drift_magnitude: number;
    baseline_fidelity: number | null;
    baseline_established: boolean;
    is_blocked: boolean;
    is_restricted: boolean;
    turn_count: number;
    acknowledgment_count: number;
    permanently_blocked: boolean;
    modified_prompt: string;
    verdict_signature: string;
    public_key: string;
    gate_mode: "" | "enforce" | "observe";
    observe_shadow_decision: string;
    observe_shadow_allowed: boolean;
    policy_violation?: boolean;
    policy_reason?: string;
    policy_collection?: string;
    policy_access_level?: string;
}
/** IPC response from the Python daemon */
export interface IPCVerdictResponse {
    type: "verdict";
    request_id: string;
    data: GovernanceVerdict;
    latency_ms: number;
}
export interface IPCHealthResponse {
    type: "health";
    request_id: string;
    data: {
        status: string;
        governance_stats: Record<string, number>;
    };
}
export interface IPCErrorResponse {
    type: "error";
    request_id: string;
    error: string;
}
export interface IPCAckResponse {
    type: "ack";
    request_id: string;
    data: {
        message: string;
    };
}
/** Interim response: escalation is pending human review */
export interface IPCEscalationPendingResponse {
    type: "escalation_pending";
    request_id: string;
    data: {
        tool_name: string;
        risk_tier: RiskTier;
        timeout_seconds: number;
    };
}
export interface IPCDriftStatusResponse {
    type: "drift_status";
    request_id: string;
    data: {
        all_fidelity_scores: number[];
        post_baseline_scores: number[];
        baseline_fidelity: number | null;
        baseline_established: boolean;
        current_drift_level: string;
        current_drift_magnitude: number;
        acknowledgment_count: number;
        permanently_blocked: boolean;
        total_turns: number;
    };
}
export type IPCResponse = IPCVerdictResponse | IPCHealthResponse | IPCErrorResponse | IPCAckResponse | IPCEscalationPendingResponse | IPCDriftStatusResponse;
/**
 * OpenClaw hook event interface.
 *
 * This matches OpenClaw's InternalHookEvent type for the
 * before_tool_call hook. The actual OpenClaw SDK types would
 * be imported from @openclaw/plugin-sdk in production.
 */
export interface OpenClawToolCallEvent {
    type: "before_tool_call";
    action: {
        tool_name: string;
        input: Record<string, unknown>;
        description?: string;
    };
    sessionKey: string;
    context: {
        agentId: string;
        workspacePath?: string;
        sandboxMode?: string;
    };
    timestamp: number;
    messages?: Array<{
        role: string;
        content: string;
    }>;
}
/** Plugin configuration */
export interface TelosPluginConfig {
    socketPath: string;
    preset: GovernancePreset;
    connectionTimeout: number;
    scoreTimeout: number;
    failPolicy: "open" | "closed";
    verbose: boolean;
}
//# sourceMappingURL=types.d.ts.map