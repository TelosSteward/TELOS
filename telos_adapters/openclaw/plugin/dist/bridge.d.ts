/**
 * TELOS Bridge — Unix Domain Socket client for the Python governance daemon.
 *
 * Communicates with the TELOS governance process via NDJSON over UDS.
 * Handles connection management, message framing, and timeout/retry.
 *
 * Design decisions (Karpathy M0 systems analysis):
 *   - UDS over HTTP: 0.05-0.2ms vs 1-5ms round-trip
 *   - NDJSON: Debuggable with socat/jq, <1KB payloads
 *   - Connection pooling: Keep-alive socket, reconnect on failure
 *   - Fail policy: Configurable (fail-open for permissive, fail-closed for strict)
 *
 * Regulatory traceability:
 *   - SAAI claim TELOS-SAAI-012: Fail policy determines behavior when daemon
 *     is unreachable — fail-closed (balanced/strict) blocks execution,
 *     fail-open (permissive) allows with logging
 *   - OWASP ASI07: UDS provides authenticated local-only communication
 *   - OWASP ASI08 (Cascading Failures): Connection timeout + fail policy
 *     prevents governance gap from cascading
 *   See: research/openclaw_regulatory_mapping.md §3, §6
 */
import { GovernanceVerdict, TelosPluginConfig } from "./types";
export declare class TelosBridge {
    private config;
    private socket;
    private connected;
    private pendingRequests;
    private buffer;
    constructor(config?: Partial<TelosPluginConfig>);
    /**
     * Connect to the TELOS governance daemon.
     * Establishes a persistent UDS connection.
     */
    connect(): Promise<void>;
    /**
     * Disconnect from the daemon.
     */
    disconnect(): void;
    /**
     * Score a tool call through the TELOS governance engine.
     *
     * @param toolName - The OpenClaw tool name (e.g., "Bash", "Read")
     * @param actionText - The action description or command text
     * @param args - Optional tool arguments
     * @returns GovernanceVerdict with decision and scoring breakdown
     */
    scoreAction(toolName: string, actionText: string, args?: Record<string, unknown>): Promise<GovernanceVerdict>;
    /**
     * Check daemon health.
     */
    health(): Promise<Record<string, unknown>>;
    /**
     * Reset the action chain (new task/session).
     */
    resetChain(): Promise<void>;
    /**
     * Acknowledge a SAAI drift BLOCK state.
     * Resets drift to NORMAL, preserving baseline. Limited to 2 per session.
     *
     * @param reason - Free-text reason for the acknowledgment
     */
    acknowledgeDrift(reason?: string): Promise<Record<string, unknown>>;
    /**
     * Get full drift tracker status and history.
     */
    getDriftStatus(): Promise<Record<string, unknown>>;
    /**
     * Whether the bridge is connected to the daemon.
     */
    get isConnected(): boolean;
    private send;
    private handleData;
    private failPolicyResponse;
    private makeAllowResponse;
    private nextRequestId;
    private log;
}
//# sourceMappingURL=bridge.d.ts.map