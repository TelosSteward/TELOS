"use strict";
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
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.TelosBridge = void 0;
const net = __importStar(require("net"));
const crypto = __importStar(require("crypto"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const DEFAULT_SOCKET_PATH = path.join(os.homedir(), ".openclaw", "hooks", "telos.sock");
const DEFAULT_CONFIG = {
    socketPath: DEFAULT_SOCKET_PATH,
    preset: "balanced",
    connectionTimeout: 5000, // 5s connection timeout
    scoreTimeout: 500, // 500ms scoring timeout (generous, engine is ~15ms)
    failPolicy: "closed", // Default: block on communication failure
    verbose: false,
};
class TelosBridge {
    constructor(config) {
        this.socket = null;
        this.connected = false;
        this.pendingRequests = new Map();
        this.buffer = "";
        this.config = { ...DEFAULT_CONFIG, ...config };
    }
    /**
     * Connect to the TELOS governance daemon.
     * Establishes a persistent UDS connection.
     */
    async connect() {
        if (this.connected)
            return;
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error(`Connection timeout (${this.config.connectionTimeout}ms) — ` +
                    `is the TELOS daemon running? Start with: telos service start`));
            }, this.config.connectionTimeout);
            this.socket = net.createConnection(this.config.socketPath, () => {
                clearTimeout(timeoutId);
                this.connected = true;
                this.log("Connected to TELOS governance daemon");
                resolve();
            });
            this.socket.on("data", (data) => {
                this.handleData(data);
            });
            this.socket.on("error", (err) => {
                clearTimeout(timeoutId);
                this.connected = false;
                this.log(`Socket error: ${err.message}`);
                // Reject all pending requests
                for (const [id, pending] of this.pendingRequests) {
                    pending.reject(err);
                    clearTimeout(pending.timeout);
                }
                this.pendingRequests.clear();
                reject(err);
            });
            this.socket.on("close", () => {
                this.connected = false;
                this.log("Connection closed");
            });
        });
    }
    /**
     * Disconnect from the daemon.
     */
    disconnect() {
        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
        this.buffer = "";
        this.pendingRequests.clear();
    }
    /**
     * Score a tool call through the TELOS governance engine.
     *
     * @param toolName - The OpenClaw tool name (e.g., "Bash", "Read")
     * @param actionText - The action description or command text
     * @param args - Optional tool arguments
     * @returns GovernanceVerdict with decision and scoring breakdown
     */
    async scoreAction(toolName, actionText, args) {
        const msg = {
            type: "score",
            request_id: this.nextRequestId(),
            tool_name: toolName,
            action_text: actionText,
            args: args || {},
            timestamp: Date.now() / 1000,
        };
        const response = await this.send(msg);
        if (response.type === "verdict") {
            return response.data;
        }
        if (response.type === "error") {
            throw new Error(`Governance error: ${response.error}`);
        }
        throw new Error(`Unexpected response type: ${response.type}`);
    }
    /**
     * Check daemon health.
     */
    async health() {
        const msg = {
            type: "health",
            request_id: this.nextRequestId(),
        };
        const response = await this.send(msg);
        if (response.type === "health") {
            return response.data;
        }
        throw new Error(`Health check failed: ${JSON.stringify(response)}`);
    }
    /**
     * Reset the action chain (new task/session).
     */
    async resetChain() {
        const msg = {
            type: "reset_chain",
            request_id: this.nextRequestId(),
        };
        await this.send(msg);
    }
    /**
     * Acknowledge a SAAI drift BLOCK state.
     * Resets drift to NORMAL, preserving baseline. Limited to 2 per session.
     *
     * @param reason - Free-text reason for the acknowledgment
     */
    async acknowledgeDrift(reason) {
        const msg = {
            type: "acknowledge_drift",
            request_id: this.nextRequestId(),
            args: { reason: reason || "" },
        };
        const response = await this.send(msg);
        if (response.type === "ack") {
            return response.data;
        }
        throw new Error(`Acknowledge drift failed: ${JSON.stringify(response)}`);
    }
    /**
     * Get full drift tracker status and history.
     */
    async getDriftStatus() {
        const msg = {
            type: "get_drift_status",
            request_id: this.nextRequestId(),
        };
        const response = await this.send(msg);
        if (response.type === "drift_status") {
            return response.data;
        }
        throw new Error(`Get drift status failed: ${JSON.stringify(response)}`);
    }
    /**
     * Whether the bridge is connected to the daemon.
     */
    get isConnected() {
        return this.connected;
    }
    // --- Private methods ---
    async send(msg) {
        if (!this.connected) {
            try {
                await this.connect();
            }
            catch {
                return this.failPolicyResponse(msg);
            }
        }
        return new Promise((resolve, reject) => {
            const requestId = "request_id" in msg ? msg.request_id : "";
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                if (this.config.failPolicy === "open") {
                    resolve(this.makeAllowResponse(requestId));
                }
                else {
                    reject(new Error(`Score timeout (${this.config.scoreTimeout}ms)`));
                }
            }, this.config.scoreTimeout);
            this.pendingRequests.set(requestId, { resolve, reject, timeout });
            const line = JSON.stringify(msg) + "\n";
            this.socket.write(line, (err) => {
                if (err) {
                    this.pendingRequests.delete(requestId);
                    clearTimeout(timeout);
                    if (this.config.failPolicy === "open") {
                        resolve(this.makeAllowResponse(requestId));
                    }
                    else {
                        reject(err);
                    }
                }
            });
        });
    }
    handleData(data) {
        this.buffer += data.toString();
        // Process complete NDJSON lines
        let newlineIndex;
        while ((newlineIndex = this.buffer.indexOf("\n")) !== -1) {
            const line = this.buffer.substring(0, newlineIndex).trim();
            this.buffer = this.buffer.substring(newlineIndex + 1);
            if (!line)
                continue;
            try {
                const response = JSON.parse(line);
                const requestId = response.request_id;
                // Handle escalation_pending: extend the timeout and keep waiting
                if (response.type === "escalation_pending") {
                    const pending = this.pendingRequests.get(requestId);
                    if (pending) {
                        // Clear the short scoring timeout
                        clearTimeout(pending.timeout);
                        // Set a new longer timeout for human review
                        const escalationData = response.data;
                        const escalationTimeout = (escalationData.timeout_seconds || 300) * 1000;
                        const newTimeout = setTimeout(() => {
                            this.pendingRequests.delete(requestId);
                            if (this.config.failPolicy === "open") {
                                pending.resolve(this.makeAllowResponse(requestId));
                            }
                            else {
                                pending.reject(new Error(`Escalation timeout (${escalationData.timeout_seconds}s)`));
                            }
                        }, escalationTimeout);
                        // Update the pending entry with the new timeout
                        this.pendingRequests.set(requestId, {
                            ...pending,
                            timeout: newTimeout,
                        });
                        this.log(`Escalation pending for ${response.data?.tool_name || "unknown"} ` +
                            `— waiting up to ${escalationData.timeout_seconds}s for human review`);
                    }
                    continue; // Don't resolve — wait for the final verdict
                }
                const pending = this.pendingRequests.get(requestId);
                if (pending) {
                    clearTimeout(pending.timeout);
                    this.pendingRequests.delete(requestId);
                    pending.resolve(response);
                }
                else {
                    this.log(`Unmatched response for request ${requestId}`);
                }
            }
            catch (e) {
                this.log(`Failed to parse response: ${line}`);
            }
        }
    }
    failPolicyResponse(msg) {
        const requestId = "request_id" in msg ? msg.request_id : "";
        if (this.config.failPolicy === "open") {
            return this.makeAllowResponse(requestId);
        }
        return {
            type: "error",
            request_id: requestId,
            error: "TELOS daemon unavailable and fail-policy is closed",
        };
    }
    makeAllowResponse(requestId) {
        return {
            type: "verdict",
            request_id: requestId,
            data: {
                allowed: true,
                decision: "execute",
                fidelity: 0,
                tool_group: "unknown",
                telos_tool_name: "unknown",
                risk_tier: "low",
                is_cross_group: false,
                purpose_fidelity: 0,
                scope_fidelity: 0,
                boundary_violation: 0,
                tool_fidelity: 0,
                chain_continuity: 0,
                boundary_triggered: false,
                human_required: false,
                latency_ms: 0,
                cascade_layers: [],
                explanation: "TELOS daemon unavailable — fail-open policy applied",
                governance_preset: this.config.preset,
                cusum_alert: false,
                cusum_tool_group: "",
                drift_level: "NORMAL",
                drift_magnitude: 0,
                baseline_fidelity: null,
                baseline_established: false,
                is_blocked: false,
                is_restricted: false,
                turn_count: 0,
                acknowledgment_count: 0,
                permanently_blocked: false,
                modified_prompt: "",
                verdict_signature: "",
                public_key: "",
            },
            latency_ms: 0,
        };
    }
    nextRequestId() {
        return crypto.randomUUID();
    }
    log(message) {
        if (this.config.verbose) {
            console.log(`[telos-governance] ${message}`);
        }
    }
}
exports.TelosBridge = TelosBridge;
//# sourceMappingURL=bridge.js.map