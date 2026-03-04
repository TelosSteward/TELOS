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

import { TelosBridge } from "./bridge";
import { loadConfig } from "./config";
import { OpenClawToolCallEvent, GovernanceVerdict, TelosPluginConfig } from "./types";

let bridge: TelosBridge | null = null;
let config: TelosPluginConfig;
let sessionStats = {
  scored: 0,
  blocked: 0,
  escalated: 0,
  totalLatencyMs: 0,
};

/**
 * Build action text from an OpenClaw tool call event.
 *
 * Extracts the most governance-relevant information from the tool
 * call to give the embedding model the best signal for boundary detection.
 */
function buildActionText(event: OpenClawToolCallEvent): string {
  const parts: string[] = [];

  // Tool description if available
  if (event.action.description) {
    parts.push(event.action.description);
  }

  // Key input fields that affect governance decisions
  const input = event.action.input || {};
  for (const key of ["command", "url", "path", "file_path", "content", "message", "query"]) {
    if (key in input) {
      const val = String(input[key]);
      parts.push(val.length > 200 ? val.substring(0, 200) + "..." : val);
    }
  }

  // Fallback: serialize top-level input keys
  if (parts.length === 0) {
    const summary = Object.entries(input)
      .map(([k, v]) => `${k}: ${String(v).substring(0, 50)}`)
      .join(", ");
    if (summary) parts.push(summary);
  }

  return parts.join(" ") || `${event.action.tool_name} (no description)`;
}

/**
 * Format a block message for the user.
 */

function resolveTelosAgentId(event: OpenClawToolCallEvent): string | undefined {
  const fromInput = (() => {
    const input = (event.action?.input || {}) as Record<string, unknown>;
    const raw = (input.TELOS_AGENT_ID ?? input.agent_id);
    return typeof raw === "string" && raw.trim() ? raw.trim() : "";
  })();
  if (fromInput) return fromInput;

  const contextAgentId = typeof event.context?.agentId === "string"
    ? event.context.agentId.trim()
    : "";
  if (contextAgentId && contextAgentId !== "openclaw" && contextAgentId !== "main") {
    return contextAgentId;
  }

  const sessionKey = typeof event.sessionKey === "string" ? event.sessionKey.trim() : "";
  if (!sessionKey) {
    return contextAgentId || undefined;
  }

  const subagentMatch = /^agent:([^:]+):subagent:(.+)$/.exec(sessionKey);
  if (subagentMatch?.[1]) {
    return subagentMatch[1].trim();
  }

  return contextAgentId || undefined;
}

function formatBlockMessage(verdict: GovernanceVerdict, toolName: string): string {
  const lines = [
    `[TELOS] Action blocked: ${toolName}`,
    `  Decision: ${verdict.decision.toUpperCase()}`,
    `  Fidelity: ${verdict.fidelity.toFixed(3)}`,
    `  Group: ${verdict.tool_group} (${verdict.risk_tier} risk)`,
  ];

  if (verdict.boundary_triggered) {
    lines.push(`  Boundary violation: ${verdict.boundary_violation.toFixed(3)}`);
  }

  if (verdict.explanation) {
    lines.push(`  Reason: ${verdict.explanation.substring(0, 200)}`);
  }

  if (verdict.human_required) {
    lines.push("  Human review required before proceeding.");
  }

  return lines.join("\n");
}

// === Hook Handlers ===

/**
 * gateway_start: Initialize the TELOS bridge when OpenClaw starts.
 */
async function onGatewayStart(): Promise<void> {
  config = loadConfig();

  bridge = new TelosBridge({
    socketPath: config.socketPath,
    preset: config.preset,
    connectionTimeout: config.connectionTimeout,
    scoreTimeout: config.scoreTimeout,
    failPolicy: config.failPolicy,
    verbose: config.verbose,
  });

  try {
    await bridge.connect();
    log("TELOS governance active");
  } catch (e) {
    log(`Failed to connect to TELOS daemon: ${e}`);
    if (config.failPolicy === "closed") {
      log("WARN: fail-closed policy — tool calls will be blocked until daemon is available");
    }
  }
}

/**
 * gateway_stop: Disconnect from the TELOS daemon.
 */
async function onGatewayStop(): Promise<void> {
  if (bridge) {
    bridge.disconnect();
    bridge = null;
  }
  log("TELOS governance disconnected");
}

/**
 * session_start: Reset action chain for a new session.
 */
async function onSessionStart(): Promise<void> {
  if (bridge?.isConnected) {
    try {
      await bridge.resetChain();
    } catch {
      // Non-critical — chain will be implicitly reset
    }
  }

  // Reset session stats
  sessionStats = { scored: 0, blocked: 0, escalated: 0, totalLatencyMs: 0 };
}

/**
 * session_end: Log session governance summary.
 */
async function onSessionEnd(): Promise<void> {
  if (sessionStats.scored > 0) {
    const avgLatency = sessionStats.totalLatencyMs / sessionStats.scored;
    log(
      `Session summary: ${sessionStats.scored} scored, ` +
      `${sessionStats.blocked} blocked, ` +
      `${sessionStats.escalated} escalated, ` +
      `avg latency ${avgLatency.toFixed(1)}ms`
    );
  }
}

/**
 * before_tool_call: Score every tool call through TELOS governance.
 *
 * This is the critical hook — it runs before every tool call and
 * can block execution by returning a modified event or throwing.
 */
async function onBeforeToolCall(
  event: OpenClawToolCallEvent
): Promise<OpenClawToolCallEvent | null> {
  if (!event.action) return event; // Guard: some events lack action
  if (!bridge) {
    // No bridge — apply fail policy
    if (config?.failPolicy === "closed") {
      log(`BLOCKED (no daemon): ${event.action.tool_name}`);
      return null; // Block
    }
    return event; // Allow (fail-open)
  }

  const actionText = buildActionText(event);

  try {
    const telosAgentId = resolveTelosAgentId(event);
    const scoringArgs: Record<string, unknown> = {
      ...(event.action.input as Record<string, unknown>),
      __session_key: event.sessionKey,
    };
    if (telosAgentId) {
      scoringArgs.agent_id = telosAgentId;
      scoringArgs.TELOS_AGENT_ID = telosAgentId;
    }

    const verdict = await bridge.scoreAction(
      event.action.tool_name,
      actionText,
      scoringArgs
    );

    // Update stats
    sessionStats.scored++;
    sessionStats.totalLatencyMs += verdict.latency_ms;

    if (!verdict.allowed) {
      sessionStats.blocked++;
      if (verdict.decision === "escalate") {
        sessionStats.escalated++;
        // Escalation-specific message: show if override was denied/timed out
        const overrideInfo = (verdict as any).override_receipt
          ? ` (override receipt: ${(verdict as any).override_receipt?.payload_hash?.substring(0, 16)}...)`
          : "";
        log(
          `[TELOS] Escalation ${verdict.human_required ? "denied/timed out" : "blocked"}: ` +
          `${event.action.tool_name} (${verdict.risk_tier} risk)${overrideInfo}`
        );
      }
      log(formatBlockMessage(verdict, event.action.tool_name));
      return null; // Block the tool call
    }

    if (verdict.decision === "escalate") {
      sessionStats.escalated++;
      // Override was approved — log the receipt hash
      const receipt = (verdict as any).override_receipt;
      if (receipt) {
        log(
          `[TELOS] Override approved: ${event.action.tool_name} ` +
          `(receipt: ${receipt.payload_hash?.substring(0, 16)}...)`
        );
      }
    }

    // Inject governance context for CLARIFY verdicts (verify-intent signal)
    if (verdict.modified_prompt) {
      event.messages = event.messages || [];
      event.messages.push({
        role: "system",
        content: verdict.modified_prompt,
      });
    }

    if (config.verbose) {
      log(
        `[${verdict.decision.toUpperCase()}] ${event.action.tool_name}: ` +
        `fidelity=${verdict.fidelity.toFixed(3)} ` +
        `(${verdict.latency_ms.toFixed(1)}ms)`
      );
    }

    return event; // Allow the tool call
  } catch (e) {
    log(`Scoring error: ${e}`);
    if (config.failPolicy === "closed") {
      return null; // Block on error
    }
    return event; // Allow on error (fail-open)
  }
}

/**
 * message_sending: Score outbound messages through governance.
 *
 * Catches data exfiltration via messaging channels (ClawHavoc pattern).
 */
async function onMessageSending(event: {
  type: "message_sending";
  action: { channel: string; content: string };
}): Promise<typeof event | null> {
  if (!bridge?.isConnected) return event;

  try {
    const verdict = await bridge.scoreAction(
      "SendMessage",
      `Send message to ${event.action.channel}: ${event.action.content}`,
      { channel: event.action.channel, message: event.action.content }
    );

    if (!verdict.allowed) {
      log(`BLOCKED message to ${event.action.channel}: ${verdict.explanation}`);
      return null;
    }

    return event;
  } catch {
    return config?.failPolicy === "closed" ? null : event;
  }
}

function log(message: string): void {
  console.log(`[telos-governance] ${message}`);
}

// === Export Plugin Registration ===

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
export default {
  id: "telos-governance",
  name: "TELOS Governance",
  version: "1.0.0",
  description: "4-layer governance cascade scoring every tool call through keyword, cosine, SetFit, and LLM layers.",

  register(api: any) {
    api.on("gateway_start", onGatewayStart);
    api.on("gateway_stop", onGatewayStop);
    api.on("session_start", onSessionStart);
    api.on("session_end", onSessionEnd);
    api.on("before_tool_call", async (event: any) => {
      // OpenClaw typed hook: return { block: true, blockReason } to block, or void to allow
      const result = await onBeforeToolCall(event as OpenClawToolCallEvent);
      if (result === null) {
        return { block: true, blockReason: "[TELOS] Action blocked by governance" };
      }
      // Allow — return void (undefined)
    });
    api.on("message_sending", async (event: any) => {
      const result = await onMessageSending(event);
      if (result === null) {
        return { block: true, blockReason: "[TELOS] Message blocked by governance" };
      }
    });

    log("TELOS governance hooks registered via api.on()");
  },
};
