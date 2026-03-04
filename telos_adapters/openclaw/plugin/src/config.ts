/**
 * TELOS Plugin Configuration
 *
 * Loads plugin configuration from environment variables, config files,
 * and CLI arguments. Determines fail policy and governance preset.
 */

import * as path from "path";
import * as os from "os";
import * as fs from "fs";
import { TelosPluginConfig, GovernancePreset } from "./types";

const DEFAULT_SOCKET_PATH = path.join(
  os.homedir(),
  ".openclaw",
  "hooks",
  "telos.sock"
);

/**
 * Fail policy for each governance preset.
 *
 * strict + balanced: fail-closed (block on daemon unavailability)
 * permissive: fail-open (allow on daemon unavailability)
 */
const PRESET_FAIL_POLICY: Record<GovernancePreset, "open" | "closed"> = {
  strict: "closed",
  balanced: "closed",
  permissive: "open",
  custom: "closed",
};

/**
 * Load plugin configuration.
 *
 * Priority order:
 *   1. Explicit overrides (function args)
 *   2. Environment variables (TELOS_*)
 *   3. Config file (~/.openclaw/hooks/telos.json)
 *   4. Defaults
 */
export function loadConfig(
  overrides?: Partial<TelosPluginConfig>
): TelosPluginConfig {
  // Start with defaults
  const config: TelosPluginConfig = {
    socketPath: DEFAULT_SOCKET_PATH,
    preset: "balanced",
    connectionTimeout: 5000,
    scoreTimeout: 500,
    failPolicy: "closed",
    verbose: false,
  };

  // Layer 1: Config file
  const configFilePath = path.join(
    os.homedir(),
    ".openclaw",
    "hooks",
    "telos.json"
  );
  if (fs.existsSync(configFilePath)) {
    try {
      const fileConfig = JSON.parse(
        fs.readFileSync(configFilePath, "utf-8")
      );
      Object.assign(config, fileConfig);
    } catch (e) {
      console.warn(`[telos-governance] Failed to read config: ${e}`);
    }
  }

  // Layer 2: Environment variables
  if (process.env.TELOS_SOCKET_PATH) {
    config.socketPath = process.env.TELOS_SOCKET_PATH;
  }
  if (process.env.TELOS_PRESET) {
    config.preset = process.env.TELOS_PRESET as GovernancePreset;
  }
  if (process.env.TELOS_SCORE_TIMEOUT) {
    config.scoreTimeout = parseInt(process.env.TELOS_SCORE_TIMEOUT, 10);
  }
  if (process.env.TELOS_FAIL_POLICY) {
    config.failPolicy = process.env.TELOS_FAIL_POLICY as "open" | "closed";
  }
  if (process.env.TELOS_VERBOSE === "1" || process.env.TELOS_VERBOSE === "true") {
    config.verbose = true;
  }

  // Layer 3: Explicit overrides
  if (overrides) {
    Object.assign(config, overrides);
  }

  // Derive fail policy from preset if not explicitly set
  if (!process.env.TELOS_FAIL_POLICY && !overrides?.failPolicy) {
    config.failPolicy = PRESET_FAIL_POLICY[config.preset] || "closed";
  }

  return config;
}
