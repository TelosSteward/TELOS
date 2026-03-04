/**
 * TELOS Plugin Configuration
 *
 * Loads plugin configuration from environment variables, config files,
 * and CLI arguments. Determines fail policy and governance preset.
 */
import { TelosPluginConfig } from "./types";
/**
 * Load plugin configuration.
 *
 * Priority order:
 *   1. Explicit overrides (function args)
 *   2. Environment variables (TELOS_*)
 *   3. Config file (~/.openclaw/hooks/telos.json)
 *   4. Defaults
 */
export declare function loadConfig(overrides?: Partial<TelosPluginConfig>): TelosPluginConfig;
//# sourceMappingURL=config.d.ts.map