"use strict";
/**
 * TELOS Plugin Configuration
 *
 * Loads plugin configuration from environment variables, config files,
 * and CLI arguments. Determines fail policy and governance preset.
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
exports.loadConfig = loadConfig;
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const fs = __importStar(require("fs"));
const DEFAULT_SOCKET_PATH = path.join(os.homedir(), ".openclaw", "hooks", "telos.sock");
/**
 * Fail policy for each governance preset.
 *
 * strict + balanced: fail-closed (block on daemon unavailability)
 * permissive: fail-open (allow on daemon unavailability)
 */
const PRESET_FAIL_POLICY = {
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
function loadConfig(overrides) {
    // Start with defaults
    const config = {
        socketPath: DEFAULT_SOCKET_PATH,
        preset: "balanced",
        connectionTimeout: 5000,
        scoreTimeout: 500,
        failPolicy: "closed",
        verbose: false,
    };
    // Layer 1: Config file
    const configFilePath = path.join(os.homedir(), ".openclaw", "hooks", "telos.json");
    if (fs.existsSync(configFilePath)) {
        try {
            const fileConfig = JSON.parse(fs.readFileSync(configFilePath, "utf-8"));
            Object.assign(config, fileConfig);
        }
        catch (e) {
            console.warn(`[telos-governance] Failed to read config: ${e}`);
        }
    }
    // Layer 2: Environment variables
    if (process.env.TELOS_SOCKET_PATH) {
        config.socketPath = process.env.TELOS_SOCKET_PATH;
    }
    if (process.env.TELOS_PRESET) {
        config.preset = process.env.TELOS_PRESET;
    }
    if (process.env.TELOS_SCORE_TIMEOUT) {
        config.scoreTimeout = parseInt(process.env.TELOS_SCORE_TIMEOUT, 10);
    }
    if (process.env.TELOS_FAIL_POLICY) {
        config.failPolicy = process.env.TELOS_FAIL_POLICY;
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
//# sourceMappingURL=config.js.map