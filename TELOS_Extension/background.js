/**
 * TELOS Extension - Background Service Worker
 * Handles Ollama communication and governance coordination
 */

const OLLAMA_BASE_URL = 'http://localhost:11434';
const MISTRAL_MODEL = 'mistral:latest';

// Global state
let currentSession = null;
let telemetricSigner = null;

// Initialize extension
chrome.runtime.onInstalled.addListener(() => {
    console.log('TELOS Extension installed');

    // Set default settings
    chrome.storage.local.set({
        ollamaUrl: OLLAMA_BASE_URL,
        model: MISTRAL_MODEL,
        tier1Threshold: 0.18,
        tier2Threshold: 0.12,
        enableTelemetricSignatures: true,
        enableLocalGovernance: true
    });
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Background received message:', request.type);

    switch(request.type) {
        case 'INIT_SESSION':
            handleInitSession(request.data, sendResponse);
            return true; // Async response

        case 'GOVERN_MESSAGE':
            handleGovernMessage(request.data, sendResponse);
            return true; // Async response

        case 'GET_STATUS':
            handleGetStatus(sendResponse);
            return true;

        case 'TEST_OLLAMA':
            testOllamaConnection(sendResponse);
            return true;

        default:
            sendResponse({ error: 'Unknown message type' });
    }
});

/**
 * Initialize a new TELOS session
 */
async function handleInitSession(data, sendResponse) {
    try {
        // Generate session ID
        const sessionId = `telos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        // Initialize telemetric signer (load from lib)
        // Note: In production, would import TelemetricSignatureMVP
        // For now, creating minimal version
        telemetricSigner = {
            sessionId: sessionId,
            turnNumber: 0,
            signDelta: async (delta) => ({
                signature: `sig_${Date.now()}`,
                turn_number: ++telemetricSigner.turnNumber,
                timestamp: Date.now()
            })
        };

        currentSession = {
            id: sessionId,
            createdAt: Date.now(),
            turnsProcessed: 0,
            pa: data.pa || null
        };

        console.log('Session initialized:', sessionId);
        sendResponse({
            success: true,
            sessionId: sessionId,
            message: 'TELOS session initialized'
        });
    } catch (error) {
        console.error('Init session error:', error);
        sendResponse({ error: error.message });
    }
}

/**
 * Govern a user message with TELOS
 */
async function handleGovernMessage(data, sendResponse) {
    try {
        if (!currentSession) {
            throw new Error('No active session. Initialize first.');
        }

        const startTime = Date.now();

        // Step 1: Calculate fidelity to PA (mock for MVP)
        const fidelity = await calculateFidelity(data.userMessage, currentSession.pa);

        // Step 2: Determine tier
        const settings = await chrome.storage.local.get(['tier1Threshold', 'tier2Threshold']);
        const tier = determineTier(fidelity, settings.tier1Threshold, settings.tier2Threshold);

        // Step 3: Generate response based on tier
        let response, responseMetadata;

        if (tier === 1) {
            // PA autonomous response
            response = await generatePAResponse(data.userMessage, currentSession.pa);
            responseMetadata = { tier: 1, method: 'PA_autonomous' };
        } else if (tier === 2) {
            // RAG-enhanced response
            response = await generateRAGResponse(data.userMessage);
            responseMetadata = { tier: 2, method: 'RAG_enhanced' };
        } else {
            // Expert escalation (for demo, just use Ollama)
            response = await generateExpertResponse(data.userMessage);
            responseMetadata = { tier: 3, method: 'Expert_reviewed' };
        }

        const deltaTime = Date.now() - startTime;

        // Step 4: Create governance delta
        const delta = {
            session_id: currentSession.id,
            turn_number: currentSession.turnsProcessed + 1,
            delta_t_ms: deltaTime,
            fidelity: fidelity,
            tier: tier,
            user_message: data.userMessage,
            response: response,
            timestamp: Date.now()
        };

        // Step 5: Sign delta with telemetric signature
        const signature = await telemetricSigner.signDelta(delta);

        currentSession.turnsProcessed++;

        sendResponse({
            success: true,
            response: response,
            metadata: {
                ...responseMetadata,
                fidelity: fidelity,
                deltaTime: deltaTime,
                signature: signature,
                sessionId: currentSession.id,
                turnNumber: currentSession.turnsProcessed
            }
        });

    } catch (error) {
        console.error('Govern message error:', error);
        sendResponse({ error: error.message });
    }
}

/**
 * Calculate fidelity to PA (mock - uses simple similarity)
 */
async function calculateFidelity(message, pa) {
    // In production, would use Ollama embeddings
    // For MVP, return mock value
    return 0.15 + Math.random() * 0.15; // 0.15-0.30 range
}

/**
 * Determine governance tier based on fidelity
 */
function determineTier(fidelity, tier1Threshold, tier2Threshold) {
    if (fidelity >= tier1Threshold) return 1; // PA autonomous
    if (fidelity >= tier2Threshold) return 2; // RAG enhanced
    return 3; // Expert escalation
}

/**
 * Generate PA autonomous response
 */
async function generatePAResponse(message, pa) {
    const response = await callOllama({
        model: MISTRAL_MODEL,
        prompt: `You are responding aligned with this principle: "${pa || 'Be helpful and informative'}"\n\nUser: ${message}\n\nAssistant:`,
        stream: false
    });

    return response.response;
}

/**
 * Generate RAG-enhanced response
 */
async function generateRAGResponse(message) {
    // Mock RAG - in production would query corpus
    return await generatePAResponse(message, "Provide helpful, accurate information");
}

/**
 * Generate expert-reviewed response
 */
async function generateExpertResponse(message) {
    // Mock expert - in production would have human review
    return await generatePAResponse(message, "Provide careful, well-considered response");
}

/**
 * Call Ollama API
 */
async function callOllama(payload) {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        throw new Error(`Ollama error: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * Test Ollama connection
 */
async function testOllamaConnection(sendResponse) {
    try {
        const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
        const data = await response.json();

        sendResponse({
            success: true,
            connected: true,
            models: data.models || []
        });
    } catch (error) {
        sendResponse({
            success: false,
            connected: false,
            error: error.message
        });
    }
}

/**
 * Get current status
 */
function handleGetStatus(sendResponse) {
    sendResponse({
        session: currentSession,
        telemetricEnabled: telemetricSigner !== null
    });
}
