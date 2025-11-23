/**
 * TELOS Extension - Content Script
 * Injects governance into web pages (future: ChatGPT, Claude, etc.)
 */

console.log('TELOS Extension content script loaded');

// Future: Intercept ChatGPT/Claude API calls and govern them locally
// For now, this is a placeholder for future functionality

// Listen for messages from page
window.addEventListener('message', (event) => {
    if (event.data.type === 'TELOS_GOVERN_REQUEST') {
        // Forward to background worker
        chrome.runtime.sendMessage({
            type: 'GOVERN_MESSAGE',
            data: event.data.payload
        }, (response) => {
            // Send response back to page
            window.postMessage({
                type: 'TELOS_GOVERN_RESPONSE',
                payload: response
            }, '*');
        });
    }
});

// Inject telemetric signatures library if needed
function injectTelemetricLib() {
    const script = document.createElement('script');
    script.src = chrome.runtime.getURL('lib/telemetric-signatures-mvp.js');
    script.onload = function() {
        console.log('TELOS telemetric signatures loaded');
    };
    (document.head || document.documentElement).appendChild(script);
}

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectTelemetricLib);
} else {
    injectTelemetricLib();
}
