/**
 * TELOS Extension - Popup UI Controller
 */

let currentSession = null;

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
    checkOllamaStatus();
    checkSessionStatus();

    // Button handlers
    document.getElementById('initButton').addEventListener('click', initializeSession);
    document.getElementById('testButton').addEventListener('click', testOllama);
    document.getElementById('sendTest').addEventListener('click', sendTestMessage);
});

/**
 * Check if Ollama is running
 */
async function checkOllamaStatus() {
    chrome.runtime.sendMessage({ type: 'TEST_OLLAMA' }, (response) => {
        const indicator = document.getElementById('ollamaIndicator');
        const status = document.getElementById('ollamaStatus');

        if (response && response.connected) {
            indicator.classList.remove('inactive');
            indicator.classList.add('active');
            status.innerHTML = `<span class="indicator active"></span>Connected`;
            status.classList.add('connected');
            status.classList.remove('disconnected');

            // Show models if available
            if (response.models && response.models.length > 0) {
                const modelNames = response.models.map(m => m.name).join(', ');
                console.log('Available models:', modelNames);
            }
        } else {
            indicator.classList.remove('active');
            indicator.classList.add('inactive');
            status.innerHTML = `<span class="indicator inactive"></span>Disconnected`;
            status.classList.add('disconnected');
            status.classList.remove('connected');
        }
    });
}

/**
 * Check if session exists
 */
function checkSessionStatus() {
    chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
        if (response && response.session) {
            updateSessionUI(response.session);
        }
    });
}

/**
 * Initialize new TELOS session
 */
async function initializeSession() {
    const button = document.getElementById('initButton');
    button.disabled = true;
    button.textContent = 'Initializing...';

    chrome.runtime.sendMessage({
        type: 'INIT_SESSION',
        data: {
            pa: 'Be helpful, accurate, and follow healthcare best practices'
        }
    }, (response) => {
        if (response && response.success) {
            currentSession = {
                id: response.sessionId,
                createdAt: Date.now(),
                turnsProcessed: 0
            };
            updateSessionUI(currentSession);

            button.textContent = 'Session Active';
            setTimeout(() => {
                button.textContent = 'Reinitialize Session';
                button.disabled = false;
            }, 2000);

            // Show test section
            document.getElementById('testSection').style.display = 'block';
        } else {
            button.textContent = 'Error - Try Again';
            button.disabled = false;
            console.error('Init failed:', response?.error);
        }
    });
}

/**
 * Test Ollama connection
 */
function testOllama() {
    const button = document.getElementById('testButton');
    button.disabled = true;
    button.textContent = 'Testing...';

    checkOllamaStatus();

    setTimeout(() => {
        button.textContent = 'Test Ollama';
        button.disabled = false;
    }, 1000);
}

/**
 * Send test message through governance
 */
function sendTestMessage() {
    const message = document.getElementById('testMessage').value.trim();
    if (!message) return;

    const output = document.getElementById('testOutput');
    output.textContent = 'Processing...\n';

    const startTime = Date.now();

    chrome.runtime.sendMessage({
        type: 'GOVERN_MESSAGE',
        data: { userMessage: message }
    }, (response) => {
        const elapsed = Date.now() - startTime;

        if (response && response.success) {
            const metadata = response.metadata;
            output.textContent = `✓ Governed in ${elapsed}ms\n\n`;
            output.textContent += `Tier: ${getTierBadge(metadata.tier)}\n`;
            output.textContent += `Fidelity: ${(metadata.fidelity * 100).toFixed(1)}%\n`;
            output.textContent += `Turn: ${metadata.turnNumber}\n\n`;
            output.textContent += `Response:\n${response.response}\n\n`;
            output.textContent += `Signature: ${metadata.signature.signature.substring(0, 32)}...\n`;

            // Update turns counter
            document.getElementById('turnsProcessed').textContent = metadata.turnNumber;
        } else {
            output.textContent = `✗ Error: ${response?.error || 'Unknown error'}`;
        }
    });
}

/**
 * Update session UI
 */
function updateSessionUI(session) {
    const indicator = document.getElementById('sessionIndicator');
    const status = document.getElementById('sessionStatus');
    const info = document.getElementById('sessionInfo');

    indicator.classList.remove('inactive');
    indicator.classList.add('active');
    status.innerHTML = `<span class="indicator active"></span>Active`;

    document.getElementById('sessionId').textContent = session.id;
    document.getElementById('sessionCreated').textContent = new Date(session.createdAt).toLocaleTimeString();
    document.getElementById('turnsProcessed').textContent = session.turnsProcessed || 0;

    info.style.display = 'block';
}

/**
 * Get tier badge HTML
 */
function getTierBadge(tier) {
    const badges = {
        1: '🟢 Tier 1 (PA Autonomous)',
        2: '🟡 Tier 2 (RAG Enhanced)',
        3: '🔴 Tier 3 (Expert Escalation)'
    };
    return badges[tier] || 'Unknown';
}
