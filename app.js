// Arya Rachel Friday - GUI JavaScript
console.log("🧠 Arya GUI JavaScript loaded");

// Show messages in chat
function showMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    console.log("💬", message);
}

// Test connection to Python
async function testConnection() {
    try {
        console.log("🔗 Testing connection...");
        const memory = await eel.get_enhanced_memory()();
        document.getElementById('systemStatus').innerHTML = `
            <strong>✅ System Online</strong><br>
            Conversations: ${memory.stats.conversation_count}<br>
            Plugins: ${memory.stats.total_plugins}<br>
            Uptime: ${Math.round(memory.stats.uptime / 60)} minutes
        `;
        showMessage('✅ Connection test successful!');
    } catch (error) {
        console.error("Connection error:", error);
        showMessage('❌ Connection failed: ' + error);
    }
}

// Get memory data
async function getMemory() {
    try {
        const memory = await eel.get_enhanced_memory()();
        showMessage('🧠 Memory loaded: ' + memory.stats.conversation_count + ' conversations');
    } catch (error) {
        showMessage('❌ Failed to get memory: ' + error);
    }
}

// List plugins
async function getPlugins() {
    try {
        const plugins = await eel.get_plugin_status()();
        const pluginList = Object.keys(plugins).join(', ') || 'No plugins loaded';
        showMessage('🧩 Loaded plugins: ' + pluginList);
    } catch (error) {
        showMessage('❌ Failed to get plugins: ' + error);
    }
}

// Send message to Arya
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (message) {
        showMessage('You: ' + message);
        input.value = '';
        
        try {
            console.log("📨 Sending message to Arya:", message);
            const response = await eel.send_message(message)();
            showMessage('Arya: ' + response);
        } catch (error) {
            console.error("Send message error:", error);
            showMessage('❌ Error: ' + error);
        }
    }
}

// File upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    
    uploadBtn.addEventListener('click', () => {
        console.log("📁 Upload button clicked");
        fileInput.click();
    });
    
    fileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        console.log("📤 Selected file:", file.name, file.size, file.type);
        showMessage(`📤 Uploading: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        
        try {
            const reader = new FileReader();
            
            reader.onload = async function(e) {
                try {
                    const fileData = e.target.result;
                    console.log("📊 File read successfully, size:", fileData.length);
                    
                    const result = await eel.upload_file(fileData, file.name, file.type)();
                    showMessage(result);
                } catch (error) {
                    console.error("Upload error:", error);
                    showMessage('❌ Upload error: ' + error);
                }
            };
            
            reader.onerror = function(error) {
                console.error("File read error:", error);
                showMessage('❌ File read error: ' + error);
            };
            
            reader.readAsDataURL(file);
            
        } catch (error) {
            console.error("File processing error:", error);
            showMessage('❌ File processing error: ' + error);
        }
        
        // Reset input
        fileInput.value = '';
    });
}

// Get supported file types
async function showSupportedFiles() {
    try {
        const supported = await eel.get_supported_file_types()();
        let message = '📁 Supported file types:\n';
        
        for (const [category, extensions] of Object.entries(supported)) {
            message += `• ${category}: ${extensions.join(', ')}\n`;
        }
        
        showMessage(message);
    } catch (error) {
        showMessage('❌ Failed to get supported files: ' + error);
    }
}

// Create a new plugin
async function createPlugin() {
    const pluginName = prompt("Enter plugin name:");
    if (pluginName) {
        try {
            const result = await eel.create_plugin(pluginName)();
            showMessage(result);
        } catch (error) {
            showMessage('❌ Failed to create plugin: ' + error);
        }
    }
}

// Handle Enter key in chat
document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Arya GUI initialized");
    showMessage('🧠 Arya Rachel Friday v3.0.0 initialized');
    setupFileUpload();
    
    // Test connection after a short delay
    setTimeout(() => {
        testConnection();
    }, 1000);
});