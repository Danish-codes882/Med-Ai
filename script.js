const API_BASE = '';
let selectedSymptoms = [];
let allSymptoms = [];
let allSymptomsRaw = [];
let isAuthenticated = false;

document.addEventListener('DOMContentLoaded', function() {
    checkSession();
    loadSymptoms();
    checkModelStatus();
});

async function checkSession() {
    try {
        const res = await fetch(API_BASE + '/api/session');
        const data = await res.json();
        if (data.authenticated) {
            isAuthenticated = true;
            document.getElementById('usernameDisplay').textContent = data.username;
            showMainApp();
        }
    } catch(e) {}
}

function showMainApp() {
    document.getElementById('authView').classList.add('hidden');
    document.getElementById('mainApp').classList.remove('hidden');
    loadHistory();
}

function showAuth() {
    document.getElementById('authView').classList.remove('hidden');
    document.getElementById('mainApp').classList.add('hidden');
}

function showLogin() {
    document.getElementById('loginForm').classList.remove('hidden');
    document.getElementById('registerForm').classList.add('hidden');
    document.getElementById('loginError').classList.add('hidden');
}

function showRegister() {
    document.getElementById('loginForm').classList.add('hidden');
    document.getElementById('registerForm').classList.remove('hidden');
    document.getElementById('registerError').classList.add('hidden');
}

async function handleLogin() {
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    const errEl = document.getElementById('loginError');
    errEl.classList.add('hidden');
    if (!username || !password) {
        errEl.textContent = 'Please fill in all fields';
        errEl.classList.remove('hidden');
        return;
    }
    try {
        const res = await fetch(API_BASE + '/api/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, password})
        });
        const data = await res.json();
        if (res.ok) {
            isAuthenticated = true;
            document.getElementById('usernameDisplay').textContent = data.username;
            showMainApp();
        } else {
            errEl.textContent = data.error;
            errEl.classList.remove('hidden');
        }
    } catch(e) {
        errEl.textContent = 'Connection error. Please try again.';
        errEl.classList.remove('hidden');
    }
}

async function handleRegister() {
    const username = document.getElementById('regUsername').value.trim();
    const email = document.getElementById('regEmail').value.trim();
    const password = document.getElementById('regPassword').value;
    const errEl = document.getElementById('registerError');
    errEl.classList.add('hidden');
    if (!username || !email || !password) {
        errEl.textContent = 'Please fill in all fields';
        errEl.classList.remove('hidden');
        return;
    }
    try {
        const res = await fetch(API_BASE + '/api/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, email, password})
        });
        const data = await res.json();
        if (res.ok) {
            isAuthenticated = true;
            document.getElementById('usernameDisplay').textContent = data.username;
            showMainApp();
        } else {
            errEl.textContent = data.error;
            errEl.classList.remove('hidden');
        }
    } catch(e) {
        errEl.textContent = 'Connection error. Please try again.';
        errEl.classList.remove('hidden');
    }
}

async function handleLogout() {
    try {
        await fetch(API_BASE + '/api/logout', {method: 'POST'});
    } catch(e) {}
    isAuthenticated = false;
    selectedSymptoms = [];
    showAuth();
    document.getElementById('loginUsername').value = '';
    document.getElementById('loginPassword').value = '';
    showLogin();
}

async function loadSymptoms() {
    try {
        const res = await fetch(API_BASE + '/api/symptoms');
        const data = await res.json();
        allSymptoms = data.symptoms;
        allSymptomsRaw = data.raw;
        renderSymptomTags(allSymptoms);
    } catch(e) {}
}

function renderSymptomTags(symptoms) {
    const container = document.getElementById('symptomTags');
    container.innerHTML = '';
    symptoms.forEach(function(symptom, i) {
        const rawIndex = allSymptoms.indexOf(symptom);
        const raw = rawIndex >= 0 ? allSymptomsRaw[rawIndex] : symptom.toLowerCase().replace(/ /g, '_');
        const tag = document.createElement('div');
        tag.className = 'symptom-tag' + (selectedSymptoms.includes(raw) ? ' selected' : '');
        tag.textContent = symptom;
        tag.onclick = function() { toggleSymptom(raw, symptom); };
        container.appendChild(tag);
    });
}

function filterSymptoms() {
    const query = document.getElementById('symptomSearch').value.toLowerCase();
    const filtered = allSymptoms.filter(function(s) { return s.toLowerCase().includes(query); });
    renderSymptomTags(filtered);
}

function toggleSymptom(raw, display) {
    const idx = selectedSymptoms.indexOf(raw);
    if (idx >= 0) {
        selectedSymptoms.splice(idx, 1);
    } else {
        selectedSymptoms.push(raw);
    }
    renderSymptomTags(allSymptoms.filter(function(s) {
        const q = document.getElementById('symptomSearch').value.toLowerCase();
        return !q || s.toLowerCase().includes(q);
    }));
    renderSelectedSymptoms();
}

function renderSelectedSymptoms() {
    const container = document.getElementById('selectedSymptoms');
    container.innerHTML = '';
    selectedSymptoms.forEach(function(raw) {
        const tag = document.createElement('span');
        tag.className = 'selected-tag';
        tag.innerHTML = raw.replace(/_/g, ' ') + ' <span class="remove-tag" onclick="toggleSymptom(\'' + raw + '\',\'\')">&times;</span>';
        container.appendChild(tag);
    });
    document.getElementById('symptomsCount').textContent = selectedSymptoms.length;
}

function quickSymptom(text) {
    const parts = text.split(',').map(function(s) { return s.trim().toLowerCase().replace(/ /g, '_'); });
    parts.forEach(function(p) {
        if (!selectedSymptoms.includes(p) && allSymptomsRaw.includes(p)) {
            selectedSymptoms.push(p);
        }
    });
    renderSymptomTags(allSymptoms);
    renderSelectedSymptoms();
    document.getElementById('chatInput').value = text;
    document.getElementById('welcomeMsg').style.display = 'none';
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    const age = parseInt(document.getElementById('ageInput').value) || 30;
    const gender = document.getElementById('genderInput').value;
    const consultType = document.getElementById('consultationType').value;
    const vitals = document.getElementById('vitalsInput').value.trim();
    const conditionsRaw = document.getElementById('conditionsInput').value.trim();
    const conditions = conditionsRaw ? conditionsRaw.split(',').map(function(c) { return c.trim(); }).filter(Boolean) : [];

    if (text) {
        const words = text.toLowerCase().split(/[\s,]+/);
        words.forEach(function(w) {
            const normalized = w.replace(/ /g, '_');
            if (allSymptomsRaw.includes(normalized) && !selectedSymptoms.includes(normalized)) {
                selectedSymptoms.push(normalized);
            }
        });
        renderSelectedSymptoms();
        renderSymptomTags(allSymptoms);
    }

    if (selectedSymptoms.length === 0 && !text) {
        return;
    }

    var symptoms = selectedSymptoms.slice();
    if (symptoms.length === 0 && text) {
        symptoms = text.split(/[\s,]+/).map(function(w) { return w.toLowerCase().replace(/ /g, '_'); });
    }

    var displayText = text || symptoms.map(function(s) { return s.replace(/_/g, ' '); }).join(', ');

    document.getElementById('welcomeMsg').style.display = 'none';
    addMessage(displayText, 'user');
    input.value = '';

    var sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    showTyping(true);

    try {
        var res = await fetch(API_BASE + '/api/consult', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                symptoms: symptoms,
                age: age,
                gender: gender,
                conditions: conditions,
                vitals: vitals,
                consultation_type: consultType
            })
        });
        var data = await res.json();
        showTyping(false);
        if (res.ok) {
            displayAnalysis(data);
            updateRightPanel(data.analysis);
        } else {
            addMessage('Analysis could not be completed: ' + (data.error || 'Unknown error'), 'system');
        }
    } catch(e) {
        showTyping(false);
        addMessage('Connection error. Please check your network and try again.', 'system');
    }
    sendBtn.disabled = false;
}

function addMessage(text, type) {
    var container = document.getElementById('chatMessages');
    var msg = document.createElement('div');
    msg.className = 'message message-' + type;

    var bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (type === 'system' && typeof text === 'object') {
        bubble.innerHTML = text.html;
    } else {
        bubble.textContent = text;
    }

    var time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});

    msg.appendChild(bubble);
    msg.appendChild(time);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

function displayAnalysis(data) {
    var a = data.analysis;
    var diseases = a.top_diseases || [];
    var html = '<div class="analysis-result">';
    html += '<div class="result-section"><div class="result-section-title">Predicted Conditions</div>';
    if (diseases.length > 0) {
        diseases.forEach(function(d) {
            html += '<div style="display:flex;justify-content:space-between;padding:4px 0;">';
            html += '<span>' + escapeHtml(d.disease) + '</span>';
            html += '<strong>' + d.probability.toFixed(1) + '%</strong></div>';
        });
    } else {
        html += '<div>No matching conditions found in our database.</div>';
    }
    html += '</div>';

    html += '<div class="result-section"><div class="result-section-title">Risk Assessment</div>';
    html += '<div>Risk Level: <strong>' + escapeHtml(a.risk_level) + '</strong></div>';
    html += '<div>Severity Index: <strong>' + (a.severity_index * 100).toFixed(1) + '%</strong></div>';
    html += '<div>Confidence: <strong>' + a.confidence_score.toFixed(1) + '%</strong></div>';
    html += '<div>Cluster Group: <strong>#' + a.cluster_id + '</strong> (distance: ' + a.cluster_distance.toFixed(2) + ')</div>';
    html += '<div>Emergency Symptoms Found: <strong>' + a.emergency_symptoms_detected + '</strong></div>';
    html += '</div>';

    html += '<div class="result-section"><div class="result-section-title">Advisory: ' + escapeHtml(a.advisory_category) + '</div>';
    html += '<div>' + escapeHtml(a.advisory_text) + '</div></div>';

    if (a.emergency_flag) {
        html += '<div style="color:var(--critical);font-weight:700;margin-top:8px;">WARNING: Emergency indicators detected. Please seek immediate medical care.</div>';
    }

    html += '<div class="disclaimer">' + escapeHtml(data.disclaimer) + '</div>';
    html += '</div>';

    addMessage({html: html}, 'system');
}

function updateRightPanel(analysis) {
    var riskBar = document.getElementById('riskBar');
    var riskLabel = document.getElementById('riskLabel');
    var level = analysis.risk_level.toLowerCase();

    riskBar.className = 'risk-bar ' + level;
    riskLabel.className = 'risk-label ' + level;
    riskLabel.textContent = analysis.risk_level;

    document.getElementById('severityValue').textContent = (analysis.severity_index * 100).toFixed(0) + '%';
    document.getElementById('confidenceValue').textContent = analysis.confidence_score.toFixed(0) + '%';
    document.getElementById('clusterValue').textContent = '#' + analysis.cluster_id;
    document.getElementById('symptomsCount').textContent = analysis.total_symptoms_analyzed;

    var diseaseList = document.getElementById('diseaseList');
    diseaseList.innerHTML = '';
    if (analysis.top_diseases && analysis.top_diseases.length > 0) {
        analysis.top_diseases.forEach(function(d) {
            var li = document.createElement('li');
            li.className = 'disease-item';
            var probClass = d.probability > 30 ? 'prob-high' : (d.probability > 15 ? 'prob-med' : 'prob-low');
            li.innerHTML = '<span class="disease-name">' + escapeHtml(d.disease) + '</span>' +
                           '<span class="disease-prob ' + probClass + '">' + d.probability.toFixed(1) + '%</span>';
            diseaseList.appendChild(li);
        });
    }

    var advisoryBox = document.getElementById('advisoryBox');
    var cat = analysis.advisory_category.toLowerCase().replace(' ', '');
    var advisoryClass = 'advisory-selfcare';
    if (cat === 'emergency') advisoryClass = 'advisory-emergency';
    else if (cat === 'urgentcare') advisoryClass = 'advisory-urgent';
    else if (cat === 'scheduledoctor') advisoryClass = 'advisory-schedule';
    advisoryBox.className = 'advisory-box ' + advisoryClass;
    advisoryBox.textContent = analysis.advisory_text;

    var banner = document.getElementById('emergencyBanner');
    if (analysis.emergency_flag) {
        banner.classList.add('active');
    } else {
        banner.classList.remove('active');
    }
}

function showTyping(show) {
    var el = document.getElementById('typingIndicator');
    if (show) {
        el.classList.add('active');
        var container = document.getElementById('chatMessages');
        container.scrollTop = container.scrollHeight;
    } else {
        el.classList.remove('active');
    }
}

function clearChat() {
    var container = document.getElementById('chatMessages');
    container.innerHTML = '<div class="welcome-message" id="welcomeMsg"><h2>Start Your Medical Consultation</h2><p>Select symptoms from the sidebar, fill in your details below, and describe how you are feeling.</p><div class="quick-actions"><div class="quick-action" onclick="quickSymptom(\'fever, headache, fatigue\')">Fever + Headache + Fatigue</div><div class="quick-action" onclick="quickSymptom(\'chest pain, shortness of breath\')">Chest Pain + Breathing Issues</div><div class="quick-action" onclick="quickSymptom(\'cough, sore throat, congestion\')">Cold / Flu Symptoms</div></div></div>';
    selectedSymptoms = [];
    renderSelectedSymptoms();
    renderSymptomTags(allSymptoms);
    resetRightPanel();
}

function resetRightPanel() {
    document.getElementById('riskBar').className = 'risk-bar';
    document.getElementById('riskLabel').textContent = '--';
    document.getElementById('riskLabel').className = 'risk-label';
    document.getElementById('severityValue').textContent = '--';
    document.getElementById('confidenceValue').textContent = '--';
    document.getElementById('clusterValue').textContent = '--';
    document.getElementById('symptomsCount').textContent = '0';
    document.getElementById('diseaseList').innerHTML = '<li class="disease-item" style="color:var(--text-muted);font-size:0.82rem;">No analysis yet</li>';
    document.getElementById('advisoryBox').className = 'advisory-box advisory-selfcare';
    document.getElementById('advisoryBox').textContent = 'Submit your symptoms to receive an AI-generated medical advisory.';
    document.getElementById('emergencyBanner').classList.remove('active');
}

function switchView(view) {
    document.querySelectorAll('.nav-item').forEach(function(el) {
        el.classList.remove('active');
        if (el.dataset.view === view) el.classList.add('active');
    });
    if (view === 'chat') {
        document.getElementById('chatView').classList.add('active');
        document.getElementById('historyView').classList.remove('active');
    } else {
        document.getElementById('chatView').classList.remove('active');
        document.getElementById('historyView').classList.add('active');
        loadHistory();
    }
}

async function loadHistory() {
    try {
        var res = await fetch(API_BASE + '/api/history');
        var data = await res.json();
        var container = document.getElementById('historyList');
        container.innerHTML = '';
        if (data.history && data.history.length > 0) {
            data.history.forEach(function(item) {
                var div = document.createElement('div');
                div.className = 'history-item';
                var riskColor = 'var(--success)';
                var riskBg = 'rgba(0,230,118,0.12)';
                if (item.risk_level === 'Critical') { riskColor = 'var(--critical)'; riskBg = 'rgba(255,23,68,0.12)'; }
                else if (item.risk_level === 'High') { riskColor = 'var(--danger)'; riskBg = 'rgba(255,82,82,0.12)'; }
                else if (item.risk_level === 'Medium') { riskColor = 'var(--warning)'; riskBg = 'rgba(255,171,64,0.12)'; }

                var date = new Date(item.created_at);
                var dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
                div.innerHTML = '<div class="history-item-header">' +
                    '<span class="history-disease">' + escapeHtml(item.predicted_disease || 'Analysis') + '</span>' +
                    '<span class="history-risk" style="background:' + riskBg + ';color:' + riskColor + ';">' + escapeHtml(item.risk_level) + '</span></div>' +
                    '<div style="font-size:0.78rem;color:var(--text-secondary);">Symptoms: ' + escapeHtml(item.symptoms.join(', ')) + '</div>' +
                    '<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;">' +
                    'Severity: ' + (item.severity_index * 100).toFixed(0) + '% | Confidence: ' + item.confidence_score.toFixed(0) + '% | ' + dateStr + '</div>';
                container.appendChild(div);
            });
        } else {
            container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px 0;">No consultation history yet. Start a new consultation to see results here.</div>';
        }
    } catch(e) {}
}

async function checkModelStatus() {
    try {
        var res = await fetch(API_BASE + '/api/scrape-status');
        if (res.ok) {
            var data = await res.json();
            var statusEl = document.getElementById('modelStatus');
            if (data.models_trained) {
                statusEl.className = 'status-badge status-online';
                statusEl.innerHTML = '<div class="status-dot"></div><span>AI Ready (' + data.disease_count + ' conditions)</span>';
            } else {
                statusEl.className = 'status-badge status-training';
                statusEl.innerHTML = '<div class="status-dot"></div><span>Training Models...</span>';
                setTimeout(checkModelStatus, 5000);
            }
        }
    } catch(e) {
        setTimeout(checkModelStatus, 5000);
    }
}

function setTheme(theme) {
    document.body.classList.remove('theme-blue', 'theme-green', 'theme-purple', 'light-mode');
    document.querySelectorAll('.theme-opt').forEach(opt => opt.classList.remove('active'));
    
    if (theme !== 'dark') {
        document.body.classList.add('theme-' + theme);
    }
    
    document.querySelector('.theme-opt.' + theme).classList.add('active');
    localStorage.setItem('medcore-theme', theme);
}

// Initial theme load
const savedTheme = localStorage.getItem('medcore-theme') || 'dark';
document.addEventListener('DOMContentLoaded', () => setTheme(savedTheme));

function toggleTheme() {
    if (document.body.classList.contains('light-mode')) {
        const current = localStorage.getItem('medcore-theme') || 'dark';
        setTheme(current);
    } else {
        document.body.classList.add('light-mode');
    }
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('mobile-open');
}

function togglePanel() {
    document.getElementById('rightPanel').classList.toggle('mobile-open');
}

function escapeHtml(text) {
    if (!text) return '';
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        document.getElementById('sidebar').classList.remove('mobile-open');
        document.getElementById('rightPanel').classList.remove('mobile-open');
    }
});
