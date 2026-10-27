/**
 * PhytoVision AI - Full-Stack Diagnostic Engine & Explainable AI Client
 */

// State Management
const state = {
  currentFile: null,
  currentSampleClass: null,
  currentAlpha: 0.45,
  lastDiagnosisResult: null,
  allClasses: [],
  samples: [],
  metrics: null,
  activeTab: 'tab-diagnostics',
  activeHeatmapView: 'split',
  webcamStream: null,
  currentUser: null,
  authToken: localStorage.getItem('plantcare_auth_token') || null,
  authMode: 'login'
};

// DOM Element Registry
const DOM = {
  tabs: document.querySelectorAll('.nav-tab'),
  panes: document.querySelectorAll('.tab-pane'),
  backendDot: document.getElementById('backend-status-dot'),
  backendText: document.getElementById('backend-status-text'),

  // Authentication
  btnOpenAuth: document.getElementById('btn-open-auth'),
  authGateBanner: document.getElementById('auth-gate-banner'),
  btnGateSignIn: document.getElementById('btn-gate-signin'),
  userProfileMenu: document.getElementById('user-profile-menu'),
  btnToggleUserMenu: document.getElementById('btn-toggle-user-menu'),
  userDropdownCard: document.getElementById('user-dropdown-card'),
  headerUserAvatar: document.getElementById('header-user-avatar'),
  headerUserName: document.getElementById('header-user-name'),
  headerUserRole: document.getElementById('header-user-role'),
  dropdownUserName: document.getElementById('dropdown-user-name'),
  dropdownUserEmail: document.getElementById('dropdown-user-email'),
  btnLogout: document.getElementById('btn-logout'),

  authModal: document.getElementById('auth-modal'),
  btnCloseAuthModal: document.getElementById('btn-close-auth-modal'),
  tabAuthLogin: document.getElementById('tab-auth-login'),
  tabAuthRegister: document.getElementById('tab-auth-register'),
  btnFillDemo: document.getElementById('btn-fill-demo'),
  authAlert: document.getElementById('auth-alert'),
  authForm: document.getElementById('auth-form'),
  groupAuthName: document.getElementById('group-auth-name'),
  authName: document.getElementById('auth-name'),
  authEmail: document.getElementById('auth-email'),
  authPassword: document.getElementById('auth-password'),
  btnTogglePwd: document.getElementById('btn-toggle-pwd'),
  btnAuthSubmit: document.getElementById('btn-auth-submit'),
  authSpinner: document.getElementById('auth-spinner'),
  authSubmitLabel: document.getElementById('auth-submit-label'),
  authModalTitle: document.getElementById('auth-modal-title'),
  authModalSubtitle: document.getElementById('auth-modal-subtitle'),

  // Diagnostic Studio
  dropzone: document.getElementById('leaf-dropzone'),
  fileInput: document.getElementById('file-input'),
  dropzonePrompt: document.getElementById('dropzone-prompt'),
  dropzonePreview: document.getElementById('dropzone-preview'),
  previewImage: document.getElementById('preview-image'),
  scanLaser: document.getElementById('scan-laser'),
  btnRemoveFile: document.getElementById('btn-remove-file'),
  btnRunDiagnosis: document.getElementById('btn-run-diagnosis'),
  curatedSamplesGrid: document.getElementById('curated-samples-grid'),
  recentScansList: document.getElementById('recent-scans-list'),
  recentScansCount: document.getElementById('recent-scans-count'),

  // Results Panes
  emptyState: document.getElementById('results-empty-state'),
  loadingState: document.getElementById('results-loading-state'),
  activeState: document.getElementById('results-active-state'),

  // Result Elements
  bannerCard: document.getElementById('diagnosis-banner-card'),
  cropBadge: document.getElementById('res-crop-badge'),
  severityBadge: document.getElementById('res-severity-badge'),
  confidenceBadge: document.getElementById('res-confidence-badge'),
  inferenceTime: document.getElementById('res-inference-time'),
  conditionTitle: document.getElementById('res-condition-title'),
  pathogenInfo: document.getElementById('res-pathogen-info'),
  confidencePercent: document.getElementById('res-confidence-percent'),
  confidenceAlert: document.getElementById('res-confidence-alert'),

  // Heatmap View Elements
  viewTabs: document.querySelectorAll('.view-tab'),
  imgSuperimposed: document.getElementById('img-superimposed'),
  imgSplitOriginal: document.getElementById('img-split-original'),
  imgSplitRaw: document.getElementById('img-split-raw'),
  imgSplitOverlay: document.getElementById('img-split-overlay'),
  imgRawHeat: document.getElementById('img-raw-heat'),
  viewPaneSuperimposed: document.getElementById('view-pane-superimposed'),
  viewPaneSplit: document.getElementById('view-pane-split'),
  viewPaneRaw: document.getElementById('view-pane-raw'),

  // Top 5 & Prescription
  top5List: document.getElementById('res-top5-list'),
  rxSymptoms: document.getElementById('res-rx-symptoms'),
  rxOrganic: document.getElementById('res-rx-organic'),
  rxChemical: document.getElementById('res-rx-chemical'),
  rxPrevention: document.getElementById('res-rx-prevention'),
  btnPrintReport: document.getElementById('btn-print-report'),

  // Camera Modal
  btnOpenCamera: document.getElementById('btn-open-camera'),
  cameraModal: document.getElementById('camera-modal'),
  cameraVideo: document.getElementById('camera-video'),
  cameraCanvas: document.getElementById('camera-canvas'),
  btnSnapshotCamera: document.getElementById('btn-snapshot-camera'),
  btnCancelCamera: document.getElementById('btn-cancel-camera'),
  btnCloseCamera: document.getElementById('btn-close-camera'),

  // Disease Modal
  diseaseModal: document.getElementById('disease-modal'),
  modalDiseaseTitle: document.getElementById('modal-disease-title'),
  modalDiseaseBody: document.getElementById('modal-disease-body'),
  btnCloseDiseaseModal: document.getElementById('btn-close-disease-modal'),

  // Architecture & Metrics
  cnnFlowDiagram: document.getElementById('cnn-flow-diagram'),
  layersTbody: document.getElementById('layers-tbody'),
  metricsSummaryCards: document.getElementById('metrics-summary-cards'),
  epochsTbody: document.getElementById('epochs-tbody'),

  // Encyclopedia
  encyclopediaGrid: document.getElementById('encyclopedia-cards-grid'),
  encyclopediaSearch: document.getElementById('encyclopedia-search'),
  cropFilterBtns: document.querySelectorAll('.filter-pill')
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupDropzone();
  setupCamera();
  setupHeatmapViews();
  setupEncyclopediaFilters();
  setupAuth();

  if (DOM.btnPrintReport) {
    DOM.btnPrintReport.addEventListener('click', () => window.print());
  }

  // Check auth session
  checkCurrentUser();

  // Check backend & fetch initial catalogs
  checkBackendHealth();
  fetchCuratedSamples();
  fetchClasses();
  fetchMetrics();
});

// ============================================================================
// NAVIGATION TABS
// ============================================================================
function setupNavigation() {
  DOM.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      DOM.tabs.forEach(t => t.classList.remove('active'));
      DOM.panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const activePane = document.getElementById(targetId);
      if (activePane) activePane.classList.add('active');
      state.activeTab = targetId;

      // Render charts if tab-metrics activated
      if (targetId === 'tab-metrics' && state.metrics) {
        renderCharts(state.metrics);
      }
    });
  });
}

// ============================================================================
// BACKEND HEALTH & HEARTBEAT
// ============================================================================
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      DOM.backendDot.className = 'status-indicator online';
      DOM.backendText.textContent = `AI Engine Online • ${data.total_classes} Classes`;
      return true;
    }
  } catch (err) {
    DOM.backendDot.className = 'status-indicator';
    DOM.backendText.textContent = 'AI Engine Offline';
  }
  return false;
}

// ============================================================================
// DROPZONE & FILE UPLOAD
// ============================================================================
function setupDropzone() {
  const { dropzone, fileInput, dropzonePrompt, dropzonePreview, previewImage, btnRemoveFile, btnRunDiagnosis } = DOM;

  dropzone.addEventListener('click', (e) => {
    if (e.target !== btnRemoveFile && !btnRemoveFile.contains(e.target)) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      selectFile(fileInput.files[0]);
      if (!state.authToken || !state.currentUser) {
        openAuthModal('login');
        showAuthAlert('🔒 Sign in required: Please sign in or use 1-Click Demo to diagnose this uploaded leaf image.', 'error');
      } else {
        runDiagnosis();
      }
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
      if (!state.authToken || !state.currentUser) {
        openAuthModal('login');
        showAuthAlert('🔒 Sign in required: Please sign in or use 1-Click Demo to diagnose this dropped leaf image.', 'error');
      } else {
        runDiagnosis();
      }
    }
  });

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFileSelection();
  });

  btnRunDiagnosis.addEventListener('click', () => {
    if (!state.authToken || !state.currentUser) {
      openAuthModal('login');
      showAuthAlert('🔒 Sign in required: Please sign in or use 1-Click Demo to run plant disease diagnostics.', 'error');
      return;
    }
    runDiagnosis();
  });
}

function selectFile(file) {
  state.currentFile = file;
  state.currentSampleClass = null; // clear sample mode

  // Deselect sample cards
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));

  const reader = new FileReader();
  reader.onload = (e) => {
    DOM.previewImage.src = e.target.result;
    DOM.dropzonePrompt.style.display = 'none';
    DOM.dropzonePreview.style.display = 'flex';
    DOM.btnRunDiagnosis.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearFileSelection() {
  state.currentFile = null;
  state.currentSampleClass = null;
  DOM.fileInput.value = '';
  DOM.dropzonePrompt.style.display = 'block';
  DOM.dropzonePreview.style.display = 'none';
  DOM.btnRunDiagnosis.disabled = true;
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
}

// ============================================================================
// CURATED SAMPLES GALLERY
// ============================================================================
async function fetchCuratedSamples() {
  try {
    const res = await fetch('/api/samples');
    if (res.ok) {
      const data = await res.json();
      state.samples = data.samples;
      renderCuratedSamples(data.samples);
    }
  } catch (err) {
    DOM.curatedSamplesGrid.innerHTML = '<div class="sample-skeleton">Could not load sample gallery.</div>';
  }
}

function renderCuratedSamples(samples) {
  if (!samples || samples.length === 0) {
    DOM.curatedSamplesGrid.innerHTML = '<div class="sample-skeleton">No samples found in dataset/valid.</div>';
    return;
  }

  DOM.curatedSamplesGrid.innerHTML = samples.map(sample => `
    <div class="sample-card" data-class="${sample.class_name}">
      <img src="${sample.api_url}" class="sample-thumb" alt="${sample.name}" loading="lazy" />
      <div class="sample-text">
        <span class="sample-name">${sample.name}</span>
        <span class="sample-tag">${sample.category}</span>
      </div>
    </div>
  `).join('');

  DOM.curatedSamplesGrid.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', () => {
      const className = card.getAttribute('data-class');
      selectSample(className, card);
    });
  });
}

function selectSample(className, cardElement) {
  state.currentSampleClass = className;
  state.currentFile = null; // reset uploaded file

  // Highlight active card
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  if (cardElement) cardElement.classList.add('active');

  // Set dropzone preview image
  DOM.previewImage.src = `/api/sample-image/${className}`;
  DOM.dropzonePrompt.style.display = 'none';
  DOM.dropzonePreview.style.display = 'flex';
  DOM.btnRunDiagnosis.disabled = false;

  // Gate diagnosis behind authentication
  if (!state.authToken || !state.currentUser) {
    openAuthModal('login');
    showAuthAlert('🔒 Sign in required: Please sign in or use 1-Click Demo to analyze this sample leaf.', 'error');
    return;
  }

  // Automatically execute diagnosis for authenticated user
  runDiagnosis();
}

// ============================================================================
// HEATMAP VIEW TABS
// ============================================================================

function setupHeatmapViews() {
  DOM.viewTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const view = tab.getAttribute('data-view');
      DOM.viewTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.activeHeatmapView = view;

      DOM.viewPaneSuperimposed.style.display = (view === 'superimposed') ? 'flex' : 'none';
      DOM.viewPaneSplit.style.display = (view === 'split') ? 'grid' : 'none';
      DOM.viewPaneRaw.style.display = (view === 'raw') ? 'flex' : 'none';
    });
  });
}

// ============================================================================
// RUN DIAGNOSIS (API INFERENCE + GRAD-CAM)
// ============================================================================
async function runDiagnosis() {
  if (!state.currentFile && !state.currentSampleClass) return;

  if (!state.authToken || !state.currentUser) {
    openAuthModal('login');
    showAuthAlert('🔒 Sign in required: Please sign in or use 1-Click Demo to run plant disease diagnostics.', 'error');
    return;
  }

  // Show scan animation line and transition to loading state
  DOM.scanLaser.style.display = 'block';
  DOM.btnRunDiagnosis.disabled = true;
  DOM.emptyState.style.display = 'none';
  DOM.activeState.style.display = 'none';
  DOM.loadingState.style.display = 'block';

  try {
    const formData = new FormData();
    formData.append('alpha', state.currentAlpha);

    if (state.currentFile) {
      formData.append('file', state.currentFile);
    } else if (state.currentSampleClass) {
      formData.append('sample_class', state.currentSampleClass);
    }

    const headers = {};
    if (state.authToken) {
      headers['Authorization'] = `Bearer ${state.authToken}`;
    }

    const response = await fetch('/api/predict', {
      method: 'POST',
      headers,
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Diagnosis failed on server.');
    }

    const data = await response.json();
    state.lastDiagnosisResult = data;

    renderDiagnosisResults(data);
    fetchRecentScans();

  } catch (error) {
    alert(`Diagnosis Error: ${error.message}`);
    DOM.emptyState.style.display = 'block';
  } finally {
    DOM.scanLaser.style.display = 'none';
    DOM.loadingState.style.display = 'none';
    DOM.btnRunDiagnosis.disabled = false;
  }
}

function renderDiagnosisResults(data) {
  const { prediction, top_5, images, telemetry } = data;

  // Primary banner
  DOM.conditionTitle.textContent = prediction.condition;
  DOM.cropBadge.textContent = prediction.crop;
  DOM.pathogenInfo.textContent = `Pathogen: ${prediction.pathogen || 'N/A'}`;
  DOM.confidencePercent.textContent = `${prediction.confidence}%`;
  DOM.inferenceTime.textContent = `${telemetry.inference_time_ms} ms`;

  // Severity styling
  DOM.severityBadge.textContent = prediction.severity || (prediction.is_healthy ? 'None' : 'Moderate');
  if (prediction.is_healthy) {
    DOM.severityBadge.className = 'severity-badge healthy';
    DOM.bannerCard.className = 'glass-card diagnosis-banner';
  } else if ((prediction.severity || '').toLowerCase().includes('severe') || (prediction.severity || '').toLowerCase().includes('emergency')) {
    DOM.severityBadge.className = 'severity-badge severe';
    DOM.bannerCard.className = 'glass-card diagnosis-banner severe';
  } else {
    DOM.severityBadge.className = 'severity-badge';
    DOM.bannerCard.className = 'glass-card diagnosis-banner warning';
  }

  // Confidence alert status
  if (prediction.confidence_status === 'high') {
    DOM.confidenceBadge.textContent = 'High Certainty';
    DOM.confidenceBadge.style.color = '#34d399';
    DOM.confidenceAlert.innerHTML = `✅ <strong>Verified Diagnostic:</strong> ${prediction.confidence_message}`;
  } else if (prediction.confidence_status === 'moderate') {
    DOM.confidenceBadge.textContent = 'Moderate Certainty';
    DOM.confidenceBadge.style.color = '#fbbf24';
    DOM.confidenceAlert.innerHTML = `⚠️ <strong>Caution:</strong> ${prediction.confidence_message}`;
  } else {
    DOM.confidenceBadge.textContent = 'Low / Inconclusive';
    DOM.confidenceBadge.style.color = '#f87171';
    DOM.confidenceAlert.innerHTML = `🛑 <strong>Low Certainty:</strong> ${prediction.confidence_message}`;
  }

  // Images
  DOM.imgSuperimposed.src = images.superimposed_base64;
  DOM.imgSplitOriginal.src = images.original_base64;
  if (DOM.imgSplitRaw) DOM.imgSplitRaw.src = images.raw_heatmap_base64;
  DOM.imgSplitOverlay.src = images.superimposed_base64;
  DOM.imgRawHeat.src = images.raw_heatmap_base64;

  // Top 5 Probabilities
  DOM.top5List.innerHTML = top_5.map((item, idx) => `
    <div class="top5-item">
      <div class="top5-row">
        <span><strong class="top5-rank">#${item.rank}</strong> ${item.crop} - ${item.condition}</span>
        <span class="top5-pct">${item.probability}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill ${idx > 0 ? 'secondary' : ''}" style="width: ${Math.min(item.probability, 100)}%;"></div>
      </div>
    </div>
  `).join('');

  // Prescription plan
  DOM.rxSymptoms.textContent = prediction.symptoms || 'No specific leaf symptom note available.';
  DOM.rxOrganic.textContent = prediction.organic_treatment || 'General cultural sanitation recommended.';
  DOM.rxChemical.textContent = prediction.chemical_treatment || 'Fungicide or bactericide as needed by certified advisor.';
  DOM.rxPrevention.textContent = prediction.prevention || 'Ensure clean seeds, crop rotation, and optimal plant ventilation.';

  DOM.activeState.style.display = 'flex';
}

// ============================================================================
// RECENT SCANS ACTIVITY
// ============================================================================
async function fetchRecentScans() {
  try {
    const res = await fetch('/api/recent-scans');
    if (res.ok) {
      const data = await res.json();
      renderRecentScans(data.scans);
    }
  } catch (err) {
    console.error('Failed to load recent scans:', err);
  }
}

function renderRecentScans(scans) {
  if (!scans || scans.length === 0) {
    DOM.recentScansList.innerHTML = '<p class="empty-text">No scans in this session yet.</p>';
    DOM.recentScansCount.textContent = '0';
    return;
  }

  DOM.recentScansCount.textContent = scans.length;
  DOM.recentScansList.innerHTML = scans.map(s => `
    <div class="scan-item">
      <div class="scan-item-info">
        <img src="${s.thumbnail_base64}" class="scan-item-thumb" alt="Scan" />
        <div>
          <span style="font-weight:700; display:block;">${s.crop} - ${s.condition}</span>
          <span style="font-size:0.72rem; color:var(--text-dim);">${s.timestamp}</span>
        </div>
      </div>
      <span style="font-family:var(--font-mono); font-weight:700; color:${s.confidence >= 70 ? 'var(--primary)' : 'var(--accent-amber)'}">${s.confidence}%</span>
    </div>
  `).join('');
}

// ============================================================================
// WEBCAM LIVE CAPTURE
// ============================================================================
function setupCamera() {
  const { btnOpenCamera, cameraModal, cameraVideo, cameraCanvas, btnSnapshotCamera, btnCancelCamera, btnCloseCamera } = DOM;

  btnOpenCamera.addEventListener('click', async () => {
    if (!state.authToken || !state.currentUser) {
      openAuthModal('login');
      showAuthAlert('🔒 Sign in required: Please sign in or use 1-Click Demo to use live camera scanner.', 'error');
      return;
    }

    try {
      cameraModal.style.display = 'flex';
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      state.webcamStream = stream;
      cameraVideo.srcObject = stream;
    } catch (err) {
      alert('Could not access camera. Please allow camera permissions or upload an image file.');
      cameraModal.style.display = 'none';
    }
  });

  const stopCamera = () => {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(track => track.stop());
      state.webcamStream = null;
    }
    cameraModal.style.display = 'none';
  };

  btnCancelCamera.addEventListener('click', stopCamera);
  btnCloseCamera.addEventListener('click', stopCamera);

  btnSnapshotCamera.addEventListener('click', () => {
    const vw = cameraVideo.videoWidth || 640;
    const vh = cameraVideo.videoHeight || 480;
    const size = Math.min(vw, vh);
    cameraCanvas.width = size;
    cameraCanvas.height = size;
    const ctx = cameraCanvas.getContext('2d');
    const startX = (vw - size) / 2;
    const startY = (vh - size) / 2;
    ctx.drawImage(cameraVideo, startX, startY, size, size, 0, 0, size, size);

    cameraCanvas.toBlob(blob => {
      const file = new File([blob], `camera_leaf_${Date.now()}.jpg`, { type: 'image/jpeg' });
      selectFile(file);
      stopCamera();
    }, 'image/jpeg', 0.95);
  });
}

// ============================================================================
// TAB 2 & 3: ARCHITECTURE & METRICS
// ============================================================================
async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (res.ok) {
      const data = await res.json();
      state.metrics = data;
      renderArchitecture(data);
      renderMetricsSummary(data);
      renderCharts(data);
      renderEpochsTable(data);
    }
  } catch (err) {
    console.error('Failed to load model metrics:', err);
  }
}

function renderArchitecture(data) {
  const { layers } = data;
  if (!layers || !DOM.cnnFlowDiagram || !DOM.layersTbody) return;

  // Flow Diagram
  DOM.cnnFlowDiagram.innerHTML = layers.map((layer, idx) => `
    <div class="flow-node ${layer.is_gradcam_target ? 'gradcam-target' : ''}">
      <span class="node-title">${layer.name}</span>
      <span class="node-subtitle">${layer.type}</span>
      ${layer.is_gradcam_target ? '<span style="font-size:0.65rem; color:#c084fc; font-weight:700;">★ Grad-CAM</span>' : ''}
    </div>
    ${idx < layers.length - 1 ? '<span class="flow-arrow">→</span>' : ''}
  `).join('');

  // Table
  DOM.layersTbody.innerHTML = layers.map(layer => `
    <tr>
      <td>${layer.index}</td>
      <td>${layer.name} ${layer.is_gradcam_target ? '<span class="target-layer-chip" style="margin-left:6px;">Target XAI</span>' : ''}</td>
      <td>${layer.type}</td>
      <td style="font-family:var(--font-mono); font-size:0.78rem;">${layer.output_shape}</td>
      <td style="font-family:var(--font-mono);">${layer.params.toLocaleString()}</td>
      <td style="font-size:0.8rem;">${getLayerRole(layer.type, layer.name)}</td>
    </tr>
  `).join('');
}

function getLayerRole(type, name) {
  if (name === 'rescaling') return 'Scales input pixels from [0, 255] to [0.0, 1.0]';
  if (name === 'conv2d') return 'Learns edge and color gradient primitives (32 filters)';
  if (name === 'conv2d_1') return 'Learns mid-level leaf textures & contours (64 filters)';
  if (name === 'conv2d_2') return 'Learns high-level pathology lesions & spots (Grad-CAM Target, 128 filters)';
  if (type === 'MaxPooling2D') return 'Spatial downsampling (2x2) for translation invariance';
  if (type === 'GlobalAveragePooling2D') return 'Averages spatial maps into compact 128-d feature vector';
  if (type === 'Dense' && name === 'dense') return 'Dense classification head (128 units, ReLU)';
  if (type === 'Dropout') return 'Regularization (20% dropout rate) to prevent overfitting';
  if (name === 'dense_1') return 'Softmax output distribution across 38 crop disease classes';
  return 'Neural Feature Transformation';
}

function renderMetricsSummary(data) {
  const summary = data.metrics_summary || {};
  DOM.metricsSummaryCards.innerHTML = `
    <div class="stat-card">
      <span class="stat-val">${summary.final_val_accuracy}%</span>
      <span class="stat-lbl">Validation Accuracy</span>
    </div>
    <div class="stat-card">
      <span class="stat-val">${summary.final_train_accuracy}%</span>
      <span class="stat-lbl">Training Accuracy</span>
    </div>
    <div class="stat-card">
      <span class="stat-val">${summary.final_val_loss}</span>
      <span class="stat-lbl">Validation Loss</span>
    </div>
    <div class="stat-card">
      <span class="stat-val">${summary.epochs_completed}</span>
      <span class="stat-lbl">Epochs Trained</span>
    </div>
  `;
}

function renderCharts(data) {
  const hist = data.training_history;
  if (!hist || !hist.accuracy) return;

  const epochs = hist.accuracy.map((_, i) => i + 1);
  const trainAcc = hist.accuracy.map(v => v * 100);
  const valAcc = hist.val_accuracy.map(v => v * 100);
  const trainLoss = hist.loss;
  const valLoss = hist.val_loss;

  drawSvgCurve('svg-accuracy-chart', epochs, trainAcc, valAcc, '%', [80, 100]);
  drawSvgCurve('svg-loss-chart', epochs, trainLoss, valLoss, '', [0, 0.6]);
}

function drawSvgCurve(svgId, xVals, y1Vals, y2Vals, unit, yDomain) {
  const svg = document.getElementById(svgId);
  if (!svg) return;

  const width = 600;
  const height = 300;
  const padL = 50;
  const padR = 25;
  const padT = 25;
  const padB = 40;

  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const yMin = yDomain ? yDomain[0] : Math.min(...y1Vals, ...y2Vals) * 0.9;
  const yMax = yDomain ? yDomain[1] : Math.max(...y1Vals, ...y2Vals) * 1.05;

  const getX = (idx) => padL + (idx / (xVals.length - 1)) * plotW;
  const getY = (val) => padT + plotH - ((val - yMin) / (yMax - yMin)) * plotH;

  const line1Points = y1Vals.map((v, i) => `${getX(i)},${getY(v)}`).join(' ');
  const line2Points = y2Vals.map((v, i) => `${getX(i)},${getY(v)}`).join(' ');

  // Grid lines
  let gridLines = '';
  for (let i = 0; i <= 4; i++) {
    const yVal = yMin + (i / 4) * (yMax - yMin);
    const yPos = getY(yVal);
    gridLines += `
      <line x1="${padL}" y1="${yPos}" x2="${width - padR}" y2="${yPos}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3" />
      <text x="${padL - 8}" y="${yPos + 4}" font-size="11" fill="#64748b" text-anchor="end" font-family="JetBrains Mono">${yVal.toFixed(unit === '%' ? 1 : 3)}${unit}</text>
    `;
  }

  // X labels
  let xLabels = '';
  xVals.forEach((epoch, i) => {
    const xPos = getX(i);
    xLabels += `
      <text x="${xPos}" y="${height - 12}" font-size="11" fill="#64748b" text-anchor="middle" font-family="JetBrains Mono">E${epoch}</text>
    `;
  });

  svg.innerHTML = `
    ${gridLines}
    ${xLabels}
    <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="${line1Points}" />
    <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="${line2Points}" />
    ${y1Vals.map((v, i) => `<circle cx="${getX(i)}" cy="${getY(v)}" r="4" fill="#38bdf8" />`).join('')}
    ${y2Vals.map((v, i) => `<circle cx="${getX(i)}" cy="${getY(v)}" r="4" fill="#10b981" />`).join('')}
  `;
}

function renderEpochsTable(data) {
  const hist = data.training_history;
  if (!hist || !hist.accuracy) return;

  DOM.epochsTbody.innerHTML = hist.accuracy.map((acc, i) => {
    const tAcc = (acc * 100).toFixed(2);
    const vAcc = (hist.val_accuracy[i] * 100).toFixed(2);
    const tLoss = hist.loss[i].toFixed(4);
    const vLoss = hist.val_loss[i].toFixed(4);
    const delta = (hist.val_accuracy[i] * 100 - acc * 100).toFixed(2);

    return `
      <tr>
        <td style="font-family:var(--font-mono); font-weight:700;">Epoch ${i + 1}</td>
        <td style="color:#38bdf8; font-family:var(--font-mono);">${tAcc}%</td>
        <td style="color:#10b981; font-family:var(--font-mono); font-weight:700;">${vAcc}%</td>
        <td style="font-family:var(--font-mono);">${tLoss}</td>
        <td style="font-family:var(--font-mono);">${vLoss}</td>
        <td style="font-family:var(--font-mono); color:${delta >= 0 ? '#34d399' : '#f87171'};">${delta >= 0 ? '+' : ''}${delta}%</td>
      </tr>
    `;
  }).join('');
}

// ============================================================================
// TAB 4: ENCYCLOPEDIA & PATHOLOGY
// ============================================================================
async function fetchClasses() {
  try {
    const res = await fetch('/api/classes');
    if (res.ok) {
      const data = await res.json();
      state.allClasses = data.classes;
      renderEncyclopedia(data.classes);
    }
  } catch (err) {
    console.error('Failed to load crop catalog:', err);
  }
}

function renderEncyclopedia(classesList) {
  DOM.encyclopediaGrid.innerHTML = classesList.map(item => `
    <div class="glass-card disease-card" data-raw="${item.raw_class}">
      <div>
        <div class="card-top">
          <span class="card-crop">${item.crop}</span>
          <span class="severity-badge ${item.is_healthy ? 'healthy' : (item.severity.toLowerCase().includes('severe') ? 'severe' : '')}">
            ${item.is_healthy ? 'Healthy' : item.severity}
          </span>
        </div>
        <h4 class="card-condition">${item.condition}</h4>
        <p class="card-symptoms">${item.symptoms || 'No specific symptoms recorded.'}</p>
      </div>
      <div class="card-footer">
        <span>Pathogen: ${item.pathogen ? item.pathogen.split('(')[0] : 'N/A'}</span>
        <span>View Details →</span>
      </div>
    </div>
  `).join('');

  DOM.encyclopediaGrid.querySelectorAll('.disease-card').forEach(card => {
    card.addEventListener('click', () => {
      const raw = card.getAttribute('data-raw');
      const item = state.allClasses.find(c => c.raw_class === raw);
      if (item) openDiseaseModal(item);
    });
  });
}

function setupEncyclopediaFilters() {
  DOM.encyclopediaSearch.addEventListener('input', (e) => {
    applyFilters(e.target.value.toLowerCase(), getActiveCropFilter());
  });

  DOM.cropFilterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.cropFilterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const crop = btn.getAttribute('data-crop');
      applyFilters(DOM.encyclopediaSearch.value.toLowerCase(), crop);
    });
  });

  DOM.btnCloseDiseaseModal.addEventListener('click', () => {
    DOM.diseaseModal.style.display = 'none';
  });
}

function getActiveCropFilter() {
  const activeBtn = document.querySelector('.filter-pill.active');
  return activeBtn ? activeBtn.getAttribute('data-crop') : 'all';
}

function applyFilters(searchTerm, cropFilter) {
  const filtered = state.allClasses.filter(c => {
    const matchesCrop = (cropFilter === 'all') || (c.crop.toLowerCase().includes(cropFilter.toLowerCase()));
    const matchesSearch = !searchTerm ||
      c.crop.toLowerCase().includes(searchTerm) ||
      c.condition.toLowerCase().includes(searchTerm) ||
      c.symptoms.toLowerCase().includes(searchTerm);
    return matchesCrop && matchesSearch;
  });

  renderEncyclopedia(filtered);
}

function openDiseaseModal(item) {
  DOM.modalDiseaseTitle.textContent = `${item.crop} - ${item.condition}`;
  DOM.modalDiseaseBody.innerHTML = `
    <div style="margin-bottom:16px;">
      <span class="severity-badge ${item.is_healthy ? 'healthy' : (item.severity.toLowerCase().includes('severe') ? 'severe' : '')}">
        Severity: ${item.severity}
      </span>
      <p style="margin-top:8px; font-size:0.88rem; color:var(--text-muted);">
        <strong>Causal Agent:</strong> ${item.pathogen}
      </p>
    </div>
    <div class="prescription-grid">
      <div class="rx-box">
        <div class="rx-title"><span class="rx-icon orange">🔍</span><h4>Symptoms</h4></div>
        <p>${item.symptoms}</p>
      </div>
      <div class="rx-box">
        <div class="rx-title"><span class="rx-icon green">🌱</span><h4>Organic / Biological Controls</h4></div>
        <p>${item.organic_treatment}</p>
      </div>
      <div class="rx-box">
        <div class="rx-title"><span class="rx-icon blue">🧪</span><h4>Chemical Controls</h4></div>
        <p>${item.chemical_treatment}</p>
      </div>
      <div class="rx-box">
        <div class="rx-title"><span class="rx-icon purple">🛡️</span><h4>Preventive Strategy</h4></div>
        <p>${item.prevention}</p>
      </div>
    </div>
    <div style="margin-top:20px; display:flex; justify-content:flex-end;">
      <button class="btn-primary btn-sm" id="btn-modal-test-sample" style="width:auto; padding:8px 18px;">
        Test This Disease in Studio
      </button>
    </div>
  `;

  DOM.diseaseModal.style.display = 'flex';

  const testBtn = document.getElementById('btn-modal-test-sample');
  if (testBtn) {
    testBtn.addEventListener('click', () => {
      DOM.diseaseModal.style.display = 'none';
      // Switch to diagnostic tab
      document.getElementById('btn-tab-diagnostics').click();
      selectSample(item.raw_class);
    });
  }
}

// ============================================================================
// AUTHENTICATION & SESSION MANAGEMENT
// ============================================================================

function openAuthModal(mode = 'login') {
  switchAuthTab(mode);
  if (DOM.authModal) DOM.authModal.style.display = 'flex';
  hideAuthAlert();
  setTimeout(() => {
    if (mode === 'register' && DOM.authName) DOM.authName.focus();
    else if (DOM.authEmail) DOM.authEmail.focus();
  }, 100);
}

function closeAuthModal() {
  if (DOM.authModal) DOM.authModal.style.display = 'none';
  hideAuthAlert();
}

function switchAuthTab(mode) {
  state.authMode = mode;
  hideAuthAlert();
  if (mode === 'login') {
    if (DOM.tabAuthLogin) DOM.tabAuthLogin.classList.add('active');
    if (DOM.tabAuthRegister) DOM.tabAuthRegister.classList.remove('active');
    if (DOM.groupAuthName) DOM.groupAuthName.style.display = 'none';
    if (DOM.authName) DOM.authName.required = false;
    if (DOM.authModalTitle) DOM.authModalTitle.textContent = 'Welcome to PlantCare AI';
    if (DOM.authModalSubtitle) DOM.authModalSubtitle.textContent = 'Sign in to access agronomic diagnosis & telemetry';
    if (DOM.authSubmitLabel) DOM.authSubmitLabel.textContent = 'Sign In to Diagnostic Studio';
  } else {
    if (DOM.tabAuthLogin) DOM.tabAuthLogin.classList.remove('active');
    if (DOM.tabAuthRegister) DOM.tabAuthRegister.classList.add('active');
    if (DOM.groupAuthName) DOM.groupAuthName.style.display = 'block';
    if (DOM.authName) DOM.authName.required = true;
    if (DOM.authModalTitle) DOM.authModalTitle.textContent = 'Create Agronomist Account';
    if (DOM.authModalSubtitle) DOM.authModalSubtitle.textContent = 'Register to sync disease inspections across devices';
    if (DOM.authSubmitLabel) DOM.authSubmitLabel.textContent = 'Create & Activate Account';
  }
}

function showAuthAlert(msg, type = 'error') {
  if (!DOM.authAlert) return;
  DOM.authAlert.className = `auth-alert ${type}`;
  DOM.authAlert.textContent = msg;
  DOM.authAlert.style.display = 'block';
}

function hideAuthAlert() {
  if (!DOM.authAlert) return;
  DOM.authAlert.style.display = 'none';
  DOM.authAlert.textContent = '';
}

async function handleAuthSubmit() {
  hideAuthAlert();
  const email = DOM.authEmail ? DOM.authEmail.value.trim() : '';
  const password = DOM.authPassword ? DOM.authPassword.value : '';
  const name = DOM.authName ? DOM.authName.value.trim() : '';

  if (!email || !password) {
    showAuthAlert('Please enter both email and password.', 'error');
    return;
  }

  if (state.authMode === 'register' && !name) {
    showAuthAlert('Please enter your full name.', 'error');
    return;
  }

  if (DOM.btnAuthSubmit) DOM.btnAuthSubmit.disabled = true;
  if (DOM.authSpinner) DOM.authSpinner.style.display = 'inline-block';
  if (DOM.authSubmitLabel) DOM.authSubmitLabel.textContent = state.authMode === 'register' ? 'Creating Account...' : 'Authenticating...';

  try {
    const endpoint = state.authMode === 'register' ? '/api/auth/register' : '/api/auth/login';
    const payload = state.authMode === 'register' 
      ? { email, password, full_name: name, role: 'Agronomist' }
      : { email, password };

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Authentication failed. Please verify your credentials.');
    }

    // Success
    state.authToken = data.token;
    state.currentUser = data.user;
    localStorage.setItem('plantcare_auth_token', data.token);
    localStorage.setItem('plantcare_user', JSON.stringify(data.user));

    showAuthAlert(state.authMode === 'register' ? 'Account created successfully! Welcome.' : 'Signed in successfully!', 'success');
    updateAuthUI(data.user);

    setTimeout(() => {
      closeAuthModal();
      if (DOM.authForm) DOM.authForm.reset();
    }, 600);

  } catch (err) {
    showAuthAlert(err.message, 'error');
  } finally {
    if (DOM.btnAuthSubmit) DOM.btnAuthSubmit.disabled = false;
    if (DOM.authSpinner) DOM.authSpinner.style.display = 'none';
    if (DOM.authSubmitLabel) DOM.authSubmitLabel.textContent = state.authMode === 'register' ? 'Create & Activate Account' : 'Sign In to Diagnostic Studio';
  }
}

function setupAuth() {
  const {
    btnOpenAuth,
    btnGateSignIn,
    btnCloseAuthModal,
    authModal,
    tabAuthLogin,
    tabAuthRegister,
    btnFillDemo,
    authForm,
    authEmail,
    authPassword,
    btnTogglePwd,
    btnToggleUserMenu,
    userDropdownCard,
    btnLogout
  } = DOM;

  // Open modal button in header
  if (btnOpenAuth) {
    btnOpenAuth.addEventListener('click', () => {
      openAuthModal('login');
    });
  }

  // Open modal button on studio gate banner
  if (btnGateSignIn) {
    btnGateSignIn.addEventListener('click', () => {
      openAuthModal('login');
    });
  }

  // Close modal button
  if (btnCloseAuthModal) {
    btnCloseAuthModal.addEventListener('click', () => {
      closeAuthModal();
    });
  }

  // Close modal on background backdrop click
  if (authModal) {
    authModal.addEventListener('click', (e) => {
      if (e.target === authModal) {
        closeAuthModal();
      }
    });
  }

  // Tab switching: Login
  if (tabAuthLogin) {
    tabAuthLogin.addEventListener('click', () => {
      switchAuthTab('login');
    });
  }

  // Tab switching: Register
  if (tabAuthRegister) {
    tabAuthRegister.addEventListener('click', () => {
      switchAuthTab('register');
    });
  }

  // 1-Click Demo Fill & Submit
  if (btnFillDemo) {
    btnFillDemo.addEventListener('click', async () => {
      switchAuthTab('login');
      if (authEmail) authEmail.value = 'demo@plantcare.ai';
      if (authPassword) authPassword.value = 'PlantCare123!';
      await handleAuthSubmit();
    });
  }

  // Password visibility toggle
  if (btnTogglePwd) {
    btnTogglePwd.addEventListener('click', () => {
      const isPwd = authPassword.type === 'password';
      authPassword.type = isPwd ? 'text' : 'password';
      btnTogglePwd.innerHTML = isPwd
        ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>`
        : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
    });
  }

  // Form Submission
  if (authForm) {
    authForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handleAuthSubmit();
    });
  }

  // User Profile Dropdown Toggle
  if (btnToggleUserMenu) {
    btnToggleUserMenu.addEventListener('click', (e) => {
      e.stopPropagation();
      const isVisible = userDropdownCard.style.display === 'block';
      userDropdownCard.style.display = isVisible ? 'none' : 'block';
      btnToggleUserMenu.setAttribute('aria-expanded', !isVisible);
    });
  }

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (userDropdownCard && userDropdownCard.style.display === 'block') {
      if (!userDropdownCard.contains(e.target) && !btnToggleUserMenu.contains(e.target)) {
        userDropdownCard.style.display = 'none';
        btnToggleUserMenu.setAttribute('aria-expanded', 'false');
      }
    }
  });

  // Logout
  if (btnLogout) {
    btnLogout.addEventListener('click', async () => {
      try {
        if (state.authToken) {
          await fetch('/api/auth/logout', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${state.authToken}` }
          });
        }
      } catch (err) {
        console.warn('Logout notification failed:', err);
      } finally {
        resetAuthUI();
      }
    });
  }
}

// Current User Session Check
async function checkCurrentUser() {
  if (!state.authToken) {
    resetAuthUI();
    return;
  }

  try {
    const res = await fetch('/api/auth/me', {
      headers: { 'Authorization': `Bearer ${state.authToken}` }
    });

    if (res.ok) {
      const data = await res.json();
      const user = data.user || data;
      state.currentUser = user;
      updateAuthUI(user);
    } else {
      resetAuthUI();
    }
  } catch (err) {
    console.warn('Auth check error:', err);
    // If cached user exists, keep offline view
    const cachedUser = localStorage.getItem('plantcare_user');
    if (cachedUser) {
      try {
        const u = JSON.parse(cachedUser);
        state.currentUser = u;
        updateAuthUI(u);
      } catch (e) {
        resetAuthUI();
      }
    } else {
      resetAuthUI();
    }
  }
}

function updateAuthUI(user) {
  if (!user) return;
  if (DOM.btnOpenAuth) DOM.btnOpenAuth.style.display = 'none';
  if (DOM.userProfileMenu) DOM.userProfileMenu.style.display = 'flex';
  if (DOM.authGateBanner) DOM.authGateBanner.style.display = 'none';

  const initials = getInitials(user.full_name || user.email);
  if (DOM.headerUserAvatar) DOM.headerUserAvatar.textContent = initials;
  if (DOM.headerUserName) DOM.headerUserName.textContent = user.full_name || user.email.split('@')[0];
  if (DOM.headerUserRole) DOM.headerUserRole.textContent = user.role || 'Agronomist';
  if (DOM.dropdownUserName) DOM.dropdownUserName.textContent = user.full_name || user.email;
  if (DOM.dropdownUserEmail) DOM.dropdownUserEmail.textContent = user.email;

  // Seamless UX: If user selected a sample or uploaded an image, run diagnosis right after login!
  if ((state.currentFile || state.currentSampleClass) && !state.lastDiagnosisResult) {
    runDiagnosis();
  }
}

function resetAuthUI() {
  state.currentUser = null;
  state.authToken = null;
  state.lastDiagnosisResult = null;
  localStorage.removeItem('plantcare_auth_token');
  localStorage.removeItem('plantcare_user');

  if (DOM.btnOpenAuth) DOM.btnOpenAuth.style.display = 'inline-flex';
  if (DOM.userProfileMenu) DOM.userProfileMenu.style.display = 'none';
  if (DOM.userDropdownCard) DOM.userDropdownCard.style.display = 'none';
  if (DOM.authGateBanner) DOM.authGateBanner.style.display = 'flex';

  if (DOM.emptyState) DOM.emptyState.style.display = 'block';
  if (DOM.activeState) DOM.activeState.style.display = 'none';
  if (DOM.loadingState) DOM.loadingState.style.display = 'none';
}

function getInitials(nameStr) {
  if (!nameStr) return 'AI';
  const clean = nameStr.replace(/^(Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.)\s*/i, '').trim();
  const parts = clean.split(/\s+/).filter(Boolean);
  if (parts.length === 0) return nameStr.slice(0, 2).toUpperCase();
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

