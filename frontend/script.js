/**
 * Quellum — A Hallucination Aware RAG
 * Vanilla JavaScript Controller (No React)
 * Supports Home View, Workspace Two-Panel View, Evidence Grounding, and Modals
 */

document.addEventListener('DOMContentLoaded', () => {

  // PDF.js needs its worker script; without this line the preview fails to render.
  if (window.pdfjsLib && window.pdfjsLib.GlobalWorkerOptions) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  }

  // =========================================================================
  // 1. Navigation & View Routing (Single Page Architecture)
  // =========================================================================
  const homeView = document.getElementById('homeView');
  const workspaceView = document.getElementById('workspaceView');
  const historyView = document.getElementById('historyView');

  const navHomeBtn = document.getElementById('navHomeBtn');
  const navWorkspaceBtn = document.getElementById('navWorkspaceBtn');
  const navHistoryBtn = document.getElementById('navHistoryBtn');
  const navLogoBtn = document.getElementById('navLogoBtn');
  const homeActiveIndicator = document.getElementById('homeActiveIndicator');
  const workspaceActiveIndicator = document.getElementById('workspaceActiveIndicator');
  const historyActiveIndicator = document.getElementById('historyActiveIndicator');

  const themeToggleBtn = document.getElementById('themeToggleBtn');
  let sessionReady = Promise.resolve();

  function switchView(target) {
    if (target === 'history') {
      if (homeView) homeView.style.display = 'none';
      if (workspaceView) workspaceView.style.display = 'none';
      if (historyView) historyView.style.display = 'flex';

      // Update Navigation styling
      if (navHomeBtn) navHomeBtn.classList.add('inactive');
      if (homeActiveIndicator) homeActiveIndicator.style.display = 'none';

      if (navWorkspaceBtn) navWorkspaceBtn.classList.add('inactive');
      if (workspaceActiveIndicator) workspaceActiveIndicator.style.display = 'none';

      if (navHistoryBtn) navHistoryBtn.classList.remove('inactive');
      if (historyActiveIndicator) historyActiveIndicator.style.display = 'block';

      if (typeof refreshAndRenderHistory === 'function') {
        refreshAndRenderHistory();
      }

      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (target === 'workspace') {
      if (homeView) homeView.style.display = 'none';
      if (historyView) historyView.style.display = 'none';
      if (workspaceView) workspaceView.style.display = 'flex';

      // Update Navigation styling
      if (navHomeBtn) navHomeBtn.classList.add('inactive');
      if (homeActiveIndicator) homeActiveIndicator.style.display = 'none';

      if (navWorkspaceBtn) navWorkspaceBtn.classList.remove('inactive');
      if (workspaceActiveIndicator) workspaceActiveIndicator.style.display = 'block';

      if (navHistoryBtn) navHistoryBtn.classList.add('inactive');
      if (historyActiveIndicator) historyActiveIndicator.style.display = 'none';

      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      if (workspaceView) workspaceView.style.display = 'none';
      if (historyView) historyView.style.display = 'none';
      if (homeView) homeView.style.display = 'flex';

      // Update Navigation styling
      if (navHomeBtn) navHomeBtn.classList.remove('inactive');
      if (homeActiveIndicator) homeActiveIndicator.style.display = 'block';

      if (navWorkspaceBtn) navWorkspaceBtn.classList.add('inactive');
      if (workspaceActiveIndicator) workspaceActiveIndicator.style.display = 'none';

      if (navHistoryBtn) navHistoryBtn.classList.add('inactive');
      if (historyActiveIndicator) historyActiveIndicator.style.display = 'none';
    }
  }

  // Handle URL hash changes
  function handleRoute() {
    const hash = (window.location.hash || '').toLowerCase();
    if (hash === '#workspace') {
      switchView('workspace');
      if (window.QuellumAPI) {
        sessionReady.then(restoreActiveDocument).catch((err) => {
          if (readActiveDocumentId()) alert(apiErrorMessage(err));
        });
      }
    } else if (hash === '#history') {
      switchView('history');
    } else {
      switchView('home');
    }
  }

  window.addEventListener('hashchange', handleRoute);

  if (navWorkspaceBtn) {
    navWorkspaceBtn.addEventListener('click', (e) => {
      e.preventDefault();
      switchView('workspace');
      window.location.hash = '#workspace';
    });
  }

  if (navHomeBtn) {
    navHomeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      switchView('home');
      window.location.hash = '#home';
    });
  }

  if (navHistoryBtn) {
    navHistoryBtn.addEventListener('click', (e) => {
      e.preventDefault();
      switchView('history');
      window.location.hash = '#history';
    });
  }

  if (navLogoBtn) {
    navLogoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      switchView('home');
      window.location.hash = '#home';
    });
  }

  // =========================================================================
  // 2. Modals (Analysis, History)
  // =========================================================================
  const analysisModal = document.getElementById('analysisModal');
  const analysisSubtitle = document.getElementById('analysisSubtitle');
  const closeAnalysisModalBtn = document.getElementById('closeAnalysisModalBtn');
  const doneAnalysisBtn = document.getElementById('doneAnalysisBtn');

  const historyModal = document.getElementById('historyModal');
  const closeHistoryModalBtn = document.getElementById('closeHistoryModalBtn');
  const closeHistoryFooterBtn = document.getElementById('closeHistoryFooterBtn');

  function openModal(modal) {
    if (!modal) return;
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal(modal) {
    if (!modal) return;
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }

  function openAnalysisModal(subtitle) {
    if (analysisSubtitle) analysisSubtitle.textContent = subtitle;
    openModal(analysisModal);
  }

  [closeAnalysisModalBtn, doneAnalysisBtn].forEach(btn => {
    if (btn) btn.addEventListener('click', () => closeModal(analysisModal));
  });

  [closeHistoryModalBtn, closeHistoryFooterBtn].forEach(btn => {
    if (btn) btn.addEventListener('click', () => closeModal(historyModal));
  });

  const historyDeleteModal = document.getElementById('historyDeleteModal');
  const historyRenameModal = document.getElementById('historyRenameModal');

  // Close modals on overlay backdrop click
  [analysisModal, historyModal, historyDeleteModal, historyRenameModal].forEach(modal => {
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal(modal);
      });
    }
  });

  // Close modals on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      [analysisModal, historyModal].forEach(modal => closeModal(modal));
    }
  });

  // Theme toggle
  let isDimmed = false;
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      isDimmed = !isDimmed;
      if (isDimmed) {
        document.body.style.filter = 'contrast(102%) brightness(96%)';
        themeToggleBtn.title = 'Switch to normal brightness';
      } else {
        document.body.style.filter = '';
        themeToggleBtn.title = 'Toggle theme';
      }
    });
  }

  // =========================================================================
  // 3. Home Page Interactions (Preserved exactly as working)
  // =========================================================================
  const pdfFileInput = document.getElementById('pdfFileInput');
  const dropzoneBox = document.getElementById('dropzoneBox');
  const choosePdfBtn = document.getElementById('choosePdfBtn');
  const choosePdfBtnText = document.getElementById('choosePdfBtnText');
  const selectedFileChip = document.getElementById('selectedFileChip');
  const fileNameLabel = document.getElementById('fileNameLabel');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const dropzoneTitle = document.getElementById('dropzoneTitle');
  const dropzoneDesc = document.getElementById('dropzoneDesc');

  const tabUploadBtn = document.getElementById('tabUploadBtn');
  const tabPasteBtn = document.getElementById('tabPasteBtn');

  const pasteInput = document.getElementById('pasteInput');
  const trySampleBtn = document.getElementById('trySampleBtn');
  const wordCountLabel = document.getElementById('wordCountLabel');
  const analyzeTextBtn = document.getElementById('analyzeTextBtn');
  const analyzeTextBtnLabel = document.getElementById('analyzeTextBtnLabel');
  const pasteBox = document.getElementById('pasteBox');
  const homeDividerOr = document.getElementById('homeDividerOr');
  const homePanelGrid = document.querySelector('.upload-panel .panel-grid');

  let homeSelectedFile = null;

  function setHomeActiveTab(tab) {
    if (tab === 'upload') {
      if (tabUploadBtn) {
        tabUploadBtn.classList.add('active');
        tabUploadBtn.setAttribute('aria-selected', 'true');
      }
      if (tabPasteBtn) {
        tabPasteBtn.classList.remove('active');
        tabPasteBtn.setAttribute('aria-selected', 'false');
      }
      if (dropzoneBox) dropzoneBox.style.display = 'flex';
      if (pasteBox) pasteBox.style.display = 'none';
      if (homeDividerOr) homeDividerOr.style.display = 'none';
      if (homePanelGrid) homePanelGrid.style.gridTemplateColumns = '1fr';
    } else {
      if (tabPasteBtn) {
        tabPasteBtn.classList.add('active');
        tabPasteBtn.setAttribute('aria-selected', 'true');
      }
      if (tabUploadBtn) {
        tabUploadBtn.classList.remove('active');
        tabUploadBtn.setAttribute('aria-selected', 'false');
      }
      if (dropzoneBox) dropzoneBox.style.display = 'none';
      if (pasteBox) pasteBox.style.display = 'flex';
      if (homeDividerOr) homeDividerOr.style.display = 'none';
      if (homePanelGrid) homePanelGrid.style.gridTemplateColumns = '1fr';
      if (pasteInput) {
        setTimeout(() => pasteInput.focus(), 50);
      }
    }
  }

  if (tabUploadBtn) tabUploadBtn.addEventListener('click', () => setHomeActiveTab('upload'));
  if (tabPasteBtn) tabPasteBtn.addEventListener('click', () => setHomeActiveTab('paste'));

  function handleHomeSelectedFile(file) {
    if (!file) return;
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a valid PDF file.');
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      alert('File exceeds the maximum size limit of 20 MB.');
      return;
    }

    homeSelectedFile = file;
    if (fileNameLabel) fileNameLabel.textContent = file.name;
    if (selectedFileChip) selectedFileChip.style.display = 'flex';
    if (dropzoneTitle) dropzoneTitle.textContent = 'PDF Document Ready';
    if (dropzoneDesc) dropzoneDesc.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB · Verified format';
    if (choosePdfBtnText) choosePdfBtnText.textContent = 'Analyze PDF';
    setHomeActiveTab('upload');
    performAnalyzeHomePdf();          // upload and go straight to the workspace
  }

  function resetHomeFileSelection() {
    homeSelectedFile = null;
    if (pdfFileInput) pdfFileInput.value = '';
    if (selectedFileChip) selectedFileChip.style.display = 'none';
    if (dropzoneTitle) dropzoneTitle.textContent = 'Upload a PDF';
    if (dropzoneDesc) dropzoneDesc.textContent = 'Check answers against the contents of your document.';
    if (choosePdfBtnText) choosePdfBtnText.textContent = 'Choose PDF';
  }

  // Flow A: Analyze PDF from Home. The PDF is uploaded once to the private, session-scoped store; the Workspace then shows the same file.
  let homeAnalyzing = false;
  async function performAnalyzeHomePdf() {
    if (!homeSelectedFile) {
      if (pdfFileInput) pdfFileInput.click();
      return;
    }
    if (homeAnalyzing) return;
    homeAnalyzing = true;
    try {
      const goToWorkspace = () => {
        resetHomeFileSelection();
        switchView('workspace');
        window.location.hash = '#workspace';
        setTimeout(() => { if (wsChatTextarea) wsChatTextarea.focus(); }, 150);
      };
      const ok = await registerAndLoadPdf(homeSelectedFile, choosePdfBtnText, goToWorkspace);
      if (!ok) return;
      goToWorkspace();   // no-op safety net if the callback did not run
    } finally {
      homeAnalyzing = false;
    }
  }

  // Flow B: Analyze Text from Home
  async function performAnalyzeHomeText() {
    const text = pasteInput ? pasteInput.value.trim() : '';
    if (!text) {
      alert('Please enter or paste some text first.');
      if (pasteInput) pasteInput.focus();
      return;
    }
    if (homeAnalyzing) return;
    homeAnalyzing = true;
    try {
      const ok = await registerAndLoadText(text, 'Text Source', analyzeTextBtnLabel);
      if (!ok) return;
      switchView('workspace');
      window.location.hash = '#workspace';
      setTimeout(() => { if (wsChatTextarea) wsChatTextarea.focus(); }, 150);
    } finally {
      homeAnalyzing = false;
    }
  }

  if (dropzoneBox) {
    dropzoneBox.addEventListener('click', (e) => {
      if (e.target.closest('#removeFileBtn')) return;
      if (homeSelectedFile) {
        performAnalyzeHomePdf();
      } else if (pdfFileInput) {
        pdfFileInput.click();
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropzoneBox.addEventListener(name, (e) => {
        e.preventDefault();
        dropzoneBox.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzoneBox.addEventListener(name, (e) => {
        e.preventDefault();
        dropzoneBox.classList.remove('drag-active');
      });
    });

    dropzoneBox.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleHomeSelectedFile(dt.files[0]);
      }
    });
  }

  if (choosePdfBtn) {
    choosePdfBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (homeSelectedFile) {
        performAnalyzeHomePdf();
      } else if (pdfFileInput) {
        pdfFileInput.click();
      }
    });
  }

  if (pdfFileInput) {
    pdfFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleHomeSelectedFile(e.target.files[0]);
      }
    });
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetHomeFileSelection();
    });
  }

  function updateHomeWordCount() {
    if (!pasteInput || !wordCountLabel) return;
    const val = pasteInput.value.trim();
    if (!val) {
      wordCountLabel.textContent = 'Evidence check ready';
      return;
    }
    const words = val.split(/\s+/).filter(Boolean).length;
    wordCountLabel.textContent = `${words} word${words === 1 ? '' : 's'} · ${val.length} chars`;
  }

  if (pasteInput) pasteInput.addEventListener('input', updateHomeWordCount);

  if (trySampleBtn) {
    trySampleBtn.addEventListener('click', () => {
      if (pasteInput) {
        pasteInput.value = "Recent studies suggest that LLM hallucinatory artifacts can be mitigated using dual cross-attention verifiers. However, model inference latency increases by 34% when cross-checking external token logits. Token loss penalties improve overall calibration on RAGTruth benchmark tasks.";
        updateHomeWordCount();
        setHomeActiveTab('paste');
      }
    });
  }

  if (analyzeTextBtn) {
    analyzeTextBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      performAnalyzeHomeText();
    });
  }


  // =========================================================================
  // 4. Workspace Page Controller (Left Panel Source/Preview + Right Panel Chat)
  // =========================================================================

  // --- Document State ---
  const wsState = {
    isLoaded: false,
    type: null, // 'pdf' | 'text'
    name: '',
    pages: 4,
    currentPage: 1,
    zoom: 100,
    pdfDoc: null,
    textContent: '',
    chatHistory: []
  };

  // Reference document data matching the user's reference image
  const sampleDocPages = [
    {
      pageNumber: 1,
      title: "Introduction to Machine Learning",
      sections: [
        {
          heading: "1. What is Machine Learning?",
          body: "Machine Learning (ML) is a subset of Artificial Intelligence (AI) that allows systems to learn from data, identify patterns, and make decisions or predictions without being explicitly programmed."
        },
        {
          heading: "2. Types of Machine Learning",
          bullets: [
            "<strong>Supervised Learning</strong> – learns from labeled data (e.g., classification, regression).",
            "<strong>Unsupervised Learning</strong> – finds hidden patterns in unlabeled data (e.g., clustering, dimensionality reduction).",
            "<strong>Reinforcement Learning</strong> – learns through rewards and penalties (e.g., game playing, robotics)."
          ]
        },
        {
          heading: "3. Common Algorithms",
          bullets: [
            "Linear Regression",
            "Decision Trees",
            "Support Vector Machines (SVM)",
            "Neural Networks"
          ]
        },
        {
          heading: "4. Applications",
          bullets: [
            "Healthcare (disease prediction, medical imaging)",
            "Finance (fraud detection, risk analysis)",
            "Retail (recommendation systems)",
            "Autonomous vehicles (object detection, path planning)"
          ]
        }
      ]
    },
    {
      pageNumber: 2,
      title: "Supervised Learning Fundamentals",
      sections: [
        {
          heading: "Empirical Risk Minimization & Objective Functions",
          body: "Supervised algorithms learn a mapping hypothesis h: X -> Y by minimizing regularized empirical risk over training instances. The choice of loss function directly influences gradient stability during backpropagation."
        },
        {
          heading: "Classification Metrics",
          bullets: [
            "Precision & Recall: Essential when classes are imbalanced.",
            "F1-Score: Harmonic mean of precision and recall.",
            "Receiver Operating Characteristic (ROC-AUC): Evaluates discrimination across varying decision thresholds."
          ]
        },
        {
          heading: "Regularization Methods",
          body: "L1 (Lasso) penalty induces sparsity by shrinking coefficients to zero, functioning as implicit feature selection. L2 (Ridge) penalty constrains weight magnitudes to prevent overfitting high-variance signals."
        }
      ]
    },
    {
      pageNumber: 3,
      title: "Unsupervised & Clustering Methods",
      sections: [
        {
          heading: "Centroid-Based & Hierarchical Clustering",
          body: "K-Means iteratively partitions data points into k Voronoi cells. Hierarchical clustering constructs dendrograms without requiring prior assumption of the cluster count k."
        },
        {
          heading: "Dimensionality Reduction",
          bullets: [
            "Principal Component Analysis (PCA): Projects variance onto orthogonal eigenvectors.",
            "t-SNE & UMAP: Non-linear manifold approximations suitable for high-dimensional visual exploration."
          ]
        },
        {
          heading: "Reinforcement Learning Feedback Loops",
          body: "An agent interacts with an environment defined as a Markov Decision Process (MDP). Bellman equations govern policy iteration to maximize discounted cumulative future rewards."
        }
      ]
    },
    {
      pageNumber: 4,
      title: "Verification, Ethics & Enterprise Deployment",
      sections: [
        {
          heading: "Retrieval-Augmented Generation (RAG) Grounding",
          body: "Modern enterprise architectures combine external dense vector retrieval with LLM generation to ground every factual assertion in verified source passages."
        },
        {
          heading: "Hallucination Mitigation Strategies",
          bullets: [
            "Dual-encoder cross checking to verify token probabilities.",
            "Citation tagging connecting sentences to specific source page offsets.",
            "Threshold-based guardrails flagging ungrounded claims before presentation to the user."
          ]
        },
        {
          heading: "Production Monitoring & Drift",
          body: "Tracking data drift and concept drift maintains operational model integrity over time as input data distributions evolve."
        }
      ]
    }
  ];

  // DOM Elements - Left Panel
  const wsPreviewHeader = document.getElementById('wsPreviewHeader');
  const wsDocBadge = document.getElementById('wsDocBadge');
  const wsDocTitle = document.getElementById('wsDocTitle');
  const wsCloseDocBtn = document.getElementById('wsCloseDocBtn');

  const wsPreviewToolbar = document.getElementById('wsPreviewToolbar');
  const wsUploadNewBtn = document.getElementById('wsUploadNewBtn');
  const wsPageCountLabel = document.getElementById('wsPageCountLabel');

  const wsPreviewBody = document.getElementById('wsPreviewBody');
  const wsThumbnailsList = document.getElementById('wsThumbnailsList');
  const wsDocScrollArea = document.getElementById('wsDocScrollArea');
  const wsPagesContainer = document.getElementById('wsPagesContainer');

  const wsZoomInBtn = document.getElementById('wsZoomInBtn');
  const wsZoomOutBtn = document.getElementById('wsZoomOutBtn');
  const wsZoomLabel = document.getElementById('wsZoomLabel');
  const wsFitWidthBtn = document.getElementById('wsFitWidthBtn');

  // DOM Elements - Initial Empty State (Left Panel)
  const wsInitialInputState = document.getElementById('wsInitialInputState');
  const wsTabUploadBtn = document.getElementById('wsTabUploadBtn');
  const wsTabPasteBtn = document.getElementById('wsTabPasteBtn');
  const wsUploadTabView = document.getElementById('wsUploadTabView');
  const wsPasteTabView = document.getElementById('wsPasteTabView');

  const wsDropzoneBox = document.getElementById('wsDropzoneBox');
  const wsChoosePdfBtn = document.getElementById('wsChoosePdfBtn');
  const wsFileInput = document.getElementById('wsFileInput');
  const wsPasteTextarea = document.getElementById('wsPasteTextarea');
  const wsUseTextBtn = document.getElementById('wsUseTextBtn');

  // DOM Elements - Right Panel (Chat)
  const wsClearChatBtn = document.getElementById('wsClearChatBtn');
  const wsChatBody = document.getElementById('wsChatBody');
  const wsChatEmptyState = document.getElementById('wsChatEmptyState');
  const wsChatThreadList = document.getElementById('wsChatThreadList');
  const wsChatTextarea = document.getElementById('wsChatTextarea');
  const wsChatSendBtn = document.getElementById('wsChatSendBtn');
  const wsChatSourceChip = document.getElementById('wsChatSourceChip');
  const wsChatSourceLabel = document.getElementById('wsChatSourceLabel');

  // Workspace active conversation session ID
  let wsActiveSessionId = null;
  let activeDocumentRestore = null;

  // --- Server-registered documents (private, tied to the anonymous session) ---
  const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
  const ACTIVE_DOCUMENT_KEY = 'quellum_active_document_id';

  function readActiveDocumentId() {
    try {
      return localStorage.getItem(ACTIVE_DOCUMENT_KEY);
    } catch (err) {
      console.warn('Active document could not be read from browser storage.', err);
      return null;
    }
  }

  function rememberActiveDocument(documentId) {
    try {
      localStorage.setItem(ACTIVE_DOCUMENT_KEY, documentId);
    } catch (err) {
      console.warn('Active document could not be saved to browser storage.', err);
    }
  }

  function forgetActiveDocument() {
    try {
      localStorage.removeItem(ACTIVE_DOCUMENT_KEY);
    } catch (err) {
      console.warn('Active document could not be cleared from browser storage.', err);
    }
  }

  function apiErrorMessage(err) {
    if (err && err.name === 'ApiError') return err.message;
    return 'Something went wrong. Please try again.';
  }

  // A different document means a different conversation: clear the chat so the old document's context can never leak into the new one.
  function newWorkspaceDocument(documentId) {
    wsState.documentId = documentId;
    rememberActiveDocument(documentId);
    wsActiveSessionId = null;
    resetChatUi();
  }

  async function restoreActiveDocument() {
    const documentId = readActiveDocumentId();
    if (!documentId || wsState.isLoaded) return;
    if (activeDocumentRestore) return activeDocumentRestore;

    activeDocumentRestore = (async () => {
      try {
        const doc = await window.QuellumAPI.getDocument(documentId);
        wsState.documentId = doc.id;
        wsActiveSessionId = null;
        if (doc.sourceType === 'pdf') {
          const bytes = await window.QuellumAPI.getDocumentFile(doc.id);
          await loadPdfIntoWorkspace(new File([bytes], doc.name, { type: 'application/pdf' }));
        } else if (doc.sourceType === 'text') {
          loadTextIntoWorkspace(doc.textContent || '', doc.name || 'Text Source');
        } else {
          throw new Error('The saved document has an unsupported type.');
        }
      } catch (err) {
        if (err && err.name === 'ApiError' && (err.code === 'expired' || err.code === 'not_found')) {
          forgetActiveDocument();
          return;
        }
        alert(apiErrorMessage(err));
      } finally {
        activeDocumentRestore = null;
      }
    })();
    return activeDocumentRestore;
  }

  async function registerAndLoadPdf(file, labelEl, onUploaded) {
    if (file.type !== 'application/pdf' && !/\.pdf$/i.test(file.name)) {
      alert('Please select a valid PDF file.');
      return false;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      alert('File exceeds the maximum size limit of 20 MB.');
      return false;
    }
    const prev = labelEl ? labelEl.textContent : '';
    if (labelEl) labelEl.textContent = 'Uploading…';
    try {
      const doc = await window.QuellumAPI.uploadPdf(file);
      newWorkspaceDocument(doc.id);
      const loading = loadPdfIntoWorkspace(file);      // its synchronous part marks the workspace as loaded
      if (typeof onUploaded === 'function') onUploaded();   // redirect now; the preview keeps rendering in the workspace
      await loading;
      return true;
    } catch (err) {
      alert(apiErrorMessage(err));
      return false;
    } finally {
      if (labelEl) labelEl.textContent = prev;
    }
  }

  async function registerAndLoadText(text, name, labelEl) {
    const prev = labelEl ? labelEl.textContent : '';
    if (labelEl) labelEl.textContent = 'Saving…';
    try {
      const doc = await window.QuellumAPI.createTextDocument(text, name);
      newWorkspaceDocument(doc.id);
      loadTextIntoWorkspace(text, name);
      return true;
    } catch (err) {
      alert(apiErrorMessage(err));
      return false;
    } finally {
      if (labelEl) labelEl.textContent = prev;
    }
  }

  // --- Document Loading: real PDF rendered with PDF.js ---
  async function loadPdfIntoWorkspace(fileOrSample) {
    wsState.isLoaded = true;
    wsState.type = 'pdf';
    wsState.zoom = 100;
    wsState.currentPage = 1;
    updateZoomDisplay();

    // Check if it's a File object or sample trigger
    if (fileOrSample instanceof File) {
      wsState.name = fileOrSample.name;
      wsState.documentId = wsState.documentId || ('doc_' + Date.now());
      wsDocTitle.textContent = fileOrSample.name;
      wsDocBadge.textContent = 'PDF';
      wsDocBadge.className = 'badge-pdf-red';
      wsChatSourceLabel.textContent = fileOrSample.name;

      // Try PDF.js if available
      if (window.pdfjsLib) {
        try {
          const fileData = await fileOrSample.arrayBuffer();
          const loadingTask = window.pdfjsLib.getDocument({ data: fileData });
          const pdfDoc = await Promise.race([
            loadingTask.promise,
            new Promise((_, reject) => setTimeout(() => reject(new Error('PDF.js timed out')), 20000)),
          ]);
          wsState.pdfDoc = pdfDoc;
          wsState.pages = pdfDoc.numPages;
          transitionToPreview();
          await renderRealPdfPages(pdfDoc);
          return;
        } catch (err) {
          console.warn('PDF.js rendering fallback to high-fidelity document viewer:', err);
        }
      }
      // PDF.js unavailable or the file could not be rendered: say so instead of showing made-up pages
      wsState.pages = 1;
      wsPagesContainer.innerHTML = '<div style="padding: 40px 24px; text-align: center; color: #475569; font-size: 14px; line-height: 1.6;">' +
        'A page preview is not available for this file in your browser.<br>The document was saved and you can still ask questions about it.</div>';
      if (wsThumbnailsList) wsThumbnailsList.innerHTML = '';
      transitionToPreview();
    }
  }

  function loadTextIntoWorkspace(text, titleName = 'pasted_source.txt') {
    if (!text || !text.trim()) {
      alert('Please enter or paste some text first.');
      return;
    }
    wsState.isLoaded = true;
    wsState.type = 'text';
    wsState.name = titleName;
    wsState.documentId = wsState.documentId || ('doc_txt_' + Date.now());
    wsState.textContent = text.trim();
    wsState.pages = 1;
    wsState.currentPage = 1;
    wsState.zoom = 100;
    updateZoomDisplay();

    wsDocTitle.textContent = titleName;
    wsDocBadge.textContent = 'TXT';
    wsDocBadge.className = 'badge-text-purple';
    wsChatSourceLabel.textContent = titleName;

    renderPastedTextDocument(text);
    transitionToPreview();
  }

  function transitionToPreview() {
    wsInitialInputState.style.display = 'none';
    wsPreviewHeader.style.display = 'flex';
    wsPreviewToolbar.style.display = 'flex';
    wsPreviewBody.style.display = 'grid';

    if (wsPageCountLabel) {
      wsPageCountLabel.textContent = `Pages: ${wsState.pages}`;
    }

    // Scroll preview to top
    if (wsDocScrollArea) wsDocScrollArea.scrollTop = 0;
  }

  function switchWorkspaceTab(tab) {
    if (tab === 'upload') {
      if (wsTabUploadBtn) {
        wsTabUploadBtn.classList.add('active');
        wsTabUploadBtn.setAttribute('aria-selected', 'true');
      }
      if (wsTabPasteBtn) {
        wsTabPasteBtn.classList.remove('active');
        wsTabPasteBtn.setAttribute('aria-selected', 'false');
      }
      if (wsUploadTabView) wsUploadTabView.style.display = 'flex';
      if (wsPasteTabView) wsPasteTabView.style.display = 'none';
    } else if (tab === 'paste') {
      if (wsTabPasteBtn) {
        wsTabPasteBtn.classList.add('active');
        wsTabPasteBtn.setAttribute('aria-selected', 'true');
      }
      if (wsTabUploadBtn) {
        wsTabUploadBtn.classList.remove('active');
        wsTabUploadBtn.setAttribute('aria-selected', 'false');
      }
      if (wsPasteTabView) wsPasteTabView.style.display = 'flex';
      if (wsUploadTabView) wsUploadTabView.style.display = 'none';
      if (wsPasteTextarea) wsPasteTextarea.focus();
    }
  }

  function resetWorkspaceDocument() {
    forgetActiveDocument();
    wsState.documentId = null;
    wsActiveSessionId = null;
    resetChatUi();
    wsState.isLoaded = false;
    wsState.type = null;
    wsState.name = '';
    wsState.pdfDoc = null;
    wsState.textContent = '';

    if (wsPreviewHeader) wsPreviewHeader.style.display = 'none';
    if (wsPreviewToolbar) wsPreviewToolbar.style.display = 'none';
    if (wsPreviewBody) wsPreviewBody.style.display = 'none';
    if (wsInitialInputState) wsInitialInputState.style.display = 'flex';

    switchWorkspaceTab('upload');

    if (wsPagesContainer) wsPagesContainer.innerHTML = '';
    if (wsThumbnailsList) wsThumbnailsList.innerHTML = '';

    if (wsChatSourceLabel) wsChatSourceLabel.textContent = 'Upload PDF or Paste Text';
  }

  if (wsCloseDocBtn) wsCloseDocBtn.addEventListener('click', resetWorkspaceDocument);
  if (wsUploadNewBtn) wsUploadNewBtn.addEventListener('click', resetWorkspaceDocument);

  async function renderRealPdfPages(pdfDoc) {
    wsPagesContainer.innerHTML = '';
    wsThumbnailsList.innerHTML = '';

    const numPages = pdfDoc.numPages;

    for (let i = 1; i <= numPages; i++) {
      const page = await pdfDoc.getPage(i);

      // 1. Thumbnail Card
      const thumbWrap = document.createElement('div');
      thumbWrap.style.display = 'flex';
      thumbWrap.style.flexDirection = 'column';
      thumbWrap.style.alignItems = 'center';

      const thumbCard = document.createElement('div');
      thumbCard.className = `thumb-card ${i === 1 ? 'active' : ''}`;
      thumbCard.id = `wsThumb_${i}`;
      thumbCard.setAttribute('data-page', i);

      const thumbCanvas = document.createElement('canvas');
      thumbCanvas.style.width = '100%';
      thumbCanvas.style.height = 'auto';
      thumbCanvas.style.borderRadius = '4px';

      const thumbViewport = page.getViewport({ scale: 0.15 });
      thumbCanvas.height = thumbViewport.height;
      thumbCanvas.width = thumbViewport.width;
      const thumbCtx = thumbCanvas.getContext('2d');
      page.render({ canvasContext: thumbCtx, viewport: thumbViewport });

      thumbCard.appendChild(thumbCanvas);

      const pageTag = document.createElement('span');
      pageTag.className = 'thumb-page-num';
      pageTag.textContent = i;

      thumbWrap.appendChild(thumbCard);
      thumbWrap.appendChild(pageTag);
      wsThumbnailsList.appendChild(thumbWrap);

      thumbCard.addEventListener('click', () => scrollToPage(i));

      // 2. Main Page Canvas Container
      const pageContainer = document.createElement('div');
      pageContainer.className = 'pdf-page-container';
      pageContainer.id = `wsPage_${i}`;
      pageContainer.style.padding = '12px';
      pageContainer.style.display = 'flex';
      pageContainer.style.flexDirection = 'column';
      pageContainer.style.alignItems = 'center';

      const pageCanvas = document.createElement('canvas');
      const pageViewport = page.getViewport({ scale: 1.25 });
      pageCanvas.height = pageViewport.height;
      pageCanvas.width = pageViewport.width;
      pageCanvas.style.width = '100%';
      pageCanvas.style.height = 'auto';
      pageCanvas.style.borderRadius = '8px';

      const pageCtx = pageCanvas.getContext('2d');
      page.render({ canvasContext: pageCtx, viewport: pageViewport });

      pageContainer.appendChild(pageCanvas);
      wsPagesContainer.appendChild(pageContainer);
    }

    setupScrollObserver();
  }

  // --- Render Pasted Text Document ---
  function renderPastedTextDocument(text) {
    wsPagesContainer.innerHTML = '';
    wsThumbnailsList.innerHTML = '';

    // Thumbnail for single document
    const thumbWrap = document.createElement('div');
    thumbWrap.style.display = 'flex';
    thumbWrap.style.flexDirection = 'column';
    thumbWrap.style.alignItems = 'center';

    const thumbCard = document.createElement('div');
    thumbCard.className = 'thumb-card active';
    thumbCard.id = 'wsThumb_1';
    thumbCard.innerHTML = `
      <div style="font-size: 6px; font-weight: 700; color: #7c3aed; margin-bottom: 3px;">PASTED TEXT</div>
      <div class="thumb-mini-lines">
        <div class="thumb-mini-line"></div>
        <div class="thumb-mini-line"></div>
        <div class="thumb-mini-line"></div>
        <div class="thumb-mini-line"></div>
      </div>
    `;

    const pageTag = document.createElement('span');
    pageTag.className = 'thumb-page-num';
    pageTag.textContent = '1';

    thumbWrap.appendChild(thumbCard);
    thumbWrap.appendChild(pageTag);
    wsThumbnailsList.appendChild(thumbWrap);

    // Format paragraphs
    const paragraphs = text.split(/\n+/).filter(Boolean);
    const words = text.split(/\s+/).filter(Boolean).length;

    const paper = document.createElement('div');
    paper.className = 'text-paper-container';
    paper.id = 'wsPage_1';

    let parasHtml = paragraphs.map(p => `<p class="pdf-rendered-p" style="font-size: 14.5px; line-height: 1.7; margin-bottom: 16px;">${escapeHtml(p)}</p>`).join('');

    paper.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 14px; margin-bottom: 24px;">
        <div>
          <span class="badge-text-purple">Pasted Text Passage</span>
          <h2 style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-top: 8px;">Source Material</h2>
        </div>
        <div style="text-align: right; font-size: 11.5px; color: #64748b;">
          <div>${words} words · ${text.length} chars</div>
          <button id="wsEditTextBtn" style="margin-top: 4px; background: none; border: none; color: #2563eb; font-weight: 600; font-size: 11px; cursor: pointer; text-decoration: underline;">Edit Text</button>
        </div>
      </div>
      <div>
        ${parasHtml}
      </div>
    `;

    wsPagesContainer.appendChild(paper);

    const editBtn = document.getElementById('wsEditTextBtn');
    if (editBtn) {
      editBtn.addEventListener('click', () => {
        resetWorkspaceDocument();
        switchWorkspaceTab('paste');
        if (wsPasteTextarea) wsPasteTextarea.value = text;
      });
    }
  }

  function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, 
      tag => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
      }[tag] || tag)
    );
  }

  // --- Scroll & Active Thumbnail Synchronization ---
  function scrollToPage(pageIndex) {
    const pageEl = document.getElementById(`wsPage_${pageIndex}`);
    if (pageEl) {
      pageEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setActiveThumbnail(pageIndex);
    }
  }

  function setActiveThumbnail(pageIndex) {
    wsState.currentPage = pageIndex;
    const thumbs = wsThumbnailsList.querySelectorAll('.thumb-card');
    thumbs.forEach(t => {
      const p = parseInt(t.getAttribute('data-page'), 10);
      if (p === pageIndex) {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    });
  }

  let scrollTimeout = null;
  function setupScrollObserver() {
    if (!wsDocScrollArea) return;
    wsDocScrollArea.addEventListener('scroll', () => {
      if (scrollTimeout) clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        const pages = wsPagesContainer.querySelectorAll('.pdf-page-container, .text-paper-container');
        const containerTop = wsDocScrollArea.getBoundingClientRect().top;

        let closestPage = 1;
        let minDiff = Infinity;

        pages.forEach((pageEl, idx) => {
          const rect = pageEl.getBoundingClientRect();
          const diff = Math.abs(rect.top - containerTop);
          if (diff < minDiff) {
            minDiff = diff;
            closestPage = idx + 1;
          }
        });

        setActiveThumbnail(closestPage);
      }, 80);
    });
  }

  // --- Zoom Controls ---
  function updateZoomDisplay() {
    if (wsZoomLabel) wsZoomLabel.textContent = `${wsState.zoom}%`;
    const pages = wsPagesContainer.querySelectorAll('.pdf-page-container, .text-paper-container');
    const scale = wsState.zoom / 100;
    pages.forEach(p => {
      p.style.transform = `scale(${scale})`;
      p.style.transformOrigin = 'top center';
    });
  }

  if (wsZoomInBtn) {
    wsZoomInBtn.addEventListener('click', () => {
      if (wsState.zoom < 160) {
        wsState.zoom += 15;
        updateZoomDisplay();
      }
    });
  }

  if (wsZoomOutBtn) {
    wsZoomOutBtn.addEventListener('click', () => {
      if (wsState.zoom > 70) {
        wsState.zoom -= 15;
        updateZoomDisplay();
      }
    });
  }

  if (wsFitWidthBtn) {
    wsFitWidthBtn.addEventListener('click', () => {
      wsState.zoom = 100;
      updateZoomDisplay();
    });
  }

  // --- Workspace Segmented Tabs & Input Listeners (Left Panel) ---
  if (wsTabUploadBtn) {
    wsTabUploadBtn.addEventListener('click', () => {
      switchWorkspaceTab('upload');
    });
  }

  if (wsTabPasteBtn) {
    wsTabPasteBtn.addEventListener('click', () => {
      switchWorkspaceTab('paste');
    });
  }

  // Choose PDF button click
  if (wsChoosePdfBtn && wsFileInput) {
    wsChoosePdfBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      wsFileInput.click();
    });
  }

  // Dropzone click to trigger file picker
  if (wsDropzoneBox && wsFileInput) {
    wsDropzoneBox.addEventListener('click', (e) => {
      if (wsChoosePdfBtn && (e.target === wsChoosePdfBtn || wsChoosePdfBtn.contains(e.target))) {
        return;
      }
      wsFileInput.click();
    });
  }

  // Native File input change
  if (wsFileInput) {
    wsFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        registerAndLoadPdf(e.target.files[0]);
        e.target.value = '';
      }
    });
  }

  // Drag-and-drop on the Workspace Dropzone Box
  if (wsDropzoneBox) {
    ['dragenter', 'dragover'].forEach(eventName => {
      wsDropzoneBox.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        wsDropzoneBox.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      wsDropzoneBox.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        wsDropzoneBox.classList.remove('drag-active');
      });
    });

    wsDropzoneBox.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        registerAndLoadPdf(dt.files[0]);
      }
    });
  }

  // Confirm / Use This Text button
  if (wsUseTextBtn && wsPasteTextarea) {
    wsUseTextBtn.addEventListener('click', () => {
      const textVal = wsPasteTextarea.value.trim();
      if (!textVal) {
        wsPasteTextarea.focus();
        return;
      }
      registerAndLoadText(textVal, 'Pasted Source Text');
    });
  }


  // =========================================================================
  // 5. Right Panel: LLM Chat ("Ask Anything")
  // =========================================================================

  // Chat UI helpers shared by live chat and restored conversations
  function resetChatUi() {
    if (wsChatThreadList) {
      wsChatThreadList.innerHTML = '';
      wsChatThreadList.style.display = 'none';
    }
    if (wsChatEmptyState) wsChatEmptyState.style.display = 'flex';
  }

  function appendUserBubble(text) {
    const el = document.createElement('div');
    el.className = 'chat-bubble-user';
    el.textContent = text;
    wsChatThreadList.appendChild(el);
    return el;
  }

  // Real analysis only: the verdict comes from the Quellum detector, the evidence from the passages that were retrieved for this answer.
  function renderAssistantBubble(msg) {
    const el = document.createElement('div');
    el.className = 'chat-bubble-assistant';
    const hr = msg.hallucinationResult || null;
    const sources = msg.sources || [];
    let card = '';

    if (hr || sources.length) {
      let verdict = '';
      if (hr && hr.status === 'ok') {
        const warn = !!hr.flagged;
        const tone = warn ? 'color: #b45309; background: #fef3c7;' : 'color: #059669; background: #d1fae5;';
        const iconColor = warn ? '#b45309' : '#059669';
        const icon = warn ? '<path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>' : '<polyline points="20 6 9 17 4 12"/>';
        const count = (hr.flaggedSentences !== null && hr.flaggedSentences !== undefined)
          ? `${hr.flaggedSentences} of ${hr.totalSentences} sentences flagged`
          : `${hr.totalSentences} sentence${hr.totalSentences === 1 ? '' : 's'} analysed`;
        verdict = `
          <div style="display: flex; align-items: center; justify-content: space-between; font-weight: 700; gap: 8px;">
            <span style="display: flex; align-items: center; gap: 5px;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${iconColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">${icon}</svg>
              <span>${escapeHtml(hr.riskLabel)}</span>
            </span>
            <span style="font-size: 11px; ${tone} padding: 2px 6px; border-radius: 4px;">${escapeHtml(count)}</span>
          </div>
          <div style="margin-top: 4px; font-size: 11.5px; color: #64748b;">Risk score ${Number(hr.risk).toFixed(2)} · a model score, not a probability or proof.</div>`;
        const flaggedSentences = (hr.sentences || []).filter(x => x.label === 'Potentially Hallucinated');
        if (flaggedSentences.length) {
          verdict += `<div style="margin-top: 6px; font-size: 12px; color: #334155;"><strong>Flagged:</strong> ` +
            flaggedSentences.map(x => `<span style="font-style: italic;">"${escapeHtml(x.text)}"</span>`).join(' ') + `</div>`;
        }
      } else if (hr) {
        verdict = `<div style="font-size: 12px; color: #64748b;">Hallucination analysis ${escapeHtml(hr.status)}${hr.reason ? ': ' + escapeHtml(hr.reason) : ''}</div>`;
      }

      let evidence = '';
      if (sources.length) {
        const first = sources[0];
        const where = first.page ? 'Page ' + first.page : 'Source text';
        evidence = `
          <div style="margin-top: 6px; font-size: 12px; color: #334155;">
            <strong>Retrieved evidence:</strong>
            <span style="font-style: italic; color: #475569;">"${escapeHtml(String(first.excerpt || '').slice(0, 220))}${String(first.excerpt || '').length > 220 ? '…' : ''}"</span>
          </div>
          <div class="grounding-citation-chip" data-goto-page="${Number(first.page) || 1}" title="Click to view the passage in the document preview">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
            <span>Jump to ${escapeHtml(where)}${sources.length > 1 ? ' · +' + (sources.length - 1) + ' more passages' : ''}</span>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
          </div>`;
      }
      card = `<div class="grounding-tag-card">${verdict}${evidence}</div>`;
    }

    el.innerHTML = `<div style="margin-bottom: 10px; line-height: 1.6;">${formatMarkdownContent(msg.content)}</div>${card}`;
    const chip = el.querySelector('.grounding-citation-chip');
    if (chip) {
      chip.addEventListener('click', () => scrollToPage(parseInt(chip.getAttribute('data-goto-page'), 10) || 1));
    }
    return el;
  }

  function renderChatThread(messages) {
    resetChatUi();
    if (!messages || !messages.length) return;
    wsChatEmptyState.style.display = 'none';
    wsChatThreadList.style.display = 'flex';
    messages.forEach(m => {
      if (m.role === 'user') appendUserBubble(m.content);
      else wsChatThreadList.appendChild(renderAssistantBubble(m));
    });
    wsChatBody.scrollTop = wsChatBody.scrollHeight;
  }

  let wsSending = false;

  async function sendChatMessage(text) {
    const query = text ? text.trim() : (wsChatTextarea ? wsChatTextarea.value.trim() : '');
    if (!query || wsSending) return;

    if (!wsState.isLoaded || !wsState.documentId) {
      alert('Upload a PDF or paste some text first, then ask your question.');
      return;
    }

    wsSending = true;
    if (wsChatTextarea) {
      wsChatTextarea.value = '';
      wsChatTextarea.style.height = 'auto';
    }
    if (wsChatEmptyState) wsChatEmptyState.style.display = 'none';
    if (wsChatThreadList) wsChatThreadList.style.display = 'flex';

    const userEl = appendUserBubble(query);
    const thinking = document.createElement('div');
    thinking.className = 'chat-bubble-assistant';
    thinking.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; color: #64748b; font-size: 13px;">
        <span class="spinner" style="border-top-color: #7c3aed; width: 14px; height: 14px;"></span>
        <span>Quellum is retrieving evidence and verifying the answer… this can take up to a minute</span>
      </div>`;
    wsChatThreadList.appendChild(thinking);
    wsChatBody.scrollTop = wsChatBody.scrollHeight;

    try {
      if (!wsActiveSessionId) {
        const title = wsState.type === 'pdf'
          ? (wsState.name ? wsState.name.replace(/\.pdf$/i, '') : 'Document Analysis')
          : (query.length > 36 ? query.slice(0, 36) + '...' : query);
        const conv = await window.QuellumAPI.createConversation({ documentId: wsState.documentId, title });
        wsActiveSessionId = conv.id;
      }
      const res = await window.QuellumAPI.sendMessage(wsActiveSessionId, query);
      thinking.remove();
      const answer = res.messages.find(m => m.role === 'assistant');
      wsChatThreadList.appendChild(renderAssistantBubble(answer));
      quellumHistoryStore.applyExchange(wsActiveSessionId, res, { title: wsState.name, sourceType: wsState.type, documentName: wsState.name });
    } catch (err) {
      thinking.remove();
      userEl.remove();                                   // nothing was saved, so do not show the question as if it had been
      if (wsChatTextarea) wsChatTextarea.value = query;  // let the user retry without retyping
      if (err && err.name === 'ApiError' && (err.code === 'expired' || err.code === 'not_found') && wsActiveSessionId) {
        wsActiveSessionId = null;
        showConversationGone(err);
      } else {
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble-assistant';
        bubble.style.color = '#b45309';
        bubble.className = 'chat-bubble-assistant chat-error-bubble';
        bubble.textContent = apiErrorMessage(err) + ' (Your question was put back in the box so you can retry.)';
        wsChatThreadList.querySelectorAll('.chat-error-bubble').forEach(b => b.remove());
        wsChatThreadList.appendChild(bubble);
      }
      if (!wsChatThreadList.children.length) resetChatUi();
    } finally {
      wsSending = false;
      wsChatBody.scrollTop = wsChatBody.scrollHeight;
    }
  }

  // Send button listener
  if (wsChatSendBtn) {
    wsChatSendBtn.addEventListener('click', () => sendChatMessage());
  }

  // Enter key in chat textarea
  if (wsChatTextarea) {
    wsChatTextarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });

    // Auto-resize
    wsChatTextarea.addEventListener('input', () => {
      wsChatTextarea.style.height = 'auto';
      wsChatTextarea.style.height = Math.min(wsChatTextarea.scrollHeight, 120) + 'px';
    });
  }

  // Clear chat button
  if (wsClearChatBtn) {
    wsClearChatBtn.addEventListener('click', () => {
      wsActiveSessionId = null;
      if (wsChatThreadList) {
        wsChatThreadList.innerHTML = '';
        wsChatThreadList.style.display = 'none';
      }
      if (wsChatEmptyState) {
        wsChatEmptyState.style.display = 'flex';
      }
    });
  }


  // =========================================================================
  // 6. Quellum History Store (IndexedDB + localStorage Abstraction)
  // =========================================================================

  // The server is the only source of truth. Older versions kept PDFs in IndexedDB and chats in localStorage: remove those leftovers
  // so no copy of a conversation can outlive the 30-day retention window.
  try {
    localStorage.removeItem('quellum_conversations_v2');
    sessionStorage.removeItem('quellum_workspace_handoff');
    if (window.indexedDB) indexedDB.deleteDatabase('QuellumDocDB');
  } catch (e) { /* storage may be blocked; nothing to clean */ }

  const quellumHistoryStore = {
    list: [],
    loaded: false,
    getAll() { return this.list; },
    get(id) { return this.list.find(c => c.id === id) || null; },
    async refresh() {
      this.list = await window.QuellumAPI.listConversations();
      this.loaded = true;
      return this.list;
    },
    upsert(conv) {
      const i = this.list.findIndex(c => c.id === conv.id);
      if (i >= 0) this.list[i] = conv; else this.list.unshift(conv);
    },
    applyExchange(id, res, fallback) {
      let conv = this.get(id);
      if (!conv) {
        conv = Object.assign({ id, title: 'Conversation', messages: [], createdAt: new Date().toISOString() }, fallback || {});
        this.list.unshift(conv);
      }
      conv.messages = (conv.messages || []).concat(res.messages);
      conv.updatedAt = res.conversation.updatedAt;
      this.list = [conv].concat(this.list.filter(c => c !== conv));
    },
    async update(id, updates) {
      const conv = await window.QuellumAPI.renameConversation(id, updates.title);
      this.upsert(conv);
      return conv;
    },
    async delete(id) {
      await window.QuellumAPI.deleteConversation(id);
      this.list = this.list.filter(c => c.id !== id);
    }
  };

  let historyRefreshSeq = 0;
  async function refreshAndRenderHistory() {
    if (quellumHistoryStore.loaded) renderHistoryView();
    const seq = ++historyRefreshSeq;
    try {
      await quellumHistoryStore.refresh();
    } catch (err) {
      if (!quellumHistoryStore.loaded && historyListScroll) {
        historyListScroll.innerHTML = '<div class="history-search-empty"><p class="history-search-empty-text">' + escapeHtml(apiErrorMessage(err)) + '</p></div>';
      }
      return;
    }
    if (seq === historyRefreshSeq) renderHistoryView();
  }

  // --- Date Grouping & Formatting Helpers ---
  function getConversationGroup(isoString) {
    const d = new Date(isoString);
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const yesterdayStart = todayStart - (24 * 60 * 60 * 1000);
    const sevenDaysAgoStart = todayStart - (7 * 24 * 60 * 60 * 1000);
    const t = d.getTime();

    if (t >= todayStart) return 'Today';
    if (t >= yesterdayStart) return 'Yesterday';
    if (t >= sevenDaysAgoStart) return 'Previous 7 Days';
    return 'Older';
  }

  function formatRelativeItemDate(isoString) {
    const d = new Date(isoString);
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const yesterdayStart = todayStart - (24 * 60 * 60 * 1000);
    const t = d.getTime();

    if (t >= todayStart || t >= yesterdayStart) {
      return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    }
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  function formatDetailDate(isoString) {
    const d = new Date(isoString);
    return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function formatDetailTime(isoString) {
    const d = new Date(isoString);
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  // --- Markdown Formatting for Quellum Responses ---
  function formatMarkdownContent(text) {
    if (!text) return '';

    // Escape entities for safety
    let safe = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Code blocks: ```...```
    safe = safe.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre class="history-code-block"><code>${code.trim()}</code></pre>`;
    });

    // Inline code: `...`
    safe = safe.replace(/`([^`]+)`/g, '<code style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 12.5px; color: #0f172a;">$1</code>');

    // Bold: **...**
    safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic: *...*
    safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Process lists and paragraphs
    const lines = safe.split('\n');
    let result = [];
    let inOl = false;
    let inUl = false;

    lines.forEach(line => {
      const trimmed = line.trim();
      const olMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
      const ulMatch = trimmed.match(/^[\u2022\-\*]\s+(.*)$/);

      if (olMatch) {
        if (inUl) { result.push('</ul>'); inUl = false; }
        if (!inOl) { result.push('<ol class="history-ordered-list">'); inOl = true; }
        result.push(`<li>${olMatch[2]}</li>`);
      } else if (ulMatch) {
        if (inOl) { result.push('</ol>'); inOl = false; }
        if (!inUl) { result.push('<ul class="history-bullet-list">'); inUl = true; }
        result.push(`<li>${ulMatch[1]}</li>`);
      } else {
        if (inOl) { result.push('</ol>'); inOl = false; }
        if (inUl) { result.push('</ul>'); inUl = false; }
        if (trimmed.length > 0) {
          if (trimmed.startsWith('<pre') || trimmed.endsWith('</pre>')) {
            result.push(line);
          } else {
            result.push(`<p>${line}</p>`);
          }
        }
      }
    });

    if (inOl) result.push('</ol>');
    if (inUl) result.push('</ul>');

    return result.join('');
  }

  // =========================================================================
  // 7. Restore Conversation in Workspace
  // =========================================================================

  // Expired / unavailable conversation: friendly message + a way to start over. Nothing from the conversation is shown.
  function showConversationGone(err) {
    const text = document.getElementById('expiredModalText');
    if (text) {
      text.textContent = (err && err.code === 'expired')
        ? 'This conversation has expired after 30 days of inactivity.'
        : 'This conversation is no longer available. It may have been deleted or may have expired after 30 days of inactivity.';
    }
    quellumHistoryStore.refresh().then(() => { if (historyView && historyView.style.display !== 'none') renderHistoryView(); }).catch(() => {});
    openModal(document.getElementById('expiredModal'));
  }

  async function restoreSessionInWorkspace(conversationId) {
    let session, pdfFile = null;
    try {
      session = await window.QuellumAPI.getConversation(conversationId);      // ownership + expiry are enforced by the server
      if (session.sourceType === 'pdf') {
        const buf = await window.QuellumAPI.getDocumentFile(session.documentId);
        pdfFile = new File([buf], session.documentName, { type: 'application/pdf' });
      }
    } catch (err) {
      if (err && err.name === 'ApiError' && (err.code === 'expired' || err.code === 'not_found')) showConversationGone(err);
      else alert(apiErrorMessage(err));
      return;
    }

    resetChatUi();
    wsActiveSessionId = session.id;
    wsState.documentId = session.documentId;
    rememberActiveDocument(session.documentId);
    switchView('workspace');
    window.location.hash = '#workspace';

    if (pdfFile) await loadPdfIntoWorkspace(pdfFile);
    else loadTextIntoWorkspace(session.textContent || '', session.documentName || 'Text Source');

    wsActiveSessionId = session.id;                       // loading a document must not detach the restored conversation
    renderChatThread(session.messages);
    quellumHistoryStore.upsert(session);
  }


  // =========================================================================
  // 8. History Page Controller & UI View
  // =========================================================================

  const historySearchInput = document.getElementById('historySearchInput');
  const historySearchClearBtn = document.getElementById('historySearchClearBtn');
  const historyNewChatBtn = document.getElementById('historyNewChatBtn');
  const historyListScroll = document.getElementById('historyListScroll');

  const historyDetailContent = document.getElementById('historyDetailContent');
  const historyDetailEmptyState = document.getElementById('historyDetailEmptyState');
  const historyDetailBadge = document.getElementById('historyDetailBadge');
  const historyDetailTitle = document.getElementById('historyDetailTitle');
  const historyDetailDocName = document.getElementById('historyDetailDocName');
  const historyDetailDateText = document.getElementById('historyDetailDateText');
  const historyDetailTimeText = document.getElementById('historyDetailTimeText');
  const historyOpenInWorkspaceBtn = document.getElementById('historyOpenInWorkspaceBtn');
  const historyHeaderMoreBtn = document.getElementById('historyHeaderMoreBtn');
  const historyHeaderMenu = document.getElementById('historyHeaderMenu');
  const historyHeaderRenameBtn = document.getElementById('historyHeaderRenameBtn');
  const historyHeaderDeleteBtn = document.getElementById('historyHeaderDeleteBtn');
  const historyMessagesContainer = document.getElementById('historyMessagesContainer');
  const historyMobileBackBtn = document.getElementById('historyMobileBackBtn');

  // Modals
  const closeDeleteModalBtn = document.getElementById('closeDeleteModalBtn');
  const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
  const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
  const deleteModalConvName = document.getElementById('deleteModalConvName');

  const closeRenameModalBtn = document.getElementById('closeRenameModalBtn');
  const cancelRenameBtn = document.getElementById('cancelRenameBtn');
  const confirmRenameBtn = document.getElementById('confirmRenameBtn');
  const renameTitleInput = document.getElementById('renameTitleInput');

  let selectedConversationId = null;
  let pendingActionConvId = null;
  let activeOpenPopover = null;

  function closeAllPopovers() {
    if (activeOpenPopover) {
      activeOpenPopover.style.display = 'none';
      activeOpenPopover = null;
    }
    if (historyHeaderMenu) {
      historyHeaderMenu.style.display = 'none';
    }
  }

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.history-dropdown-wrap') && !e.target.closest('.history-item-more-btn')) {
      closeAllPopovers();
    }
  });

  // Render detail for a specific conversation
  function renderHistoryDetail(conv) {
    if (!conv) {
      if (historyDetailContent) historyDetailContent.style.display = 'none';
      if (historyDetailEmptyState) historyDetailEmptyState.style.display = 'flex';
      return;
    }

    if (historyDetailContent) historyDetailContent.style.display = 'flex';
    if (historyDetailEmptyState) historyDetailEmptyState.style.display = 'none';

    // Header badge
    if (historyDetailBadge) {
      if (conv.sourceType === 'pdf') {
        historyDetailBadge.className = 'history-detail-badge pdf';
        historyDetailBadge.innerHTML = `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="9" y1="15" x2="15" y2="15"/>
          </svg>
        `;
      } else {
        historyDetailBadge.className = 'history-detail-badge text';
        historyDetailBadge.innerHTML = `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
        `;
      }
    }

    // Title and Meta
    if (historyDetailTitle) historyDetailTitle.textContent = conv.title;
    if (historyDetailDocName) historyDetailDocName.textContent = conv.documentName;
    if (historyDetailDateText) historyDetailDateText.textContent = formatDetailDate(conv.updatedAt || conv.createdAt);
    if (historyDetailTimeText) historyDetailTimeText.textContent = formatDetailTime(conv.updatedAt || conv.createdAt);

    // Messages Container
    if (historyMessagesContainer) {
      historyMessagesContainer.innerHTML = '';

      if (conv.messages && conv.messages.length > 0) {
        conv.messages.forEach(msg => {
          if (msg.role === 'user') {
            const userWrap = document.createElement('div');
            userWrap.className = 'history-user-msg-wrap';
            userWrap.innerHTML = `
              <div class="history-user-bubble-row">
                <div class="history-user-bubble">${escapeHtml(String(msg.content || ''))}</div>
                <div class="history-user-avatar">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                  </svg>
                </div>
              </div>
              <span class="history-user-time">${formatDetailTime(msg.createdAt)}</span>
            `;
            historyMessagesContainer.appendChild(userWrap);
          } else {
            const asstWrap = document.createElement('div');
            asstWrap.className = 'history-assistant-msg-wrap';

            // Evidence section
            let evidenceHtml = '';
            if (msg.sources && msg.sources.length > 0) {
              const evidenceCards = msg.sources.map(src => {
                const isPage = typeof src.page === 'number' || (src.page && src.page !== 'null');
                const badgeLabel = isPage ? `Page ${src.page}` : 'Text Source';
                const badgeIcon = isPage
                  ? `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>`
                  : `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>`;

                return `
                  <div class="history-evidence-card">
                    <span class="history-page-badge">
                      ${badgeIcon}
                      <span>${badgeLabel}</span>
                    </span>
                    <span class="history-evidence-text">${escapeHtml(String(src.excerpt || ''))}</span>
                  </div>
                `;
              }).join('');

              let groundingBadgeHtml = '';
              if (msg.hallucinationResult && msg.hallucinationResult.status === 'ok') {
                const warn = !!msg.hallucinationResult.flagged;
                groundingBadgeHtml = `
                  <span class="history-grounding-badge" style="${warn ? 'background: #fef3c7; color: #b45309; border-color: #fde68a;' : ''}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="${warn ? '#b45309' : '#059669'}" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                    <span>${escapeHtml(msg.hallucinationResult.riskLabel || '')}</span>
                  </span>
                `;
              }

              evidenceHtml = `
                <div class="history-evidence-section">
                  <div class="history-evidence-header">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
                    <span>Evidence from source</span>
                    ${groundingBadgeHtml}
                  </div>
                  ${evidenceCards}
                </div>
              `;
            }

            const formattedHtml = formatMarkdownContent(msg.content);

            asstWrap.innerHTML = `
              <div class="history-assistant-row">
                <div class="history-q-avatar">
                  <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="46" cy="46" r="30" stroke="url(#qPinkOrangeYellow)" stroke-width="14" stroke-linecap="round" />
                    <path d="M58 58 L78 78" stroke="url(#qTailGrad)" stroke-width="14" stroke-linecap="round" />
                  </svg>
                </div>
                <div class="history-assistant-card">
                  ${formattedHtml}
                  ${evidenceHtml}
                </div>
              </div>
              <span class="history-assistant-time">${formatDetailTime(msg.createdAt)}</span>
            `;
            historyMessagesContainer.appendChild(asstWrap);
          }
        });
      }
    }
  }

  // Render Left Panel Grouped List and wire interactions
  function renderHistoryView() {
    closeAllPopovers();

    const allConversations = quellumHistoryStore.getAll();
    const query = (historySearchInput ? historySearchInput.value : '').trim().toLowerCase();

    // Toggle Search Clear Button
    if (historySearchClearBtn) {
      historySearchClearBtn.style.display = query.length > 0 ? 'flex' : 'none';
    }

    if (!historyListScroll) return;
    historyListScroll.innerHTML = '';

    // 1. Global Empty State (0 conversations in store)
    if (allConversations.length === 0) {
      historyListScroll.innerHTML = `
        <div class="history-global-empty">
          <div class="history-empty-icon-box">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <h3 class="history-empty-title">No conversations yet</h3>
          <p class="history-empty-subtitle">Start exploring your documents in Workspace. Your conversations will appear here.</p>
          <button class="history-empty-cta-btn" id="historyGlobalEmptyNewChatBtn" type="button">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            <span>Start New Chat</span>
          </button>
        </div>
      `;

      const globalEmptyBtn = document.getElementById('historyGlobalEmptyNewChatBtn');
      if (globalEmptyBtn) {
        globalEmptyBtn.addEventListener('click', () => {
          wsActiveSessionId = null;
          if (wsChatThreadList) {
            wsChatThreadList.innerHTML = '';
            wsChatThreadList.style.display = 'none';
          }
          if (wsChatEmptyState) wsChatEmptyState.style.display = 'flex';
          switchView('workspace');
          window.location.hash = '#workspace';
        });
      }

      renderHistoryDetail(null);
      return;
    }

    // 2. Filter conversations based on search
    let filtered = allConversations;
    if (query) {
      filtered = allConversations.filter(c => {
        const titleMatch = (c.title || '').toLowerCase().includes(query);
        const docMatch = (c.documentName || '').toLowerCase().includes(query);
        const msgMatch = (c.messages || []).some(m => (m.content || '').toLowerCase().includes(query));
        return titleMatch || docMatch || msgMatch;
      });
    }

    // 3. Search Empty State
    if (filtered.length === 0) {
      historyListScroll.innerHTML = `
        <div class="history-search-empty">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <p class="history-search-empty-text">No matching conversations</p>
          <button class="history-search-reset-btn" id="historyResetSearchBtn" type="button">Clear search</button>
        </div>
      `;

      const resetBtn = document.getElementById('historyResetSearchBtn');
      if (resetBtn) {
        resetBtn.addEventListener('click', () => {
          if (historySearchInput) historySearchInput.value = '';
          renderHistoryView();
        });
      }

      renderHistoryDetail(null);
      return;
    }

    // 4. Determine Active Conversation
    if (!selectedConversationId || !filtered.some(c => c.id === selectedConversationId)) {
      selectedConversationId = filtered[0].id;
    }

    // 5. Group conversations into: Today, Yesterday, Previous 7 Days, Older
    const groups = {
      'Today': [],
      'Yesterday': [],
      'Previous 7 Days': [],
      'Older': []
    };

    filtered.forEach(conv => {
      const g = getConversationGroup(conv.updatedAt || conv.createdAt);
      if (groups[g]) {
        groups[g].push(conv);
      } else {
        groups['Older'].push(conv);
      }
    });

    // 6. Render each group
    ['Today', 'Yesterday', 'Previous 7 Days', 'Older'].forEach(groupName => {
      const items = groups[groupName];
      if (items && items.length > 0) {
        const groupContainer = document.createElement('div');
        groupContainer.className = 'history-date-group';

        const headerEl = document.createElement('div');
        headerEl.className = 'history-date-header';
        headerEl.textContent = groupName;
        groupContainer.appendChild(headerEl);

        items.forEach(item => {
          const itemEl = document.createElement('div');
          const isSelected = item.id === selectedConversationId;
          itemEl.className = `history-item ${isSelected ? 'active' : ''}`;
          itemEl.setAttribute('data-conv-id', item.id);

          // Get preview of latest question
          let previewText = '';
          if (item.messages && item.messages.length > 0) {
            const userMsgs = item.messages.filter(m => m.role === 'user');
            if (userMsgs.length > 0) {
              previewText = userMsgs[userMsgs.length - 1].content;
            } else {
              previewText = item.messages[0].content;
            }
          }

          // Badge SVG
          const badgeClass = item.sourceType === 'pdf' ? 'pdf' : 'text';
          const badgeSvg = item.sourceType === 'pdf'
            ? `<div class="history-item-badge pdf">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="9" y1="15" x2="15" y2="15"/>
                </svg>
              </div>`
            : `<div class="history-item-badge text">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/>
                  <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
              </div>`;

          itemEl.innerHTML = `
            ${badgeSvg}
            <div class="history-item-content">
              <span class="history-item-title">${escapeHtml(String(item.title || ''))}</span>
              <span class="history-item-doc">${escapeHtml(String(item.documentName || (item.sourceType === 'pdf' ? 'PDF document' : 'Text Source')))}</span>
              <span class="history-item-preview">${escapeHtml(String(previewText || ''))}</span>
            </div>
            <div class="history-item-right">
              <span class="history-item-time">${formatRelativeItemDate(item.updatedAt || item.createdAt)}</span>
              <div class="history-dropdown-wrap">
                <button class="history-item-more-btn" title="Options" type="button">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
                </button>
                <div class="history-popover-menu" style="display: none;">
                  <button class="history-popover-item item-open-ws-btn" type="button">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    <span>Open in Workspace</span>
                  </button>
                  <button class="history-popover-item item-rename-btn" type="button">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>
                    <span>Rename</span>
                  </button>
                  <button class="history-popover-item danger item-delete-btn" type="button">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    <span>Delete</span>
                  </button>
                </div>
              </div>
            </div>
          `;

          // Card click to select conversation
          itemEl.addEventListener('click', (e) => {
            if (e.target.closest('.history-dropdown-wrap')) return;
            selectedConversationId = item.id;
            document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
            itemEl.classList.add('active');
            renderHistoryDetail(item);
          });

          // Three-dot options menu
          const moreBtn = itemEl.querySelector('.history-item-more-btn');
          const popover = itemEl.querySelector('.history-popover-menu');

          if (moreBtn && popover) {
            moreBtn.addEventListener('click', (e) => {
              e.stopPropagation();
              const isCurrentlyOpen = popover.style.display === 'flex';
              closeAllPopovers();
              if (!isCurrentlyOpen) {
                popover.style.display = 'flex';
                activeOpenPopover = popover;
              }
            });

            // Action: Open in Workspace
            const openWsBtn = itemEl.querySelector('.item-open-ws-btn');
            if (openWsBtn) {
              openWsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                closeAllPopovers();
                restoreSessionInWorkspace(item.id);
              });
            }

            // Action: Rename
            const renameBtn = itemEl.querySelector('.item-rename-btn');
            if (renameBtn) {
              renameBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                closeAllPopovers();
                pendingActionConvId = item.id;
                if (renameTitleInput) renameTitleInput.value = item.title;
                openModal(historyRenameModal);
                setTimeout(() => { if (renameTitleInput) renameTitleInput.focus(); }, 100);
              });
            }

            // Action: Delete
            const deleteBtn = itemEl.querySelector('.item-delete-btn');
            if (deleteBtn) {
              deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                closeAllPopovers();
                pendingActionConvId = item.id;
                if (deleteModalConvName) deleteModalConvName.textContent = `"${item.title}"`;
                openModal(historyDeleteModal);
              });
            }
          }

          groupContainer.appendChild(itemEl);
        });

        historyListScroll.appendChild(groupContainer);
      }
    });

    // 7. Render Selected Conversation in Right Panel
    const activeConv = filtered.find(c => c.id === selectedConversationId) || filtered[0];
    renderHistoryDetail(activeConv);
  }

  // --- Right Panel Header Actions ---
  if (historyOpenInWorkspaceBtn) {
    historyOpenInWorkspaceBtn.addEventListener('click', () => {
      if (selectedConversationId) {
        restoreSessionInWorkspace(selectedConversationId);
      }
    });
  }

  if (historyHeaderMoreBtn && historyHeaderMenu) {
    historyHeaderMoreBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = historyHeaderMenu.style.display === 'flex';
      closeAllPopovers();
      if (!isOpen) {
        historyHeaderMenu.style.display = 'flex';
        activeOpenPopover = historyHeaderMenu;
      }
    });
  }

  if (historyHeaderRenameBtn) {
    historyHeaderRenameBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      closeAllPopovers();
      if (!selectedConversationId) return;
      const conv = quellumHistoryStore.get(selectedConversationId);
      if (conv) {
        pendingActionConvId = conv.id;
        if (renameTitleInput) renameTitleInput.value = conv.title;
        openModal(historyRenameModal);
        setTimeout(() => { if (renameTitleInput) renameTitleInput.focus(); }, 100);
      }
    });
  }

  if (historyHeaderDeleteBtn) {
    historyHeaderDeleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      closeAllPopovers();
      if (!selectedConversationId) return;
      const conv = quellumHistoryStore.get(selectedConversationId);
      if (conv) {
        pendingActionConvId = conv.id;
        if (deleteModalConvName) deleteModalConvName.textContent = `"${conv.title}"`;
        openModal(historyDeleteModal);
      }
    });
  }

  // --- + New Chat Button ---
  if (historyNewChatBtn) {
    historyNewChatBtn.addEventListener('click', () => {
      wsActiveSessionId = null;
      if (wsChatThreadList) {
        wsChatThreadList.innerHTML = '';
        wsChatThreadList.style.display = 'none';
      }
      if (wsChatEmptyState) wsChatEmptyState.style.display = 'flex';
      switchView('workspace');
      window.location.hash = '#workspace';
      setTimeout(() => {
        if (wsChatTextarea) wsChatTextarea.focus();
      }, 150);
    });
  }

  // --- Search Input & Clear ---
  if (historySearchInput) {
    historySearchInput.addEventListener('input', () => {
      renderHistoryView();
    });
  }

  if (historySearchClearBtn) {
    historySearchClearBtn.addEventListener('click', () => {
      if (historySearchInput) {
        historySearchInput.value = '';
        historySearchInput.focus();
      }
      renderHistoryView();
    });
  }

  // --- Rename Modal Actions ---
  if (confirmRenameBtn) {
    confirmRenameBtn.addEventListener('click', async () => {
      if (!pendingActionConvId || !renameTitleInput) return;
      const newTitle = renameTitleInput.value.trim();
      if (!newTitle) {
        renameTitleInput.focus();
        return;
      }
      try {
        await quellumHistoryStore.update(pendingActionConvId, { title: newTitle });
      } catch (err) {
        alert(apiErrorMessage(err));
      }
      closeModal(historyRenameModal);
      renderHistoryView();
    });
  }

  if (renameTitleInput) {
    renameTitleInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        confirmRenameBtn.click();
      }
    });
  }

  [closeRenameModalBtn, cancelRenameBtn].forEach(btn => {
    if (btn) btn.addEventListener('click', () => closeModal(historyRenameModal));
  });

  // --- Delete Modal Actions ---
  if (confirmDeleteBtn) {
    confirmDeleteBtn.addEventListener('click', async () => {
      if (!pendingActionConvId) return;
      const id = pendingActionConvId;
      try {
        await quellumHistoryStore.delete(id);
        if (wsActiveSessionId === id) resetWorkspaceDocument();     // its document may be gone with it: do not leave a stale source open
      } catch (err) {
        if (!(err && err.code === 'not_found' || err && err.code === 'expired')) alert(apiErrorMessage(err));
        else quellumHistoryStore.list = quellumHistoryStore.list.filter(c => c.id !== id);
      }
      if (selectedConversationId === id) {
        selectedConversationId = null;
      }
      pendingActionConvId = null;
      closeModal(historyDeleteModal);
      renderHistoryView();
    });
  }

  [closeDeleteModalBtn, cancelDeleteBtn].forEach(btn => {
    if (btn) btn.addEventListener('click', () => closeModal(historyDeleteModal));
  });

  // Mobile back button
  if (historyMobileBackBtn) {
    historyMobileBackBtn.addEventListener('click', () => {
      const leftPanel = document.getElementById('historyLeftPanel');
      const rightPanel = document.getElementById('historyRightPanel');
      if (leftPanel && rightPanel) {
        leftPanel.style.display = 'flex';
        rightPanel.style.display = 'none';
      }
    });
  }


  // =========================================================================
  // 9. Expired-conversation modal, anonymous session bootstrap, initial route
  // =========================================================================
  const expiredModal = document.getElementById('expiredModal');
  const expiredNewChatBtn = document.getElementById('expiredNewChatBtn');
  const expiredCloseBtn = document.getElementById('expiredCloseBtn');
  if (expiredNewChatBtn) {
    expiredNewChatBtn.addEventListener('click', () => {
      closeModal(expiredModal);
      resetWorkspaceDocument();
      switchView('workspace');
      window.location.hash = '#workspace';
    });
  }
  if (expiredCloseBtn) expiredCloseBtn.addEventListener('click', () => closeModal(expiredModal));

  // Establishes the HttpOnly session cookie on first visit (the browser never sees the credential) and warms the history cache.
  if (window.QuellumAPI) {
    sessionReady = window.QuellumAPI.session();
    sessionReady.then(() => quellumHistoryStore.refresh()).catch(() => { /* surfaced when the user acts */ });
  }

  handleRoute();

});
