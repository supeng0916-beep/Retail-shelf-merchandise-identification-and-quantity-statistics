'use strict';

const API_BASE = '/api/v1';
const MAX_DETECT_FILE_SIZE = 10 * 1024 * 1024;
const MIN_SKU_FILES = 3;
const MAX_SKU_FILES = 8;
const HEALTH_POLL_MS = 30_000;
const JOB_POLL_MS = 1_500;

const state = {
  detectFile: null,
  detectPreviewUrl: '',
  detectConfidence: 0.25,
  skuItems: [],
  ingestJobId: '',
  publishPollTimer: null,
  lastPreview: null,
  lastRemovedItemId: '',
  activeOnlyView: false,
};

document.addEventListener('DOMContentLoaded', () => {
  bindNavigation();
  bindRecognitionView();
  bindSkuView();
  updateConfidenceLabel(state.detectConfidence);
  resetRecognitionResult();
  resetSkuBatch({ keepForm: true });
  checkHealth();
  loadSkuLibrary();
  window.setInterval(checkHealth, HEALTH_POLL_MS);
});

function bindNavigation() {
  const navLinks = Array.from(document.querySelectorAll('.nav-link'));
  navLinks.forEach((button) => {
    button.addEventListener('click', () => switchView(button.dataset.view || 'recognitionView'));
  });
}

function switchView(viewId) {
  document.querySelectorAll('.view').forEach((view) => {
    view.classList.toggle('active', view.id === viewId);
  });
  document.querySelectorAll('.nav-link').forEach((button) => {
    button.classList.toggle('active', button.dataset.view === viewId);
  });
}

function bindRecognitionView() {
  const uploadZone = document.getElementById('detectUploadZone');
  const chooseBtn = document.getElementById('detectChooseBtn');
  const fileInput = document.getElementById('detectFileInput');
  const detectBtn = document.getElementById('detectBtn');
  const confSlider = document.getElementById('confSlider');

  chooseBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', (event) => {
    const [file] = Array.from(event.target.files || []);
    if (file) {
      setDetectFile(file);
    }
    fileInput.value = '';
  });

  confSlider.addEventListener('input', (event) => {
    state.detectConfidence = Number(event.target.value || 0.25);
    updateConfidenceLabel(state.detectConfidence);
  });

  detectBtn.addEventListener('click', runRecognition);

  bindDropZone(uploadZone, {
    multiple: false,
    onFiles: (files) => {
      if (files[0]) {
        setDetectFile(files[0]);
      }
    },
  });
}

function bindSkuView() {
  const uploadZone = document.getElementById('skuUploadZone');
  const chooseBtn = document.getElementById('skuChooseBtn');
  const analyzeBtn = document.getElementById('skuAnalyzeBtn');
  const publishBtn = document.getElementById('skuPublishBtn');
  const resetBtn = document.getElementById('skuResetBtn');
  const gotoRecognitionBtn = document.getElementById('gotoRecognitionBtn');
  const undoRemoveBtn = document.getElementById('undoRemoveBtn');
  const toggleActiveOnlyBtn = document.getElementById('toggleActiveOnlyBtn');
  const fileInput = document.getElementById('skuFileInput');
  const fileList = document.getElementById('skuFileList');

  chooseBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length) {
      appendSkuFiles(files);
    }
    fileInput.value = '';
  });

  analyzeBtn.addEventListener('click', runIngestPreview);
  publishBtn.addEventListener('click', publishSkuBatch);
  resetBtn.addEventListener('click', () => resetSkuBatch({ keepForm: false }));
  gotoRecognitionBtn.addEventListener('click', () => switchView('recognitionView'));
  undoRemoveBtn.addEventListener('click', restoreLastRemovedSkuItem);
  toggleActiveOnlyBtn.addEventListener('click', toggleActiveOnlyView);

  bindDropZone(uploadZone, {
    multiple: true,
    onFiles: appendSkuFiles,
  });

  fileList.addEventListener('click', (event) => {
    const removeButton = event.target.closest('[data-remove-id]');
    if (!removeButton) {
      return;
    }
    removeSkuItem(removeButton.dataset.removeId || '');
  });
}

function bindDropZone(element, options) {
  const { multiple, onFiles } = options;

  element.addEventListener('dragover', (event) => {
    event.preventDefault();
    element.classList.add('drag-over');
  });

  element.addEventListener('dragleave', () => {
    element.classList.remove('drag-over');
  });

  element.addEventListener('drop', (event) => {
    event.preventDefault();
    element.classList.remove('drag-over');
    const files = Array.from(event.dataTransfer?.files || []);
    onFiles(multiple ? files : files.slice(0, 1));
  });

  element.addEventListener('click', (event) => {
    if (event.target.closest('button')) {
      return;
    }
    const inputId = multiple ? 'skuFileInput' : 'detectFileInput';
    document.getElementById(inputId).click();
  });
}

function validateImageFile(file, maxSize = MAX_DETECT_FILE_SIZE) {
  if (!file.type.startsWith('image/')) {
    return '请上传图片文件，例如 JPEG、PNG、WEBP。';
  }
  if (file.size > maxSize) {
    return '图片超过 10MB，请压缩后再上传。';
  }
  return '';
}

function setDetectFile(file) {
  const error = validateImageFile(file);
  if (error) {
    window.alert(error);
    return;
  }

  state.detectFile = file;
  if (state.detectPreviewUrl) {
    URL.revokeObjectURL(state.detectPreviewUrl);
  }
  state.detectPreviewUrl = URL.createObjectURL(file);

  document.getElementById('detectUploadTitle').textContent = file.name;
  document.getElementById('detectUploadSubtitle').textContent = `${formatBytes(file.size)}，已准备识别`;
  document.getElementById('originalImg').src = state.detectPreviewUrl;
  document.getElementById('annotatedImg').removeAttribute('src');
  document.getElementById('detectBtn').disabled = false;
  resetRecognitionResult();
}

function updateConfidenceLabel(value) {
  document.getElementById('confValue').textContent = Number(value).toFixed(2);
}

async function runRecognition() {
  if (!state.detectFile) {
    window.alert('请先选择一张货架图片。');
    return;
  }

  const formData = new FormData();
  formData.append('file', state.detectFile);

  const url = `${API_BASE}/detect?conf=${encodeURIComponent(state.detectConfidence)}&return_image=true`;
  setLoading(true, '正在执行货架识别...');
  document.getElementById('detectBtn').disabled = true;

  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    const data = await parseJson(response);
    if (!response.ok) {
      throw new Error(data.detail || `识别失败，HTTP ${response.status}`);
    }
    renderRecognitionResult(data);
  } catch (error) {
    window.alert(error.message || '识别失败。');
  } finally {
    setLoading(false);
    document.getElementById('detectBtn').disabled = !state.detectFile;
  }
}

function resetRecognitionResult() {
  document.getElementById('resultPanel').classList.add('hidden');
  document.getElementById('totalCount').textContent = '--';
  document.getElementById('inferenceTime').textContent = '--';
  document.getElementById('embeddingTime').textContent = '--';
  document.getElementById('totalTime').textContent = '--';
  document.getElementById('skuTable').innerHTML = '';
}

function renderRecognitionResult(data) {
  document.getElementById('resultPanel').classList.remove('hidden');
  document.getElementById('totalCount').textContent = String(data.total_count ?? 0);
  document.getElementById('inferenceTime').textContent = formatDuration(data.inference_time_ms);
  document.getElementById('embeddingTime').textContent = formatDuration(data.embedding_time_ms);
  document.getElementById('totalTime').textContent = formatDuration(data.total_time_ms);

  if (data.annotated_image) {
    document.getElementById('annotatedImg').src = data.annotated_image;
  }

  const boxMap = new Map();
  for (const box of data.boxes || []) {
    const skuId = box.sku_id || 'unknown';
    const current = boxMap.get(skuId) || [];
    current.push(box);
    boxMap.set(skuId, current);
  }

  const rows = Object.entries(data.by_sku || {}).sort((left, right) => right[1] - left[1]);
  const tableWrap = document.getElementById('skuTable');

  if (!rows.length) {
    tableWrap.innerHTML = '<p class="progress-caption">当前图片中没有识别到商品。</p>';
    return;
  }

  const htmlRows = rows.map(([skuId, count]) => {
    const boxes = boxMap.get(skuId) || [];
    const firstNamedBox = boxes.find((box) => box.sku_name && box.sku_name !== 'unknown');
    const productName = skuId === 'unknown'
      ? '未知商品'
      : escapeHtml(firstNamedBox?.sku_name || skuId);
    const bestScore = boxes.length
      ? Math.max(...boxes.map((box) => Number(box.match_score || 0)))
      : 0;

    return `
      <tr>
        <td>${productName}</td>
        <td>${escapeHtml(skuId)}</td>
        <td>${count}</td>
        <td>${bestScore ? bestScore.toFixed(3) : '--'}</td>
      </tr>
    `;
  }).join('');

  tableWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>商品名</th>
          <th>SKU ID</th>
          <th>数量</th>
          <th>最高分</th>
        </tr>
      </thead>
      <tbody>${htmlRows}</tbody>
    </table>
  `;

  document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function appendSkuFiles(files) {
  const nextItems = [];

  for (const file of files) {
    const error = validateImageFile(file);
    if (error) {
      window.alert(`${file.name}：${error}`);
      continue;
    }

    const item = createSkuItem(file);
    nextItems.push(item);
    fillSkuImageDimensions(item);
  }

  if (!nextItems.length) {
    return;
  }

  state.skuItems = state.skuItems.filter((item) => !item.removed).concat(nextItems);
  invalidatePreviewState('已添加新图片，请重新执行预分析。');
  renderSkuFileList();
  refreshSkuControls();
}

function createSkuItem(file) {
  return {
    id: `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
    file,
    url: URL.createObjectURL(file),
    removed: false,
    status: 'pending',
    label: '待分析',
    width: 0,
    height: 0,
  };
}

function fillSkuImageDimensions(item) {
  const image = new Image();
  image.onload = () => {
    item.width = image.naturalWidth;
    item.height = image.naturalHeight;
    renderSkuFileList();
  };
  image.src = item.url;
}

function removeSkuItem(itemId) {
  const item = state.skuItems.find((entry) => entry.id === itemId);
  if (!item || item.removed) {
    return;
  }

  item.removed = true;
  item.status = 'removed';
  item.label = '已删除';
  state.lastRemovedItemId = item.id;
  invalidatePreviewState('你已删除图片，请重新执行预分析。');
  renderSkuFileList();
  refreshSkuControls();
}

function restoreLastRemovedSkuItem() {
  if (!state.lastRemovedItemId) {
    return;
  }

  const item = state.skuItems.find((entry) => entry.id === state.lastRemovedItemId);
  if (!item || !item.removed) {
    state.lastRemovedItemId = '';
    refreshSkuControls();
    return;
  }

  item.removed = false;
  item.status = 'pending';
  item.label = '待分析';
  state.lastRemovedItemId = '';
  invalidatePreviewState('已撤销最近删除，请重新执行预分析。');
  renderSkuFileList();
  refreshSkuControls();
}

function toggleActiveOnlyView() {
  state.activeOnlyView = !state.activeOnlyView;
  renderSkuFileList();
  refreshSkuControls();
}

function invalidatePreviewState(message) {
  state.ingestJobId = '';
  state.lastPreview = null;
  stopPublishPolling();
  document.getElementById('skuPublishBtn').disabled = true;
  document.getElementById('gotoRecognitionBtn').classList.add('hidden');
  setProgress('publish', 0, '尚未开始发布。');
  setOcrSignal(message || '请先执行预分析。', 'info');

  state.skuItems.forEach((item) => {
    if (!item.removed) {
      item.status = 'pending';
      item.label = '待分析';
    }
  });
}

function getActiveSkuItems() {
  return state.skuItems.filter((item) => !item.removed);
}

function refreshSkuControls() {
  const activeItems = getActiveSkuItems();
  const count = activeItems.length;
  const analyzeBtn = document.getElementById('skuAnalyzeBtn');
  const publishBtn = document.getElementById('skuPublishBtn');
  const undoRemoveBtn = document.getElementById('undoRemoveBtn');
  const toggleActiveOnlyBtn = document.getElementById('toggleActiveOnlyBtn');
  const toolbarInfo = document.getElementById('skuToolbarInfo');
  const removedCount = state.skuItems.filter((item) => item.removed).length;
  const visibleCount = getVisibleSkuItems().length;

  document.getElementById('skuUploadTitle').textContent = count
    ? `当前已选择 ${count} 张商品图片`
    : '拖拽商品照片到这里';

  if (!count) {
    document.getElementById('skuUploadSubtitle').textContent = '请上传同一个商品的 3-8 张不同角度照片';
  } else if (count < MIN_SKU_FILES) {
    document.getElementById('skuUploadSubtitle').textContent = `还差 ${MIN_SKU_FILES - count} 张，至少需要 ${MIN_SKU_FILES} 张图片`;
  } else if (count > MAX_SKU_FILES) {
    document.getElementById('skuUploadSubtitle').textContent = `当前超过上限 ${MAX_SKU_FILES} 张，请删除多余图片后再预分析`;
  } else {
    document.getElementById('skuUploadSubtitle').textContent = '数量符合要求，可开始预分析';
  }

  analyzeBtn.disabled = count < MIN_SKU_FILES || count > MAX_SKU_FILES;
  publishBtn.disabled = !Boolean(state.lastPreview && state.lastPreview.is_publishable);
  undoRemoveBtn.disabled = !Boolean(state.lastRemovedItemId);
  toggleActiveOnlyBtn.textContent = state.activeOnlyView ? '显示全部图片' : '只看有效图片';
  toolbarInfo.textContent = state.activeOnlyView
    ? `当前仅显示有效图片：${visibleCount} 张，已隐藏 ${removedCount} 张已删除图片`
    : `当前显示全部图片：有效 ${count} 张，已删除 ${removedCount} 张`;

  if (!count) {
    setProgress('preview', 0, '尚未选择图片。');
  } else if (count < MIN_SKU_FILES) {
    setProgress('preview', 0, `当前仅 ${count} 张图片，至少需要 ${MIN_SKU_FILES} 张。`);
  } else if (count > MAX_SKU_FILES) {
    setProgress('preview', 0, `当前 ${count} 张图片，超过 ${MAX_SKU_FILES} 张上限，请删除后再继续。`);
  }
}

function getVisibleSkuItems() {
  return state.activeOnlyView
    ? state.skuItems.filter((item) => !item.removed)
    : state.skuItems;
}

function renderSkuFileList() {
  const list = document.getElementById('skuFileList');
  if (!state.skuItems.length) {
    list.innerHTML = '<div class="empty-card">还没有选择图片。请上传同一个商品的 3-8 张多角度照片，系统会先去重并校验是否混入了其他商品。</div>';
    return;
  }

  const visibleItems = getVisibleSkuItems();
  if (!visibleItems.length) {
    list.innerHTML = '<div class="empty-card">当前筛选为“只看有效图片”，但没有可显示的有效图片。你可以关闭筛选或重新添加图片。</div>';
    return;
  }

  list.innerHTML = visibleItems.map((item) => `
    <article class="thumb-card ${item.removed ? 'is-removed' : ''}">
      <div class="thumb-media">
        <img src="${item.url}" alt="${escapeHtml(item.file.name)}" />
      </div>
      <div class="thumb-body">
        <p class="thumb-title">${escapeHtml(item.file.name)}</p>
        <p class="thumb-meta">
          ${formatBytes(item.file.size)}
          ${item.width && item.height ? ` · ${item.width}×${item.height}` : ''}
        </p>
        <div class="thumb-tag-row">
          <span class="file-tag ${escapeHtml(item.status)}">${escapeHtml(item.label)}</span>
          ${item.removed ? '' : `<button class="thumb-remove" type="button" data-remove-id="${item.id}">删除</button>`}
        </div>
      </div>
    </article>
  `).join('');
}

async function runIngestPreview() {
  const activeItems = getActiveSkuItems();
  if (activeItems.length < MIN_SKU_FILES) {
    window.alert(`请至少选择 ${MIN_SKU_FILES} 张同一商品照片。`);
    return;
  }
  if (activeItems.length > MAX_SKU_FILES) {
    window.alert(`单次最多只能上传 ${MAX_SKU_FILES} 张，请先删除多余图片。`);
    return;
  }

  const formData = new FormData();
  activeItems.forEach((item) => formData.append('files', item.file));

  document.getElementById('skuAnalyzeBtn').disabled = true;
  document.getElementById('skuPublishBtn').disabled = true;
  setProgress('preview', 0, '正在上传图片并执行预分析...');
  setOcrSignal('系统正在去重并检查这批图片是否属于同一个商品。', 'info');

  try {
    const result = await uploadWithProgress(`${API_BASE}/sku/ingest-preview`, formData, (percent) => {
      setProgress('preview', percent, `正在上传与预分析... ${percent}%`);
    });

    state.ingestJobId = result.job_id || '';
    state.lastPreview = result;
    applyMetadataSuggestion(result.suggestion || {});
    applyPreviewResult(result);
    refreshSkuControls();
  } catch (error) {
    setProgress('preview', 0, error.message || '预分析失败。');
    setOcrSignal('预分析失败，请检查图片后重试。', 'error');
  } finally {
    document.getElementById('skuAnalyzeBtn').disabled = getActiveSkuItems().length < MIN_SKU_FILES || getActiveSkuItems().length > MAX_SKU_FILES;
  }
}

function applyPreviewResult(result) {
  const acceptedSet = new Set((result.files || []).map((item) => item.original_name));
  const duplicateSet = new Set(result.duplicate_files || []);
  const flaggedSet = new Set(result.flagged_files || []);

  state.skuItems.forEach((item) => {
    if (item.removed) {
      item.status = 'removed';
      item.label = '已删除';
      return;
    }

    if (flaggedSet.has(item.file.name)) {
      item.status = 'flagged';
      item.label = '疑似非同一商品';
      return;
    }
    if (duplicateSet.has(item.file.name)) {
      item.status = 'duplicate';
      item.label = '重复已跳过';
      return;
    }
    if (acceptedSet.has(item.file.name)) {
      item.status = 'accepted';
      item.label = '已接收';
      return;
    }

    item.status = 'pending';
    item.label = '待分析';
  });

  renderSkuFileList();

  const accepted = Number(result.accepted_count || 0);
  const duplicates = Number(result.duplicate_count || 0);
  const previewText = `预分析完成：有效 ${accepted} 张，重复跳过 ${duplicates} 张。`;
  setProgress('preview', 100, `${previewText} ${result.validation_message || ''}`.trim());

  if (result.is_publishable) {
    const signalMessage = buildSuggestionMessage(result, true);
    setOcrSignal(signalMessage, duplicates > 0 ? 'warning' : 'success');
  } else {
    setOcrSignal(buildSuggestionMessage(result, false), 'error');
  }
}

function applyMetadataSuggestion(suggestion) {
  const brandInput = document.getElementById('brandInput');
  const productInput = document.getElementById('productInput');

  if (!brandInput.value.trim() && suggestion.brand) {
    brandInput.value = suggestion.brand;
  }
  if (!productInput.value.trim() && suggestion.product_name) {
    productInput.value = suggestion.product_name;
  }
}

function buildSuggestionMessage(result, isSuccess) {
  const suggestion = result.suggestion || {};
  const parts = [];

  if (result.validation_message) {
    parts.push(result.validation_message);
  }
  if (suggestion.brand) {
    parts.push(`品牌建议：${suggestion.brand}`);
  }
  if (suggestion.product_name) {
    parts.push(`商品名建议：${suggestion.product_name}`);
  }
  if (suggestion.ocr_text) {
    parts.push(`OCR 文本：${suggestion.ocr_text}`);
  } else if (isSuccess) {
    parts.push('未识别到明显 OCR 文本，可手动补充商品信息。');
  }

  return parts.join(' | ');
}

async function publishSkuBatch() {
  if (!state.ingestJobId || !state.lastPreview) {
    window.alert('请先执行预分析。');
    return;
  }
  if (!state.lastPreview.is_publishable) {
    window.alert(state.lastPreview.validation_message || '当前批次未通过校验，无法发布。');
    return;
  }

  const payload = {
    job_id: state.ingestJobId,
    brand: document.getElementById('brandInput').value.trim(),
    product_name: document.getElementById('productInput').value.trim(),
    variant: document.getElementById('variantInput').value.trim(),
    size: document.getElementById('sizeInput').value.trim(),
    category: document.getElementById('categoryInput').value,
  };

  if (!payload.product_name) {
    window.alert('请先填写商品名。');
    return;
  }

  document.getElementById('skuPublishBtn').disabled = true;
  document.getElementById('gotoRecognitionBtn').classList.add('hidden');
  setProgress('publish', 5, '正在排队发布并同步索引...');
  setOcrSignal('已提交发布任务，正在同步 CLIP 特征与索引。', 'info');

  try {
    const response = await fetch(`${API_BASE}/sku/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await parseJson(response);
    if (!response.ok) {
      throw new Error(result.detail || `发布失败，HTTP ${response.status}`);
    }

    stopPublishPolling();
    pollPublishJob(result.job_id || state.ingestJobId);
  } catch (error) {
    document.getElementById('skuPublishBtn').disabled = false;
    setProgress('publish', 0, error.message || '发布失败。');
    setOcrSignal(error.message || '发布失败。', 'error');
  }
}

function pollPublishJob(jobId) {
  const tick = async () => {
    try {
      const response = await fetch(`${API_BASE}/sku/ingest/jobs/${encodeURIComponent(jobId)}`);
      const result = await parseJson(response);
      if (!response.ok) {
        throw new Error(result.detail || `任务查询失败，HTTP ${response.status}`);
      }

      const progress = Number(result.progress || 0);
      const message = result.message || `当前阶段：${result.phase || '处理中'}`;
      setProgress('publish', progress, message);

      if (result.status === 'completed') {
        stopPublishPolling();
        document.getElementById('skuPublishBtn').disabled = false;
        document.getElementById('gotoRecognitionBtn').classList.remove('hidden');
        setOcrSignal('商品已入库并同步索引，可前往识别中心测试。', 'success');
        loadSkuLibrary();
      } else if (result.status === 'failed') {
        stopPublishPolling();
        document.getElementById('skuPublishBtn').disabled = false;
        setOcrSignal(`发布失败：${message}`, 'error');
      }
    } catch (error) {
      stopPublishPolling();
      document.getElementById('skuPublishBtn').disabled = false;
      setProgress('publish', 0, error.message || '任务查询失败。');
      setOcrSignal(error.message || '任务查询失败。', 'error');
    }
  };

  tick();
  state.publishPollTimer = window.setInterval(tick, JOB_POLL_MS);
}

function stopPublishPolling() {
  if (state.publishPollTimer) {
    window.clearInterval(state.publishPollTimer);
    state.publishPollTimer = null;
  }
}

function resetSkuBatch(options = {}) {
  const keepForm = Boolean(options.keepForm);

  stopPublishPolling();
  state.skuItems.forEach((item) => {
    if (item.url) {
      URL.revokeObjectURL(item.url);
    }
  });

  state.skuItems = [];
  state.ingestJobId = '';
  state.lastPreview = null;
  state.lastRemovedItemId = '';
  state.activeOnlyView = false;
  document.getElementById('skuAnalyzeBtn').disabled = true;
  document.getElementById('skuPublishBtn').disabled = true;
  document.getElementById('gotoRecognitionBtn').classList.add('hidden');
  document.getElementById('undoRemoveBtn').disabled = true;
  document.getElementById('toggleActiveOnlyBtn').textContent = '只看有效图片';
  document.getElementById('skuToolbarInfo').textContent = '当前显示全部图片';
  document.getElementById('skuUploadTitle').textContent = '拖拽商品照片到这里';
  document.getElementById('skuUploadSubtitle').textContent = '请上传同一个商品的 3-8 张不同角度照片';
  document.getElementById('skuFileList').innerHTML = '<div class="empty-card">还没有选择图片。请上传同一个商品的 3-8 张多角度照片，系统会先去重并校验是否混入了其他商品。</div>';
  setProgress('preview', 0, '尚未开始预分析。');
  setProgress('publish', 0, '尚未开始发布。');
  setOcrSignal('OCR 建议、批次校验结果与错误提示会显示在这里。', 'info');

  if (!keepForm) {
    document.getElementById('brandInput').value = '';
    document.getElementById('productInput').value = '';
    document.getElementById('variantInput').value = '';
    document.getElementById('sizeInput').value = '';
    document.getElementById('categoryInput').value = 'Beverages';
  }
}

function setProgress(kind, percent, message) {
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  const textNode = document.getElementById(`${kind}ProgressText`);
  const barNode = document.getElementById(`${kind}ProgressBar`);
  const statusNode = document.getElementById(`${kind}Status`);

  textNode.textContent = `${safePercent}%`;
  barNode.style.width = `${safePercent}%`;
  statusNode.textContent = message;
}

function setOcrSignal(message, tone) {
  const box = document.getElementById('ocrSignal');
  box.textContent = message;
  box.className = `signal-box state-${tone || 'info'}`;
}

async function loadSkuLibrary() {
  const target = document.getElementById('skuLibraryTable');
  target.innerHTML = '<p class="progress-caption">正在加载 SKU 列表...</p>';

  try {
    const response = await fetch(`${API_BASE}/sku/list?page=1&page_size=12`);
    const result = await parseJson(response);
    if (!response.ok) {
      throw new Error(result.detail || `列表请求失败，HTTP ${response.status}`);
    }

    if (!Array.isArray(result.items) || !result.items.length) {
      target.innerHTML = '<p class="progress-caption">当前还没有 SKU 记录。</p>';
      return;
    }

    const rows = result.items.map((item) => `
      <tr>
        <td>${escapeHtml(item.name || item.sku_id)}</td>
        <td>${escapeHtml(item.sku_id || '')}</td>
        <td>${escapeHtml(item.category || '--')}</td>
        <td>${Number(item.image_count || 0)}</td>
      </tr>
    `).join('');

    target.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>商品名</th>
            <th>SKU ID</th>
            <th>分类</th>
            <th>图片数</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (error) {
    target.innerHTML = `<p class="progress-caption">${escapeHtml(error.message || '加载失败。')}</p>`;
  }
}

async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  const badge = document.getElementById('backendBadge');

  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await parseJson(response);
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    if (data.status === 'healthy') {
      dot.className = 'status-dot healthy';
      const gpuText = data.gpu_memory_used_mb != null && data.gpu_memory_total_mb != null
        ? ` | GPU ${Math.round(data.gpu_memory_used_mb)}/${Math.round(data.gpu_memory_total_mb)} MB`
        : '';
      text.textContent = `服务正常 | ${data.sku_count} 个 SKU | ${data.index_size} 条向量${gpuText}`;
      badge.textContent = String(data.model_backend || '').toUpperCase();
      badge.classList.add('show');
    } else {
      dot.className = 'status-dot';
      text.textContent = '服务启动中...';
      badge.classList.remove('show');
    }
  } catch (error) {
    dot.className = 'status-dot error';
    text.textContent = 'API 暂不可用';
    badge.classList.remove('show');
  }
}

function setLoading(isVisible, message = '处理中...') {
  const overlay = document.getElementById('loadingOverlay');
  const label = document.getElementById('loadingText');
  label.textContent = message;
  overlay.classList.toggle('hidden', !isVisible);
}

function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.responseType = 'json';

    xhr.upload.addEventListener('progress', (event) => {
      if (!event.lengthComputable) {
        return;
      }
      const percent = Math.round((event.loaded / event.total) * 100);
      onProgress(percent);
    });

    xhr.onload = () => {
      const response = xhr.response || safeJsonParse(xhr.responseText);
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(response);
        return;
      }
      reject(new Error(response?.detail || `上传失败，HTTP ${xhr.status}`));
    };

    xhr.onerror = () => reject(new Error('上传过程发生网络错误。'));
    xhr.send(formData);
  });
}

async function parseJson(response) {
  try {
    return await response.json();
  } catch (error) {
    return {};
  }
}

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch (error) {
    return {};
  }
}

function formatDuration(ms) {
  const value = Number(ms || 0);
  return `${value.toFixed(1)} ms`;
}

function formatBytes(bytes) {
  if (!bytes) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = Number(bytes);
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
