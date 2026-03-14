<template>
  <div class="page demo-page" :class="'ds-' + selectedDataset">
    <div class="hero">
      <div class="hero-title">多领域半监督识别演示</div>
      <div class="hero-controls">
        <el-radio-group ref="datasetSwitchEl" class="dataset-switch" v-model="selectedDataset" size="large">
          <el-radio-button label="food101">Food101</el-radio-button>
          <el-radio-button label="stl10">STL-10</el-radio-button>
          <el-radio-button label="eurosat">EuroSAT</el-radio-button>
          <el-radio-button label="cifar100">CIFAR-100</el-radio-button>
        </el-radio-group>
        <el-tag class="dataset-chip" size="small" effect="light">{{ datasetChipText }}</el-tag>
      </div>
    </div>

    <transition name="panel-swap" mode="out-in">
      <el-row class="equal-row" :gutter="16" :key="selectedDataset">
        <el-col class="col-stretch" :xs="24" :md="10">
          <el-card class="glass-card panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div class="panel-title">待测图像录入</div>
              </div>
            </template>

            <el-upload
              class="predictor-upload"
              drag
              :action="'http://localhost:8000/api/predict/' + selectedDataset"
              :on-success="handleUploadSuccess"
              :on-error="handleUploadError"
              :before-upload="beforeUpload"
              :show-file-list="false"
            >
              <div class="upload-core">
                <img v-if="previewUrl" class="upload-preview" :src="previewUrl" alt="uploaded preview" />
                <el-icon v-else class="ai-pulse"><Cpu /></el-icon>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  当前数据集：<span class="dataset-highlight">{{ datasetChipText }}</span>
                </div>
              </template>
            </el-upload>
          </el-card>
        </el-col>

        <el-col class="col-stretch" :xs="24" :md="14">
          <el-card class="glass-card panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div class="panel-title">神经推理分析 (Top-5)</div>
              </div>
            </template>

            <div v-if="loading" class="panel-body">
              <el-skeleton :rows="6" animated />
            </div>

            <div v-else-if="predictionResults" class="panel-body">
              <div v-for="(item, index) in predictionResults" :key="(item.label_en || item.label || '') + '-' + index" class="pill-item">
                <div class="pill-head">
                  <div class="pill-label">{{ displayLabel(item) }}</div>
                  <div class="pill-score">{{ formatPercent(displayPercents[index] ?? 0) }}</div>
                </div>
                <div class="pill-bar" :style="pillStyle(item.score)">
                  <div class="pill-bar__fill" :style="pillFillStyle(item.score, displayPercents[index] ?? 0)" />
                </div>
              </div>
            </div>

            <div v-else class="panel-body empty">
              <el-empty description="等待上传图片进行识别" />
            </div>
          </el-card>
        </el-col>
      </el-row>
    </transition>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Cpu } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { formatLabelDisplay } from '../lib/labelMaps'

const selectedDataset = ref('food101')
const predictionResults = ref(null)
const loading = ref(false)
const displayPercents = ref([])
const datasetSwitchEl = ref(null)
const previewUrl = ref('')

const datasetChipText = computed(() => {
  const map = {
    food101: '美食 (Food101)',
    stl10: '工业 (STL-10)',
    eurosat: '遥感 (EuroSAT)',
    cifar100: '通用 (CIFAR-100)',
  }
  return map[selectedDataset.value] || selectedDataset.value
})

const datasetTagType = computed(() => {
  const map = {
    food101: 'warning',
    stl10: 'info',
    eurosat: 'success',
    cifar100: 'primary',
  }
  return map[selectedDataset.value] || 'primary'
})

function clamp01(value) {
  return Math.min(1, Math.max(0, value))
}

function lerp(a, b, t) {
  return a + (b - a) * t
}

function hexToRgb(hex) {
  const normalized = hex.replace('#', '').trim()
  const bigint = parseInt(normalized, 16)
  return {
    r: (bigint >> 16) & 255,
    g: (bigint >> 8) & 255,
    b: bigint & 255,
  }
}

function rgbToCss({ r, g, b }, alpha = 1) {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

// Confidence color: low -> deep blue, high -> rose
function confidenceColor(score) {
  const t = clamp01(score)
  const low = hexToRgb('#1E293B')
  const high = hexToRgb('#F472B6')
  return {
    r: Math.round(lerp(low.r, high.r, t)),
    g: Math.round(lerp(low.g, high.g, t)),
    b: Math.round(lerp(low.b, high.b, t)),
  }
}

function pillStyle(score) {
  const c = confidenceColor(score)
  return {
    '--pill': rgbToCss(c, 1),
    '--pill-glow': rgbToCss(c, 0.35),
  }
}

function pillFillStyle(score, currentPercent) {
  const pct = clamp01(currentPercent / 100)
  const c = confidenceColor(score)
  return {
    width: `${(pct * 100).toFixed(2)}%`,
    background: `linear-gradient(90deg, ${rgbToCss(c, 0.75)}, ${rgbToCss(c, 1)})`,
    boxShadow: `0 0 18px ${rgbToCss(c, 0.35)}`,
  }
}

function formatPercent(value) {
  return `${Number(value).toFixed(1)}%`
}

function displayLabel(itemOrLabel) {
  return formatLabelDisplay(itemOrLabel)
}

function animatePercent(index, target, durationMs = 850) {
  const start = performance.now()
  const from = displayPercents.value[index] ?? 0

  const tick = (now) => {
    const t = Math.min(1, (now - start) / durationMs)
    const eased = 1 - Math.pow(1 - t, 3)
    displayPercents.value[index] = lerp(from, target, eased)
    if (t < 1) requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}

function revokePreviewUrl() {
  if (previewUrl.value && previewUrl.value.startsWith('blob:')) {
    try {
      URL.revokeObjectURL(previewUrl.value)
    } catch {
      // ignore
    }
  }
  previewUrl.value = ''
}

function setPreviewFromFile(file) {
  if (!(file instanceof Blob)) return
  revokePreviewUrl()
  previewUrl.value = URL.createObjectURL(file)
}

const beforeUpload = (file) => {
  setPreviewFromFile(file)
  loading.value = true
  predictionResults.value = null
  displayPercents.value = []
  return true
}

const handleUploadSuccess = (response) => {
  loading.value = false
  if (response?.success) {
    predictionResults.value = response.predictions
    nextTick(() => {
      const items = predictionResults.value || []
      displayPercents.value = items.map(() => 0)
      items.forEach((item, idx) => animatePercent(idx, clamp01(item.score) * 100))
    })
    ElMessage.success('识别成功！')
  } else {
    ElMessage.error('识别失败：' + (response?.error || 'unknown'))
  }
}

const handleUploadError = () => {
  loading.value = false
  ElMessage.error('请求失败：后端服务未启动或无法连接 (localhost:8000)')
}

watch(selectedDataset, () => {
  predictionResults.value = null
  displayPercents.value = []
  revokePreviewUrl()
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
})

function updateSegmentIndicator() {
  const el = datasetSwitchEl.value?.$el ?? datasetSwitchEl.value
  if (!el) return

  const checked = el.querySelector('.el-radio-button__original-radio:checked')
  const inner = checked?.nextElementSibling
  if (!(inner instanceof HTMLElement)) return

  const elRect = el.getBoundingClientRect()
  const innerRect = inner.getBoundingClientRect()
  const left = innerRect.left - elRect.left
  const width = innerRect.width

  const leftPx = Math.round(left)
  const widthPx = Math.round(width)

  el.style.setProperty('--seg-left', `${leftPx}px`)
  el.style.setProperty('--seg-width', `${widthPx}px`)
}

const onResize = () => updateSegmentIndicator()

onMounted(() => {
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
  window.addEventListener('resize', onResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  revokePreviewUrl()
})
</script>

<style scoped>

.hero {
  text-align: center;
  padding: 30px 0 28px;
}

.hero-title {
  font-size: 24px;
  line-height: 1.18;
  font-weight: 700;
  letter-spacing: 0.4px;
  color: #1E293B;
}

.hero-controls {
  margin-top: 20px;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.dataset-chip {
  border-radius: 999px;
  color: rgba(71, 85, 105, 0.72);
  font-size: 14px;
  line-height: 20px;
  border: 1px solid rgba(195, 177, 225, 0.18);
  background: rgba(255, 255, 255, 0.72);
}

.dataset-highlight {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-weight: 750;
  color: rgba(30, 41, 59, 0.86);
  border: 1px solid rgba(195, 177, 225, 0.18);
  background: rgba(255, 255, 255, 0.72);
}

.demo-page.ds-food101 .dataset-chip,
.demo-page.ds-food101 .dataset-highlight {
  background: rgba(248, 200, 220, 0.42);
  border-color: rgba(248, 200, 220, 0.55);
}

.demo-page.ds-stl10 .dataset-chip,
.demo-page.ds-stl10 .dataset-highlight {
  background: rgba(178, 226, 210, 0.42);
  border-color: rgba(178, 226, 210, 0.58);
}

.demo-page.ds-eurosat .dataset-chip,
.demo-page.ds-eurosat .dataset-highlight {
  background: rgba(195, 177, 225, 0.38);
  border-color: rgba(195, 177, 225, 0.56);
}

.demo-page.ds-cifar100 .dataset-chip,
.demo-page.ds-cifar100 .dataset-highlight {
  background: rgba(224, 242, 255, 0.55);
  border-color: rgba(224, 242, 255, 0.78);
}

.equal-row {
  align-items: stretch;
}

.col-stretch {
  display: flex;
}

.panel-card {
  width: 100%;
  border-radius: 28px;
}

.panel-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.panel-title {
  font-size: 20px;
  font-weight: 800;
  color: var(--el-text-color-primary);
}

.panel-body {
  min-height: 360px;
}

.panel-body.empty {
  display: flex;
  align-items: center;
  justify-content: center;
}

.predictor-upload {
  width: 100%;
}

/* Upload Core: animated gradient border (no dashed border) */
.predictor-upload :deep(.el-upload-dragger) {
  border-radius: 28px;
  border: 1px solid transparent;
  padding: 24px;
  background:
    linear-gradient(0deg, rgba(255, 255, 255, 0.65), rgba(255, 255, 255, 0.65)) padding-box,
    linear-gradient(120deg, rgba(244, 114, 182, 0.82), rgba(195, 177, 225, 0.86), rgba(30, 41, 59, 0.70), rgba(244, 114, 182, 0.82)) border-box;
  background-size: 100% 100%, 280% 280%;
  animation: border-flow 7s ease-in-out infinite;
  transition: transform 180ms ease, filter 180ms ease;
  box-shadow: 0 10px 40px rgba(165, 180, 252, 0.10);
}

.predictor-upload :deep(.el-upload-dragger:hover) {
  transform: translateY(-1px);
  filter: brightness(1.05);
}

.predictor-upload :deep(.el-upload-dragger.is-dragover),
.predictor-upload :deep(.el-upload.is-dragover .el-upload-dragger) {
  filter: brightness(1.1);
  box-shadow: 0 0 0 1px rgba(244, 114, 182, 0.24), 0 20px 80px rgba(0, 0, 0, 0.05);
}

@keyframes border-flow {
  0% {
    background-position: 0 0, 0% 50%;
  }
  50% {
    background-position: 0 0, 100% 50%;
  }
  100% {
    background-position: 0 0, 0% 50%;
  }
}

.upload-core {
  height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 18px;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.06);
  background: rgba(255, 255, 255, 0.35);
}

.ai-pulse {
  font-size: 54px;
  color: rgba(71, 85, 105, 0.88);
  filter: drop-shadow(0 0 10px rgba(244, 114, 182, 0.18));
  animation: ai-pulse 1.9s ease-in-out infinite;
}

@keyframes ai-pulse {
  0%,
  100% {
    transform: translateY(0) scale(1);
    opacity: 0.86;
  }
  50% {
    transform: translateY(-2px) scale(1.04);
    opacity: 1;
  }
}

.pill-item {
  margin-bottom: 14px;
}

.pill-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}

.pill-label {
  font-weight: 650;
  color: rgba(71, 85, 105, 0.95);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pill-score {
  font-weight: 800;
  letter-spacing: 0.6px;
  color: rgba(71, 85, 105, 0.90);
}

.pill-bar {
  position: relative;
  height: 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.62);
  border: none;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.04);
  transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
  transform-origin: center;
  will-change: transform;
}

.pill-bar::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  opacity: 0;
  transition: opacity 160ms ease;
  /* subtle highlight layer so even the “empty” part reacts */
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.55), rgba(255, 255, 255, 0.18)),
    repeating-linear-gradient(135deg, rgba(195, 177, 225, 0.18) 0 8px, rgba(224, 242, 255, 0.12) 8px 16px);
  mix-blend-mode: overlay;
}

.pill-item:hover .pill-bar {
  transform: scaleY(1.22);
  filter: brightness(1.03);
  box-shadow:
    inset 0 0 0 1px rgba(0, 0, 0, 0.04),
    0 0 0 2px rgba(195, 177, 225, 0.26),
    0 10px 30px rgba(165, 180, 252, 0.12);
}

.pill-item:hover .pill-bar::after {
  opacity: 1;
}

.pill-bar__fill {
  height: 100%;
  border-radius: 999px;
  background: var(--pill);
  box-shadow: 0 0 18px var(--pill-glow);
  transition: width 120ms linear;
}
</style>
