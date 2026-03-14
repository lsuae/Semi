<template>
  <div class="page">
    <el-space direction="vertical" fill :size="24">
      <el-card>
        <template #header>
          <div class="topbar">
            <div class="topbar-left">
              <div class="topbar-title">算法过程可视化</div>
            </div>

            <div class="topbar-center">
              <el-radio-group
                ref="datasetSwitchEl"
                class="dataset-switch"
                v-model="selectedDataset"
                size="large"
                name="dataset"
              >
                <el-radio-button label="food101">美食</el-radio-button>
                <el-radio-button label="stl10">工业</el-radio-button>
                <el-radio-button label="eurosat">遥感</el-radio-button>
                <el-radio-button label="cifar100">通用</el-radio-button>
              </el-radio-group>
            </div>

            <div class="topbar-right">
              <el-radio-group
                ref="featureSwitchEl"
                class="feature-switch"
                v-model="activePanel"
                size="large"
                name="viz-panel"
                @change="() => requestAnimationFrame(updateSegmentIndicator)"
              >
                <el-radio-button label="tsne">t-SNE</el-radio-button>
                <el-radio-button label="pseudo">伪标签</el-radio-button>
                <el-radio-button label="clip">Top-5</el-radio-button>
              </el-radio-group>
            </div>
          </div>
        </template>
      </el-card>

      <transition name="panel-swap" mode="out-in">
        <el-row class="viz-panel-row" :gutter="0" :key="selectedDataset + '-' + activePanel">
          <el-col :span="24" v-if="activePanel === 'tsne'">
            <el-card class="viz-card">
              <template #header>
                <div class="tsne-header">
                  <div class="panel-title">特征空间映射 (t-SNE)</div>
                  <div class="tsne-header-right">
                    <div class="tsne-stage">{{ tsneStageLabel }}</div>
                    <el-slider
                      class="tsne-slider"
                      v-model="tsneStep"
                      :min="0"
                      :max="1"
                      :step="1"
                      :show-stops="false"
                      :show-tooltip="false"
                    />
                  </div>
                </div>
              </template>

              <el-row :gutter="16" class="canvas-row">
                <el-col :span="10">
                  <div class="chart-surface">
                    <div ref="tsneEl" class="chart" />
                  </div>
                </el-col>
                <el-col :span="4">
                  <div class="hover-panel">
                    <div class="hover-title">样本分析</div>
                    <div class="hover-sub">点击散点锁定样本；悬停可预览</div>

                    <div class="hover-image">
                      <el-image
                        v-if="analysisInfo.image_src"
                        :src="analysisInfo.image_src"
                        fit="contain"
                        style="width: 100%; height: 160px"
                      />
                      <el-empty v-else description="暂无图片" />
                    </div>

                    <div class="hover-meta">
                      <div class="meta-row"><span class="k">idx</span><span class="v">{{ analysisInfo.idx ?? '-' }}</span></div>
                      <div class="meta-row"><span class="k">target</span><span class="v">{{ analysisInfo.target ?? '-' }}</span></div>
                      <div class="meta-row"><span class="k">pred</span><span class="v">{{ analysisInfo.pred ?? '-' }}</span></div>
                      <div class="meta-row"><span class="k">置信度</span><span class="v">{{ analysisInfo.confidence ?? '-' }}</span></div>
                    </div>
                  </div>
                </el-col>

                <el-col :span="10">
                  <div class="clip-panel clip-panel-fill">
                    <div class="clip-head">
                      <div class="clip-title">Top-5 文本匹配</div>
                      <div class="clip-mode">{{ clipMeta.mode || '-' }}</div>
                    </div>
                    <div class="clip-sub">idx：{{ analysisInfo.idx ?? '-' }}（点击散点更新）</div>
                    <div ref="clipEl" class="clip-mini-chart" />
                  </div>
                </el-col>
              </el-row>
          </el-card>
        </el-col>

        <el-col :span="24" v-else-if="activePanel === 'pseudo'">
          <el-card class="viz-card">
            <template #header>
              <div class="section-title">伪标签置信度分布（阈值通过比例）</div>
            </template>

            <div class="toolbar">
              <div class="toolbar-left" style="min-width: 320px">
                <div class="hint">Threshold: {{ threshold.toFixed(2) }}</div>
                <el-slider v-model="threshold" :min="0" :max="1" :step="0.01" />
              </div>
              <div class="stats">
                <div class="stat">
                  <div class="k">样本数</div>
                  <div class="v">{{ pseudoStats.n ?? '-' }}</div>
                </div>
                <div class="stat">
                  <div class="k">均值</div>
                  <div class="v">{{ formatFloat(pseudoStats.mean) }}</div>
                </div>
                <div class="stat">
                  <div class="k">通过比例</div>
                  <div class="v">{{ formatPercent(pseudoStats.pass_ratio) }}</div>
                </div>
                <div class="stat">
                  <div class="k">通过均值</div>
                  <div class="v">{{ formatFloat(pseudoStats.pass_mean_confidence) }}</div>
                </div>
              </div>
            </div>

            <div ref="pseudoEl" class="chart" />
          </el-card>
        </el-col>

        <el-col :span="24" v-else>
          <el-card class="viz-card">
            <template #header>
              <div class="section-title">CLIP 文本-视觉匹配矩阵（Top-5）</div>
            </template>

            <div class="toolbar">
              <div class="toolbar-left" style="min-width: 320px">
                <div class="hint">选择样例（或点击 t-SNE 点）</div>
                <el-select
                  v-model="selectedIdx"
                  filterable
                  clearable
                  placeholder="选择一个 idx"
                  style="max-width: 520px"
                >
                  <el-option
                    v-for="s in samples"
                    :key="s.idx"
                    :label="`idx_${s.idx} (${s.group}) target=${s.target} pred=${s.pred}`"
                    :value="s.idx"
                  />
                </el-select>
              </div>
              <div class="hint">模式：{{ clipMeta.mode || '-' }}</div>
            </div>

            <el-row :gutter="16">
              <el-col :span="8">
                <div class="image-panel">
                  <el-image
                    v-if="selectedImageSrc"
                    :src="selectedImageSrc"
                    fit="contain"
                    style="width: 100%; height: 360px"
                  />
                  <el-empty v-else description="暂无样例图片" />
                </div>
              </el-col>
              <el-col :span="16">
                <div ref="clipEl" class="chart" style="height: 360px" />
              </el-col>
            </el-row>
          </el-card>
        </el-col>
        </el-row>
      </transition>
    </el-space>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { api } from '../lib/api'

const selectedDataset = ref('food101')
const datasetSwitchEl = ref(null)
const featureSwitchEl = ref(null)
const activePanel = ref('tsne')
const tsneStep = ref(0)
const threshold = ref(0.95)

// 仅展示 Initial/Final 两个阶段：
// - Initial：iteration=0（若缺文件则后端会用 synthetic 随机初态降级）
// - Final：不传 iteration（后端读取最终 coords_<dataset>.json）
const tsneMeta = ref({ source: '-', missing: false })

const tsneEl = ref(null)
const pseudoEl = ref(null)
const clipEl = ref(null)

const samples = ref([])
const selectedIdx = ref(null)
const selectedImageUrl = ref(null)
const clipMeta = ref({ mode: '' })

const hoverRaw = ref(null)
const selectedRaw = ref(null)

let tsneChart = null
let pseudoChart = null
let clipChart = null

const pseudoStats = ref({})

const tsneMarks = computed(() => ({ 0: 'Initial', 1: 'Final' }))

const tsneStageLabel = computed(() => (tsneStep.value === 0 ? 'Initial' : 'Final'))

const selectedImageSrc = computed(() => {
  const base = api?.defaults?.baseURL || ''
  const url = selectedImageUrl.value
  if (!url) return ''
  if (url.startsWith('http://') || url.startsWith('https://')) return url
  if (!base) return url
  return base.replace(/\/$/, '') + url
})

const hoverInfo = computed(() => {
  const raw = hoverRaw.value
  const base = api?.defaults?.baseURL || ''
  const url = raw?.image_url
  const image_src = !url
    ? ''
    : (url.startsWith('http://') || url.startsWith('https://'))
      ? url
      : base
        ? base.replace(/\/$/, '') + url
        : url

  return {
    idx: raw?.idx ?? null,
    target: raw?.target ?? null,
    pred: raw?.pred ?? null,
    confidence: raw?.confidence ?? null,
    image_src,
  }
})

const analysisInfo = computed(() => {
  // Prefer the clicked/selected sample; fallback to hover preview.
  const base = api?.defaults?.baseURL || ''

  const s = selectedRaw.value
  const h = hoverRaw.value

  const idx = (selectedIdx.value ?? s?.idx ?? h?.idx ?? null)
  const target = (s?.target ?? h?.target ?? null)
  const pred = (s?.pred ?? h?.pred ?? null)
  const confidence = (s?.confidence ?? h?.confidence ?? null)

  const url = (selectedImageUrl.value ?? s?.image_url ?? h?.image_url ?? '')
  const image_src = !url
    ? ''
    : (url.startsWith('http://') || url.startsWith('https://'))
      ? url
      : base
        ? base.replace(/\/$/, '') + url
        : url

  return { idx, target, pred, confidence, image_src }
})

function formatFloat(v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return '-'
  return v.toFixed(4)
}

function formatPercent(v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return '-'
  return (v * 100).toFixed(2) + '%'
}

function isPanel(name) {
  return activePanel.value === name
}

async function loadTsne() {
  try {
    const params = { limit: 50000 }
    // 0=Initial: 显式请求 iteration=0；1=Final: 不传 iteration，读取最终坐标文件
    if (tsneStep.value === 0) params.iteration = 0

    const { data } = await api.get(`/api/viz/${selectedDataset.value}/coords`, { params })
    const points = Array.isArray(data?.points) ? data.points : []

    tsneMeta.value = { source: data?.source || '-', missing: !!data?.missing }

    tsneChart?.off('click')
    tsneChart?.on('click', async (params) => {
      const item = params?.data
      const v = item?.value
      if (!Array.isArray(v) || v.length < 4) return
      const idx = v[3]
      const imageUrl = v[4] || null
      if (typeof idx !== 'number') return
      selectedIdx.value = idx
      if (imageUrl) selectedImageUrl.value = imageUrl
      selectedRaw.value = {
        idx: v[3],
        image_url: v[4] || null,
        target: v[5] ?? null,
        pred: v[6] ?? null,
        group: v[7] ?? null,
        confidence: null,
      }

      // Top-5 is embedded in t-SNE view too.
      if (isPanel('clip') || isPanel('tsne')) await loadClipTopk()
    })

    tsneChart?.off('mouseover')
    tsneChart?.on('mouseover', (params) => {
      if (params?.seriesType !== 'scatter') return
      const v = params?.data?.value
      if (!Array.isArray(v) || v.length < 8) return
      hoverRaw.value = {
        idx: v[3],
        image_url: v[4],
        target: v[5],
        pred: v[6],
        group: v[7],
        confidence: null,
      }
    })
    tsneChart?.off('globalout')
    tsneChart?.on('globalout', () => {
      hoverRaw.value = null
    })

    tsneChart?.setOption({
      animationDuration: 800,
      animationDurationUpdate: 800,
      animationEasing: 'cubicOut',
      animationEasingUpdate: 'cubicOut',
      tooltip: {
        trigger: 'item',
        formatter: (p) => {
          const v = p?.data?.value
          if (!Array.isArray(v)) return ''
          const label = v[2]
          const idx = v[3]
          return `idx: ${idx}<br/>label: ${label}<br/>x: ${Number(v[0]).toFixed(3)}<br/>y: ${Number(v[1]).toFixed(3)}`
        },
      },
      xAxis: { type: 'value' },
      yAxis: { type: 'value' },
      series: [
        {
          id: 'tsne',
          type: 'scatter',
          symbolSize: 4,
          // 用稳定 id（idx）绑定点，切换 Initial/Final 时会平滑移动
          data: points.map((p) => ({
            id: p.idx,
            value: [p.x, p.y, p.label, p.idx, p.image_url, p.target, p.pred, p.group],
          })),
          encode: { x: 0, y: 1 },
        },
      ],
    }, true)
  } catch (e) {
    ElMessage.error('t-SNE 数据加载失败（请确认后端 API 已启动）')
  }
}

async function loadPseudo() {
  try {
    const { data } = await api.get(`/api/viz/${selectedDataset.value}/pseudo-confidence`, {
      params: { threshold: threshold.value, bins: 20 },
    })
    pseudoStats.value = data

    const edges = Array.isArray(data?.bin_edges) ? data.bin_edges : []
    const counts = Array.isArray(data?.bin_counts) ? data.bin_counts : []
    const xLabels = edges.length >= 2 ? edges.slice(0, -1).map((x, i) => `${x.toFixed(2)}-${edges[i + 1].toFixed(2)}`) : []

    pseudoChart?.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: xLabels, axisLabel: { interval: 1, rotate: 30 } },
      yAxis: { type: 'value' },
      series: [{ type: 'bar', data: counts }],
    })
  } catch (e) {
    ElMessage.error('伪标签置信度加载失败（请确认后端 API 已启动）')
  }
}

async function loadSamples() {
  try {
    const { data } = await api.get(`/api/viz/${selectedDataset.value}/samples`, { params: { limit: 500 } })
    const arr = Array.isArray(data?.samples) ? data.samples : []
    samples.value = arr
    if (selectedIdx.value == null && arr.length > 0) {
      selectedIdx.value = arr[0].idx
      selectedImageUrl.value = arr[0].url
    }
  } catch (e) {
    samples.value = []
  }
}

async function loadClipTopk() {
  if (!(isPanel('clip') || isPanel('tsne'))) return
  if (selectedIdx.value == null) {
    clipMeta.value = { mode: '' }
    clipChart?.setOption({ series: [{ type: 'bar', data: [] }] })
    return
  }

  try {
    const { data } = await api.get(`/api/viz/${selectedDataset.value}/text-match/topk`, {
      params: { idx: selectedIdx.value, k: 5 },
    })

    clipMeta.value = { mode: data?.mode || '' }
    if (data?.image_url) selectedImageUrl.value = data.image_url

    const topk = Array.isArray(data?.topk) ? data.topk : []
    const labels = topk.map((x) => x.label_zh || x.label)
    const scores = topk.map((x) => x.score)

    const compact = isPanel('tsne')
    const panelWidth = clipEl.value instanceof HTMLElement ? clipEl.value.clientWidth : 0
    const tight = compact && panelWidth > 0 && panelWidth < 360
    const gridLeft = compact ? (tight ? 110 : 140) : 120
    const yLabelWidth = compact ? (tight ? 128 : 200) : 220

    clipChart?.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: gridLeft, right: 16, top: 12, bottom: 12 },
      xAxis: compact
        ? { type: 'value', name: 'score', axisLabel: { fontSize: 10 } }
        : { type: 'value', name: 'score' },
      yAxis: {
        type: 'category',
        data: labels,
        inverse: true,
        axisLabel: compact
          ? { width: yLabelWidth, overflow: 'truncate' }
          : undefined,
      },
      series: [{ type: 'bar', data: scores, barMaxWidth: 16 }],
    })
  } catch (e) {
    ElMessage.error('Top-5 匹配加载失败（请确认后端 API 已启动）')
  }
}

function initCharts() {
  if (tsneEl.value && !tsneChart) tsneChart = echarts.init(tsneEl.value)
  if (pseudoEl.value && !pseudoChart) pseudoChart = echarts.init(pseudoEl.value)
  if (clipEl.value && !clipChart) clipChart = echarts.init(clipEl.value)
}

async function waitForChartContainer(elRef, timeoutMs = 1200) {
  const start = performance.now()
  while (performance.now() - start < timeoutMs) {
    await nextTick()
    await new Promise((resolve) => requestAnimationFrame(resolve))

    const el = elRef.value
    if (el instanceof HTMLElement) {
      const w = el.clientWidth
      const h = el.clientHeight
      if (w > 0 && h > 0) return el
    }
  }
  return null
}

function disposeCharts() {
  tsneChart?.dispose(); tsneChart = null
  pseudoChart?.dispose(); pseudoChart = null
  clipChart?.dispose(); clipChart = null
}

async function reloadActive() {
  if (isPanel('tsne')) {
    await waitForChartContainer(tsneEl)
    await waitForChartContainer(clipEl)
    initCharts()
    await loadTsne()
    await loadClipTopk()
    return
  }
  if (isPanel('pseudo')) {
    await waitForChartContainer(pseudoEl)
    initCharts()
    await loadPseudo()
    return
  }

  // clip
  await waitForChartContainer(clipEl)
  initCharts()
  await loadSamples()
  await loadClipTopk()
}

function updateSegmentIndicator() {
  const els = [datasetSwitchEl.value, featureSwitchEl.value]
  for (const r of els) {
    const el = r?.$el ?? r
    if (!el) continue

    const checked = el.querySelector('.el-radio-button__original-radio:checked')
    const inner = checked?.nextElementSibling
    if (!(inner instanceof HTMLElement)) continue

    const elRect = el.getBoundingClientRect()
    const innerRect = inner.getBoundingClientRect()
    const left = innerRect.left - elRect.left
    const width = innerRect.width

    const leftPx = Math.round(left)
    const widthPx = Math.round(width)

    el.style.setProperty('--seg-left', `${leftPx}px`)
    el.style.setProperty('--seg-width', `${widthPx}px`)
  }
}

onMounted(() => {
  initCharts()
  reloadActive()
  window.addEventListener('resize', handleResize)
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  disposeCharts()
})

function handleResize() {
  tsneChart?.resize()
  pseudoChart?.resize()
  clipChart?.resize()
  updateSegmentIndicator()
}

watch(selectedDataset, async () => {
  tsneStep.value = 0
  selectedIdx.value = null
  selectedImageUrl.value = null
  selectedRaw.value = null
  disposeCharts()
  await nextTick()
  await reloadActive()
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
})

watch(threshold, async () => {
  if (!isPanel('pseudo')) return
  await loadPseudo()
})

watch(tsneStep, async () => {
  if (!isPanel('tsne')) return
  await loadTsne()
})

watch(selectedIdx, async (v) => {
  if (!(isPanel('clip') || isPanel('tsne'))) return
  if (v == null) {
    selectedRaw.value = null
    selectedImageUrl.value = null
    await loadClipTopk()
    return
  }
  const found = samples.value.find((s) => s.idx === v)
  if (found?.url) selectedImageUrl.value = found.url
  // 如果 samples 没加载（t-SNE 视图不必预加载），不要覆盖 click handler 填的 selectedRaw。
  if (found) {
    selectedRaw.value = {
      idx: found.idx,
      image_url: found.url ?? null,
      target: found.target ?? null,
      pred: found.pred ?? null,
      group: found.group ?? null,
      confidence: null,
    }
  }
  await loadClipTopk()
})

watch(activePanel, async () => {
  disposeCharts()
  await nextTick()
  await reloadActive()
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
})
</script>

<style scoped>

/* Match DemoPage centering behavior: keep global .page padding,
   and center this page content to a readable max width. */
.page {
  width: 100%;
  max-width: none;
  margin: 0;
}

/* Top-level el-row uses negative margins when gutter > 0; keep it aligned/centered. */
.viz-panel-row {
  margin-left: 0 !important;
  margin-right: 0 !important;
}

.card-header {
  display: block;
}

.topbar {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 12px;
  height: 76px;
  padding: 0 8px;
}

.topbar-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.topbar-title {
  font-size: 24px;
  font-weight: 850;
  color: #1E293B;
  letter-spacing: 0.4px;
}

.topbar-center {
  display: flex;
  justify-content: center;
}

.topbar-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
}


.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.panel-title {
  font-size: 20px;
  font-weight: 850;
  letter-spacing: 0.3px;
}

.tsne-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.tsne-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.tsne-stage {
  font-size: 12px;
  color: rgba(71, 85, 105, 0.55);
  font-weight: 700;
}

.canvas-row {
  margin-top: 24px;
}


/* Larger breathing room inside chart cards */
.viz-card :deep(.el-card__body) {
  padding: 32px;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.title {
  font-weight: 650;
  letter-spacing: 0.2px;
}

.subtitle {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.section-title {
  font-weight: 600;
}

.chart {
  height: 420px;
  width: 100%;
  border-radius: 14px;
}


.chart-surface {
  border-radius: 20px;
  padding: 12px;
  border: 1px solid var(--el-border-color-lighter);
  background:
    repeating-linear-gradient(
      0deg,
      rgba(71, 85, 105, 0.05) 0,
      rgba(71, 85, 105, 0.05) 1px,
      transparent 1px,
      transparent 24px
    ),
    repeating-linear-gradient(
      90deg,
      rgba(71, 85, 105, 0.05) 0,
      rgba(71, 85, 105, 0.05) 1px,
      transparent 1px,
      transparent 24px
    ),
    var(--el-fill-color-lighter);
}

.hover-panel {
  border-radius: 20px;
  border: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-lighter);
  padding: 14px;
  height: 100%;
}

.clip-panel {
  border-radius: 20px;
  border: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-lighter);
  padding: 14px;
}

.clip-panel-fill {
  height: 100%;
}

.clip-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.clip-title {
  font-weight: 800;
  white-space: nowrap;
}

.clip-mode {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-weight: 650;
  white-space: nowrap;
}

.clip-sub {
  margin-top: 4px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.clip-mini-chart {
  height: 320px;
  width: 100%;
  margin-top: 10px;
  border-radius: 14px;
}

.hover-title {
  font-weight: 800;
}

.hover-sub {
  margin-top: 4px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.hover-image {
  margin-top: 12px;
  border-radius: 14px;
  overflow: hidden;
  background: var(--sb-glass);
  border: 1px solid var(--el-border-color-lighter);
}

.hover-meta {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.meta-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.meta-row .k {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.meta-row .v {
  font-weight: 750;
  color: var(--el-text-color-primary);
}

/* Feature switch: pill slider, softer than default blue */
.feature-switch {
  position: relative;
  background: rgba(30, 41, 59, 0.05);
  padding: 4px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 0;
  border: none !important;
  box-shadow: none !important;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
  --seg-left: 0px;
  --seg-width: 0px;
}

.feature-switch::before {
  content: "";
  position: absolute;
  left: var(--seg-left, 0px);
  width: var(--seg-width, 0px);
  top: 4px;
  bottom: 4px;
  border-radius: 999px;
  background: #ffffff;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: left 350ms cubic-bezier(0.4, 0, 0.2, 1), width 350ms cubic-bezier(0.4, 0, 0.2, 1);
  pointer-events: none;
  z-index: 1;
}

.feature-switch :deep(.el-radio-button__inner) {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #64748b;
  font-weight: 600;
  padding: 10px 16px;
  border-radius: 999px;
  position: relative;
  z-index: 2;
  background-image: none !important;
  border-left: 0 !important;
  border-right: 0 !important;
  outline: none !important;
  -webkit-tap-highlight-color: transparent;
}

.feature-switch :deep(.el-radio-button) {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

.feature-switch :deep(.el-radio-button__inner::before),
.feature-switch :deep(.el-radio-button__inner::after) {
  display: none !important;
}

.feature-switch :deep(.el-radio-button__original-radio:focus + .el-radio-button__inner),
.feature-switch :deep(.el-radio-button__original-radio:focus-visible + .el-radio-button__inner) {
  outline: none !important;
  box-shadow: none !important;
  border-color: transparent !important;
}

.feature-switch :deep(.el-radio-button.is-active .el-radio-button__inner) {
  color: #1e293b !important;
  font-weight: 800;
}

.feature-switch :deep(.el-radio-button__inner:hover) {
  color: #1e293b;
}

/* Thicker rounded slider for Initial/Final */
.tsne-slider {
  width: 220px;
}

.tsne-slider :deep(.el-slider__runway) {
  height: 6px;
  border-radius: 999px;
  background: rgba(30, 41, 59, 0.08);
}

.tsne-slider :deep(.el-slider__bar) {
  height: 6px;
  border-radius: 999px;
  background: rgba(30, 41, 59, 0.18);
}

.tsne-slider :deep(.el-slider__button) {
  width: 14px;
  height: 14px;
  border-width: 2px;
  border-color: rgba(30, 41, 59, 0.25);
}

.image-panel {
  width: 100%;
  border-radius: 14px;
  border: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-lighter);
  overflow: hidden;
  padding: 10px;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.toolbar-left {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.hint {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.stats {
  display: flex;
  gap: 16px;
  color: var(--el-text-color-regular);
  flex-wrap: wrap;
}

.stat {
  padding: 10px 12px;
  border: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-lighter);
  border-radius: 12px;
  min-width: 120px;
}

.k {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  margin-bottom: 2px;
}

.v {
  font-weight: 650;
  color: var(--el-text-color-primary);
}
</style>
