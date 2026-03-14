<template>
  <div class="page">
    <el-space direction="vertical" fill size="large">
      <el-card>
        <template #header>
          <div class="card-header">
            <div class="header-left">
              <div class="title">Benchmark 报告</div>
              <div class="subtitle">指标表 · 混淆矩阵 · 工程化产物</div>
            </div>
            <el-radio-group ref="datasetSwitchEl" class="dataset-switch" v-model="selectedDataset" size="large">
              <el-radio-button label="food101">Food101</el-radio-button>
              <el-radio-button label="stl10">STL-10</el-radio-button>
              <el-radio-button label="eurosat">EuroSAT</el-radio-button>
              <el-radio-button label="cifar100">CIFAR-100</el-radio-button>
            </el-radio-group>
          </div>
        </template>

        <el-alert
          type="info"
          show-icon
          :closable="false"
          title="此页先把报告板块结构搭起来：指标表 / 混淆矩阵交互 / Model Zoo。"
        />
      </el-card>

      <transition name="panel-swap" mode="out-in">
        <el-card :key="selectedDataset">
          <template #header>
            <div class="section-title">全量实验数据表（示例）</div>
          </template>

          <el-table :data="summaryRows" style="width: 100%" v-loading="loadingSummary">
            <el-table-column prop="dataset" label="Dataset" width="120" />
            <el-table-column prop="domain" label="领域" width="120" />
            <el-table-column prop="top1_acc" label="Top-1 Acc" />
            <el-table-column prop="notes" label="备注" />
          </el-table>
        </el-card>
      </transition>

      <el-card>
        <template #header>
          <div class="section-title">Model Zoo（待接入）</div>
        </template>
        <el-empty description="后续补充：PyTorch / ONNX / 量化权重下载与速度对比" />
      </el-card>
    </el-space>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { api } from '../lib/api'

const selectedDataset = ref('food101')
const datasetSwitchEl = ref(null)

const cmEl = ref(null)
let cmChart = null

const loadingSummary = ref(false)
const summaryRows = ref([])

function initCm() {
  if (cmEl.value && !cmChart) cmChart = echarts.init(cmEl.value)
}

function disposeCm() {
  cmChart?.dispose();
  cmChart = null
}

async function loadSummary() {
  loadingSummary.value = true
  try {
    const { data } = await api.get('/api/report/summary')
    summaryRows.value = Array.isArray(data?.rows) ? data.rows : []
  } catch (e) {
    ElMessage.error('报告汇总加载失败（请确认后端 API 已启动）')
  } finally {
    loadingSummary.value = false
  }
}

async function loadConfusionMatrix() {
  try {
    initCm()
    const { data } = await api.get(`/api/report/${selectedDataset.value}/confusion-matrix`)
    const cm = data?.confusion_matrix || []

    const size = Array.isArray(cm) ? cm.length : 0
    const heat = []
    for (let i = 0; i < size; i++) {
      const row = cm[i] || []
      for (let j = 0; j < row.length; j++) {
        heat.push([j, i, row[j]])
      }
    }

    cmChart?.setOption({
      tooltip: { position: 'top' },
      grid: { height: '75%', top: 30, left: 60, right: 20 },
      xAxis: { type: 'category', data: Array.from({ length: size }, (_, i) => i), splitArea: { show: true } },
      yAxis: { type: 'category', data: Array.from({ length: size }, (_, i) => i), splitArea: { show: true } },
      visualMap: {
        min: 0,
        max: Math.max(...heat.map((x) => x[2] || 0), 1),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
      },
      series: [
        {
          name: 'cm',
          type: 'heatmap',
          data: heat,
          emphasis: { itemStyle: { borderWidth: 1 } },
        },
      ],
    })
  } catch (e) {
    ElMessage.error('混淆矩阵加载失败（请确认后端 API 已启动）')
  }
}

function handleResize() {
  cmChart?.resize()
  updateSegmentIndicator()
}

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

onMounted(async () => {
  initCm()
  await loadSummary()
  await loadConfusionMatrix()
  window.addEventListener('resize', handleResize)
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  disposeCm()
})

watch(selectedDataset, async () => {
  await loadConfusionMatrix()
  nextTick(() => requestAnimationFrame(updateSegmentIndicator))
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
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
  height: 520px;
  width: 100%;
  border-radius: 14px;
}
</style>
