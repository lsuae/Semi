<template>
  <div class="page">
    <el-space direction="vertical" fill size="large">
      <el-card>
        <template #header>
          <div class="header">
            <div class="title">交互式“少样本”体验</div>
            <div class="subtitle">4-shot Challenge · 数据清理工具</div>
          </div>
        </template>

        <el-alert
          type="warning"
          show-icon
          :closable="false"
          title="此页先搭建功能板块骨架，后续再接入具体算法/数据流。"
        />
      </el-card>

      <el-card>
        <template #header>
          <div class="section">The 4-shot challenge（每类 4 张标签）</div>
        </template>
        <el-empty description="后续展示：4-shot 与无半监督增强的对比案例" />
      </el-card>

      <el-card>
        <template #header>
          <div class="section">数据清理工具（相似度检索 / 一键导出）</div>
        </template>
        <el-empty description="后续接入：基于特征向量的 Cosine Similarity 检索与导出" />
      </el-card>
    </el-space>
  </div>
</template>

<script setup>
</script>

<style scoped>
.header {
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

.section {
  font-weight: 650;
}
</style>
