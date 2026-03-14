<template>
  <el-container class="app-shell">
    <el-header class="nav-header">
      <div class="nav-inner">
        <div class="brand">
          <div class="title">多领域细粒度半监督学习可视化系统</div>
        </div>
      </div>
    </el-header>

    <el-container class="app-body">
      <el-aside class="side">
        <el-menu class="side-menu" :default-active="activePath" router>
          <el-menu-item index="/">
            <el-icon><Grid /></el-icon>
            <span>多领域演示</span>
          </el-menu-item>
          <el-menu-item index="/visualization">
            <el-icon><DataAnalysis /></el-icon>
            <span>算法可视化</span>
          </el-menu-item>
          <el-menu-item index="/interactive">
            <el-icon><MagicStick /></el-icon>
            <span>少样本体验</span>
          </el-menu-item>
          <el-menu-item index="/benchmark">
            <el-icon><Document /></el-icon>
            <span>复现报告</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-main class="content">
        <div class="content-inner">
          <router-view />
        </div>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { DataAnalysis, Document, Grid, MagicStick } from '@element-plus/icons-vue'

const route = useRoute()
const activePath = computed(() => route.path)
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
}

.nav-header {
  display: flex;
  align-items: center;
  border-bottom: 1px solid rgba(195, 177, 225, 0.14);
  background: rgba(255, 255, 255, 0.60);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  position: sticky;
  top: 0;
  z-index: 10;
  box-shadow: 0 8px 28px rgba(165, 180, 252, 0.08);
  border-bottom-left-radius: 24px;
  border-bottom-right-radius: 24px;
  overflow: hidden;
}

.nav-inner {
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 24px 28px 16px 8px;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}

.title {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: 0.2px;
  color: #1E293B;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}


.app-body {
  min-height: calc(100vh - 60px);
  display: block !important;
  --side-collapsed: 72px;
  --side-expanded: 224px;
  --side-gap: 12px;
  --side-pad: calc(var(--side-collapsed) + var(--side-gap));
}

.side {
  width: var(--side-collapsed) !important;
  transition: width 220ms ease;
  border-right: none;
  background: transparent;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
  padding: 10px 8px;
  border-radius: 24px;
  overflow: hidden;
  position: fixed;
  left: var(--side-gap);
  top: 92px;
  bottom: var(--side-gap);
  z-index: 20;
}

.side:hover {
  width: var(--side-expanded) !important;
  background: rgba(255, 255, 255, 0.70);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: 0 20px 80px rgba(0, 0, 0, 0.05);
}

.side-menu {
  border-right: none;
  background: transparent;
  --el-menu-bg-color: transparent;
  --el-menu-hover-bg-color: rgba(195, 177, 225, 0.18);
  --el-menu-active-color: #C3B1E1;
  border-radius: 12px;
  width: 100%;
}

.side-menu :deep(.el-menu-item) {
  border-radius: 18px;
  margin: 4px 0;
  display: flex;
  align-items: center;
  gap: 14px;
  min-height: 56px;
  padding: 0 12px;
}

.side-menu :deep(.el-menu-item .el-icon) {
  flex: 0 0 auto;
  font-size: 22px;
  color: #1E293B;
}

.side-menu :deep(.el-menu-item span) {
  opacity: 0;
  transform: translateX(-6px);
  transition: opacity 180ms ease, transform 180ms ease;
  white-space: nowrap;
  font-size: 18px;
  color: #1E293B;
}

.side:hover .side-menu :deep(.el-menu-item span) {
  opacity: 1;
  transform: translateX(0);
}

.side-menu :deep(.el-menu-item.is-active) {
  background: rgba(244, 114, 182, 0.12);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
}

.side-menu :deep(.el-menu-item.is-active span) {
  font-weight: 600;
  letter-spacing: 1px;
}

.content {
  padding: 0;
  display: flex !important;
  justify-content: center !important;
  width: 100%;
  /* Center within the remaining viewport area to the right of the fixed sidebar.
     Only reserve space on the left (for sidebar), avoid a “ghost sidebar” on the right. */
  padding-left: var(--side-pad) !important;
  padding-right: var(--side-gap) !important;
  transition: padding-left 220ms ease, padding-right 220ms ease;
  min-width: 0;
}

.side:hover ~ .content {
  padding-left: calc(var(--side-expanded) + var(--side-gap)) !important;
  padding-right: var(--side-gap) !important;
}

.content-inner {
  width: 100%;
  max-width: none;
  margin: 0;
}

@media (max-width: 960px) {
  .content {
    padding-left: 0;
    padding-right: 0;
  }
  .side {
    position: static;
    width: var(--side-collapsed) !important;
    top: auto;
    bottom: auto;
    left: auto;
  }
}
</style>