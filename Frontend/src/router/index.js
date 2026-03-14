import { createRouter, createWebHashHistory } from 'vue-router'

import DemoPage from '../pages/DemoPage.vue'
import VisualizationPage from '../pages/VisualizationPage.vue'
import InteractivePage from '../pages/InteractivePage.vue'
import BenchmarkPage from '../pages/BenchmarkPage.vue'

const router = createRouter({
    history: createWebHashHistory(),
    routes: [
        { path: '/', name: 'demo', component: DemoPage },
        { path: '/visualization', name: 'visualization', component: VisualizationPage },
        { path: '/interactive', name: 'interactive', component: InteractivePage },
        { path: '/benchmark', name: 'benchmark', component: BenchmarkPage },
    ],
})

export default router
