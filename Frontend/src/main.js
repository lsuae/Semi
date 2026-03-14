import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

// Custom theme overrides should be loaded AFTER component library CSS.
import './style.css'

const app = createApp(App)

app.use(ElementPlus)
app.use(router)

app.mount('#app')