import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import dsv from '@rollup/plugin-dsv';


// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), dsv()],
})
// https://github.com/vitejs/vite/discussions/8271
