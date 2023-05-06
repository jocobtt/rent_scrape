import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import Csv from 'vite-plugin-csv';


// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), Csv()],
})
