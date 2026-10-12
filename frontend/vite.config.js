import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import dsv from '@rollup/plugin-dsv';

const API_ORIGIN = `http://localhost:${process.env.API_PORT || 8000}`;

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), dsv()],
  server: {
    proxy: {
      '/predict':            API_ORIGIN,
      '/challenger_predict': API_ORIGIN,
      // '^/model/' not '/model': the Predict Rent page itself lives at /model and must be served by Vite
      '^/model/':            API_ORIGIN,
      '/monitoring':         API_ORIGIN,
      '/health':             API_ORIGIN,
      '/ensemble':           API_ORIGIN,
      '/torch':              API_ORIGIN,
    },
  },
})
// https://github.com/vitejs/vite/discussions/8271
