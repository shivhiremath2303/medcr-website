import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    watch: {
      // The repo root also holds a ~48k-file Python .venv and a backend tree.
      // Watching them pegs the CPU and stalls the dev server.
      ignored: [
        '**/.venv/**',
        '**/backend/**',
        '**/dist/**',
        '**/__pycache__/**',
        '**/*.py[cod]'
      ]
    }
  }
})
