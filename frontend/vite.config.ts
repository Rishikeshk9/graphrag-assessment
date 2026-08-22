import { defineConfig } from 'vitest/config'
import { fileURLToPath, URL } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const fromHere = (path: string) => fileURLToPath(new URL(path, import.meta.url))

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fromHere('./src'),
      // Test files live outside this package, so their imports must resolve
      // against this package's node_modules rather than their own directory.
      'react/jsx-dev-runtime': fromHere('./node_modules/react/jsx-dev-runtime'),
      'react/jsx-runtime': fromHere('./node_modules/react/jsx-runtime'),
      react: fromHere('./node_modules/react'),
      'react-dom': fromHere('./node_modules/react-dom'),
      '@testing-library/react': fromHere('./node_modules/@testing-library/react'),
      '@testing-library/jest-dom': fromHere('./node_modules/@testing-library/jest-dom'),
      cytoscape: fromHere('./node_modules/cytoscape'),
    },
  },
  // `fs.allow` lets Vitest load the repository-level /tests directory.
  server: { host: '127.0.0.1', port: 5173, fs: { allow: ['..'] } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['../tests/frontend/**/*.test.{ts,tsx}'],
  },
})
