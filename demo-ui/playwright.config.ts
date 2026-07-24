import { defineConfig, devices } from '@playwright/test'

const BACKEND_HOST = '127.0.0.1'
const BACKEND_PORT = 8120
const WEB_PORT = 5173

export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: `http://${BACKEND_HOST}:${WEB_PORT}`,
    trace: 'off',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: `python3 -m trainer.demo_api.app --host ${BACKEND_HOST} --port ${BACKEND_PORT}`,
      cwd: '..',
      url: `http://${BACKEND_HOST}:${BACKEND_PORT}/api/health`,
      timeout: 30_000,
      reuseExistingServer: true,
      env: {
        DEMO_UI_ORIGINS: `http://${BACKEND_HOST}:${WEB_PORT},http://localhost:${WEB_PORT}`,
      },
    },
    {
      command: `npm run dev:web -- --port ${WEB_PORT} --strictPort`,
      url: `http://${BACKEND_HOST}:${WEB_PORT}`,
      timeout: 30_000,
      reuseExistingServer: true,
      env: {
        VITE_API_BASE_URL: `http://${BACKEND_HOST}:${BACKEND_PORT}`,
      },
    },
  ],
})
