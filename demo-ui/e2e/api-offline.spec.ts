import { test, expect } from '@playwright/test'

test('shows offline status when API is unreachable', async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    await route.abort('failed')
  })
  await page.goto('/')
  await expect(page.getByTestId('connection-state')).toHaveText('오프라인', { timeout: 10_000 })
  await expect(page.getByRole('button', { name: '재연결' })).toBeVisible()
})
