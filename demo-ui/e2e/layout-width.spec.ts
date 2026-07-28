import { expect, test } from '@playwright/test'

async function horizontalOverflow(page: import('@playwright/test').Page): Promise<number> {
  return page.evaluate(() => {
    const doc = document.documentElement
    return doc.scrollWidth - doc.clientWidth
  })
}

test('셸은 긴 모니터 페이로드에도 가로로 넘치지 않는다', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })

  await page.getByRole('button', { name: '세션 시작' }).click()
  await expect(page.locator('.sim-monitor__row').first()).toBeVisible()

  expect(await horizontalOverflow(page)).toBe(0)
})
