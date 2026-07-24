import { expect, test } from '@playwright/test'

test('sequence question updates chat and order panel', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('SP0035 Level 01')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('zone-A-2')).toBeVisible()

  await page.getByPlaceholder('도장 관련 질문을 입력하세요').fill('다음에 어떤 구역을 칠해야 해?')
  await page.getByRole('button', { name: '전송' }).click()

  await expect(page.getByText(/다음 도장 구역: A-2/)).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'next')
  await expect(page.getByTestId('zone-A-1')).toHaveAttribute('data-status', 'completed')
})
