import { expect, test } from '@playwright/test'

async function horizontalOverflow(page: import('@playwright/test').Page): Promise<number> {
  return page.evaluate(() => {
    const doc = document.documentElement
    return doc.scrollWidth - doc.clientWidth
  })
}

test('셸은 긴 모니터 페이로드에도 가로로 넘치지 않는다', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })

  // 모니터 행 하나로 넘침 조건을 재현한다. 페이로드는 nowrap이라 조상 flex 체인에
  // min-width:0이 없으면 셸 폭 전체를 밀어낸다.
  await page.evaluate(() => {
    const scroll =
      document.querySelector('.sim-monitor__scroll') ?? document.querySelector('.sim-monitor')
    if (scroll === null) {
      throw new Error('sim-monitor not rendered')
    }
    const row = document.createElement('div')
    row.className = 'sim-monitor__row'
    row.innerHTML = `<span class="sim-monitor__dir">INPUT</span><span class="sim-monitor__payload">${'{"k":"'.padEnd(4000, 'x')}"}</span>`
    scroll.appendChild(row)
  })

  expect(await horizontalOverflow(page)).toBe(0)
})
