// 교정 시연 녹화·캡처 — KPI 시험 방법 문서(docs/year2, docs/year3)의 그림과 영상을 다시 만든다.
//
//   (8130이 실데이터로 떠 있어야 한다. 저장소 루트에서, demo-ui 의존성을 쓴다)
//   node trainer/pose_lab/record_demo.mjs issued     docs/year2/img <영상 폴더>   (화면 기본: 교정 알고리즘 안내 전부)
//   node trainer/pose_lab/record_demo.mjs six_values docs/year3/img <영상 폴더>   (숨김: 관절 값 6개 이하)
//
// 남기는 것:
//   <img>/demo_<mode>_1_입력.png, _2_안내.png, _3_결과.png  — 예시 0(오른손·위아래 대표값)의 세 단계
//   <img>/main_*.png (issued만)                              — 도장 동작 비교 화면의 안내 문구·예상 결과·3D
//   <영상>/교정시연_<year>.webm                               — 예시 0·3(오른손, 머리 대표값)을 차례로 재생
import { mkdirSync, renameSync, readdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = (await import(resolve(here, "../../demo-ui/node_modules/@playwright/test/index.js"))).default;
const { chromium } = pkg;

const [mode, imgDir, videoDir] = process.argv.slice(2);
if (!mode || !imgDir || !videoDir) {
  console.error("usage: record_demo.mjs <issued|six_values|delivered> <img dir> <video dir>");
  process.exit(2);
}
const BASE = process.env.POSE_LAB_URL ?? "http://127.0.0.1:8130";
const VIEWPORT = { width: 1600, height: 1000 };
mkdirSync(imgDir, { recursive: true });
mkdirSync(videoDir, { recursive: true });

const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"] });
const errors = [];

async function waitStage(page, stage) {
  await page.waitForFunction((s) => window.__demo?.stage >= s, stage, { timeout: 180000 });
}

// 1) 세 단계 캡처
{
  const page = await browser.newPage({ viewport: VIEWPORT });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${BASE}/demo.html?mode=${mode}&scenario=0`, { waitUntil: "domcontentloaded" });
  await waitStage(page, 1);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(imgDir, `demo_${mode}_1_입력.png`), fullPage: true });
  // 안내를 모두 보인 뒤, 결과로 넘어가기 전(2단계 끝)에 찍는다.
  await page.waitForFunction(
    () => window.__demo?.stage === 2 && window.__demo.data && window.__demo.cuesShown === window.__demo.data.cues.length,
    null, { timeout: 180000 },
  );
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(imgDir, `demo_${mode}_2_안내.png`), fullPage: true });
  await page.waitForFunction(() => window.__demo?.finished, null, { timeout: 60000 });
  await page.screenshot({ path: join(imgDir, `demo_${mode}_3_결과.png`), fullPage: true });
  await page.close();
}

// 2) 영상 — 예시 0(오른손 대표값)과 3(머리 대표값)
{
  const context = await browser.newContext({ viewport: VIEWPORT, recordVideo: { dir: videoDir, size: VIEWPORT } });
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  // 오른손 대표값은 비스듬히, 머리(자세 높이기) 대표값은 몸이 펴지는 것이 보이게 옆에서.
  for (const [scenario, view] of [[0, "oblique"], [3, "side"]]) {
    await page.goto(`${BASE}/demo.html?mode=${mode}&scenario=${scenario}&view=${view}`, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__demo?.finished, null, { timeout: 180000 });
    await page.waitForTimeout(3000);
  }
  const video = page.video();
  await context.close();
  const raw = await video.path();
  renameSync(raw, join(videoDir, `교정시연_${mode}.webm`));
}

// 3) 도장 동작 비교 화면(2차년도 시연 절차) — 안내 문구와 예상 결과, 3D의 '안내 적용 후' 층
if (mode === "issued") {
  const page = await browser.newPage({ viewport: { width: 1875, height: 1000 } });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#queryPass")?.options.length > 3, null, { timeout: 200000 });
  await page.selectOption("#queryPass", { index: 3 });
  await page.waitForTimeout(1500);
  await page.waitForFunction(() => !document.querySelector("#run")?.disabled, null, { timeout: 60000 });
  await page.click("#run");
  await page.waitForFunction(() => !document.querySelector("#heroBody").hidden, null, { timeout: 120000 });
  await page.waitForTimeout(1500);
  await page.evaluate(() => { document.querySelector(".improvement-detail").open = true; });
  await page.screenshot({ path: join(imgDir, "main_1_비교결과.png") });
  await page.locator("#improvementSection").screenshot({ path: join(imgDir, "main_2_예상결과.png") });
  const corrected = page.locator("#view3dLayerCorrected");
  if (await corrected.count()) {
    await corrected.check();
    await page.waitForTimeout(800);
    await page.locator("#view3d").screenshot({ path: join(imgDir, "main_3_3D_안내적용후.png") });
  }
  await page.close();
}

await browser.close();
for (const name of readdirSync(videoDir)) console.log("video:", join(videoDir, name));
console.log("오류:", errors.length ? errors.join(" | ") : "없음");
