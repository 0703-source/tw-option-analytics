"""H3: UI-thread responsiveness on the live deployment.
Isolated  = ARIMA/GARCH run in the page's Web Worker (as deployed); longest requestAnimationFrame gap recorded by the page.
Blocking  = the identical worker code executed on the main thread (ablation); longest frame gap measured the same way.
10 fresh page loads x 3 stocks each. Writes out/h3.json."""
import json, os, statistics
from playwright.sync_api import sync_playwright

URL = "https://0703-source.github.io/tw-option-analytics/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BLOCK_JS = """async (code) => {
  const src = document.getElementById('worker-src').textContent;
  const {P, split} = series(code);
  let done = null; const self = {postMessage: m => { if (m.type === 'done') done = m; }};
  new Function('self', src + '\\nreturn self;')(self);
  let gap = 0, last = 0, on = true;
  const f = ts => { if (!on) return; if (last) gap = Math.max(gap, ts - last); last = ts; requestAnimationFrame(f); };
  requestAnimationFrame(f);
  await new Promise(r => setTimeout(r, 100));
  const t0 = performance.now(); self.onmessage({data: {P, split, id: 1}}); const ms = performance.now() - t0;
  await new Promise(r => setTimeout(r, 100)); on = false;
  return {ms, gap, ok: !!done};
}"""
rows = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    for rep in range(10):
        page = b.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL + "?r=%d" % rep, wait_until="networkidle")
        page.click('button[data-tab="forecast"]')
        for i, code in enumerate(["2330", "2317", "2454"]):
            if i:
                page.locator("#fc-stock button").nth(i).click()
            page.wait_for_function(f"FC.res['{code}'] && FC.res['{code}'].statGap !== undefined", timeout=120000)
            iso = page.evaluate(f"({{ms: FC.res['{code}'].statMs, gap: FC.res['{code}'].statGap}})")
            blk = page.evaluate(BLOCK_JS, code)
            rows.append({"rep": rep, "code": code, "iso_ms": iso["ms"], "iso_gap": iso["gap"], "blk_ms": blk["ms"], "blk_gap": blk["gap"], "blk_ok": blk["ok"]})
            print(rows[-1], flush=True)
        page.close()
    b.close()

def summ(k):
    v = sorted(r[k] for r in rows)
    return {"median": statistics.median(v), "p95": v[int(round(0.95 * (len(v) - 1)))], "max": v[-1], "n": len(v)}
out = {"rows": rows, **{k: summ(k) for k in ("iso_ms", "iso_gap", "blk_ms", "blk_gap")}}
json.dump(out, open(os.path.join(OUT, "h3.json"), "w"), indent=1)
print({k: out[k] for k in ("iso_ms", "iso_gap", "blk_ms", "blk_gap")})
