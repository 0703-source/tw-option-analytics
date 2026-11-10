"""Fetch the latest TWSE / TAIFEX data and build site/index.html.

Raw exchange responses are cached in data/raw/. Each run re-downloads the
current and previous month (plus today's TXO chain) and reuses older months.
"""
import base64
import datetime
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
TZ = datetime.timezone(datetime.timedelta(hours=8))
NOW = datetime.datetime.now(TZ)
STOCKS = {"2330": ("台積電", "TSMC"), "2317": ("鴻海", "Foxconn"), "2454": ("聯發科", "MediaTek")}
START = (2024, 1)
errors = []


def get(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tw-option-analytics research demo)", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccups: back off and retry
            last = e
            time.sleep(5 * (i + 1))
    raise last


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def shift(ym, k):
    y, m = ym
    m += k
    while m < 1:
        y, m = y - 1, m + 12
    return (y, m)


cur = (NOW.year, NOW.month)
refresh = {f"{cur[0]}{cur[1]:02d}", "%d%02d" % shift(cur, -1)}


def fetch_month(url, path, ym):
    if os.path.exists(path) and ym not in refresh:
        return
    try:
        b = get(url)
        j = json.loads(b.decode("utf-8"))
        if j.get("stat") == "OK":
            with open(path, "wb") as f:
                f.write(b)
        elif not os.path.exists(path) and ym != f"{cur[0]}{cur[1]:02d}":
            errors.append(f"{url}: {j.get('stat')}")
    except Exception as e:
        errors.append(f"{url}: {e}")
    time.sleep(2.5)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for code in STOCKS:
        for ym in months(START, cur):
            fetch_month(f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={ym}01&stockNo={code}", os.path.join(RAW, f"{code}-{ym}01.json"), ym)
    for ym in months(shift(cur, -5), cur):
        fetch_month(f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={ym}01", os.path.join(RAW, f"TAIEX-{ym}01.json"), ym)
    try:
        b = get("https://openapi.taifex.com.tw/v1/DailyMarketReportOpt")
        rows = json.loads(b.decode("utf-8-sig"))
        if sum(1 for r in rows if r.get("Contract") == "TXO") > 500:
            with open(os.path.join(RAW, "opt.json"), "wb") as f:
                f.write(b)
        else:
            errors.append("TAIFEX: too few TXO rows, kept previous chain")
    except Exception as e:
        errors.append(f"TAIFEX: {e}")


def load(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8-sig"))


def build():
    files = sorted(os.listdir(RAW))
    stocks = {}
    for code, (name, en) in STOCKS.items():
        seen, rows = set(), []
        for fn in files:
            if fn.startswith(code + "-"):
                j = load(os.path.join(RAW, fn))
                for r in j.get("data", []) if j.get("stat") == "OK" else []:
                    if r[0] not in seen:
                        seen.add(r[0])
                        rows.append(r[:9])
        stocks[code] = {"name": name, "en": en, "rows": rows}
    taiex = []
    for fn in files:
        if fn.startswith("TAIEX-"):
            j = load(os.path.join(RAW, fn))
            if j.get("stat") == "OK":
                taiex.extend(j["data"])
    opt = load(os.path.join(RAW, "opt.json"))
    t = [o for o in opt if o["Contract"] == "TXO" and o["TradingSession"] == "一般" and re.fullmatch(r"\d{6}", o["ContractMonth(Week)"])]
    exps = sorted({o["ContractMonth(Week)"] for o in t})[:3]
    t = [o for o in t if o["ContractMonth(Week)"] in exps]
    txo = [[o["ContractMonth(Week)"], o["StrikePrice"], o["CallPut"], o["Close"], o["Volume"], o["SettlementPrice"], o["OpenInterest"], o["BestBid"], o["BestAsk"]] for o in t]
    d = t[0]["Date"]
    atm = max((o for o in t if o["ContractMonth(Week)"] == exps[0] and o["CallPut"] == "買權"), key=lambda o: int(o["Volume"] or 0))
    line = f"交易日期,契約,到期月份(週別),履約價,買賣權,結算價,未沖銷契約數\r\n{d[:4]}/{d[4:6]}/{d[6:]},TXO,{exps[0]},{atm['StrikePrice']},{atm['CallPut']},{atm['SettlementPrice']},{atm['OpenInterest']}"
    meta_path = os.path.join(ROOT, "data", "meta.json")
    meta = load(meta_path) if os.path.exists(meta_path) else {}
    if fetched and not errors:
        meta["captured"] = NOW.strftime("%Y-%m-%d %H:%M") + " (Taipei)"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    data = {"captured": meta.get("captured", "unknown"), "stocks": stocks, "taiex": taiex, "txo": {"date": d, "rows": txo}, "big5": base64.b64encode(line.encode("big5")).decode()}
    with open(os.path.join(ROOT, "template.html"), encoding="utf-8") as f:
        tpl = f.read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
    with open(os.path.join(ROOT, "site", "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"rows: " + ", ".join(f"{c}={len(s['rows'])}" for c, s in stocks.items()) + f"; TAIEX={len(taiex)}; TXO={len(txo)} ({','.join(exps)})")
    print(f"latest stock date: {stocks['2330']['rows'][-1][0]}  TXO date: {d}  captured: {data['captured']}")


if __name__ == "__main__":
    fetched = "--build-only" not in sys.argv
    if fetched:
        fetch()
    for e in errors:
        print("WARN", e)
    build()
