"""Download the TWSE STOCK_DAY months used in the study (2023-01 .. 2026-11) into raw/ if they are missing."""
import json, os, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw"); os.makedirs(RAW, exist_ok=True)
months = [f"{y}{m:02d}" for y in range(2023, 2027) for m in range(1, 13) if f"{y}{m:02d}" <= "202611"]
for code in ("2330", "2317", "2454"):
    for ym in months:
        path = os.path.join(RAW, f"{code}-{ym}01.json")
        if os.path.exists(path):
            continue
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={ym}01&stockNo={code}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tw-option-analytics research)"})
        b = urllib.request.urlopen(req, timeout=60).read()
        assert json.loads(b.decode())["stat"] == "OK", url
        open(path, "wb").write(b)
        time.sleep(2.5)
print(len(os.listdir(RAW)), "files")
