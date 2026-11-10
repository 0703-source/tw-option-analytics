"""Formal experiments for the ICCAI 2027 SS2 paper.

One-step-ahead next-day close forecasts for 2330 / 2317 / 2454 on TWSE daily data,
2023-01-03 .. 2026-10-30, chronological 80/20 split. Models: random walk, ARIMA (ADF + AIC grid),
GARCH(1,1) (constant-mean), stacked LSTM (5 seeds), equal-weighted ensembles.
Also: GARCH volatility accuracy, Brent vs bisection IV on the live TXO chain, ETL statistics.
Writes results.json and figures (PNG, 300 dpi) next to this file.
"""
import json, os, re, time, warnings, math
import numpy as np
import pandas as pd
from scipy import stats, optimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
START, END = "2023-01-01", "2026-10-31"
STOCKS = {"2330": "TSMC", "2317": "Foxconn", "2454": "MediaTek"}
SEEDS = [0, 1, 2, 3, 4]
etl = {"raw_rows": 0, "roc_dates": 0, "nulls": 0, "exceptions": 0}


def num(s):
    s = str(s).replace(",", "").strip()
    if s in ("--", "-", "", "X0.00"):
        etl["nulls"] += 1
        return np.nan
    return float(s.lstrip("X+"))


def load(code):
    rows = {}
    for fn in sorted(os.listdir(RAW)):
        if not fn.startswith(code + "-"):
            continue
        j = json.load(open(os.path.join(RAW, fn), encoding="utf-8-sig"))
        for r in j.get("data", []):
            etl["raw_rows"] += 1
            try:
                y, m, d = map(int, r[0].strip().split("/")); etl["roc_dates"] += 1
                rows[f"{y + 1911:04d}-{m:02d}-{d:02d}"] = num(r[6])
            except Exception:
                etl["exceptions"] += 1
    s = pd.Series(rows).sort_index()
    return s[(s.index >= START) & (s.index <= END)].dropna()


def metrics(pred, act, prev):
    e = pred - act
    rmse = float(np.sqrt(np.mean(e ** 2))); mae = float(np.mean(np.abs(e))); mape = float(np.mean(np.abs(e / act)) * 100)
    dpa, dpp = np.sign(act - prev), np.sign(pred - prev)
    m = dpa != 0
    da = float(np.mean(dpa[m] == dpp[m]) * 100) if not np.allclose(pred, prev) else None
    return {"rmse": rmse, "mae": mae, "mape": mape, "da": da, "n_dir": int(m.sum()), "hits": int((dpa[m] == dpp[m]).sum())}


def dm_test(e1, e2, h=1):
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction, squared-error loss. e1: model, e2: benchmark.
    Negative statistic = model has lower loss. Returns (stat, two-sided p)."""
    d = e1 ** 2 - e2 ** 2; n = len(d); dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0 / n
    stat = dbar / math.sqrt(var)
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def fit_arima(P, split):
    tr = P[:split]
    d, x = 0, tr.copy()
    adf = []
    while True:
        st, pv = adfuller(x, autolag="AIC")[:2]; adf.append({"d": d, "stat": float(st), "p": float(pv)})
        if pv < 0.05 or d == 2:
            break
        x = np.diff(x); d += 1
    best, grid = None, []
    for p in range(6):
        for q in range(6):
            try:
                r = ARIMA(tr, order=(p, d, q)).fit()
                grid.append(((p, d, q), float(r.aic)))
                if best is None or r.aic < best[1]:
                    best = ((p, d, q), r.aic, r)
            except Exception:
                pass
    order, aic, res = best
    full = res.apply(P)  # same parameters, filtered through the test period (no refit, no look-ahead)
    pred = full.get_prediction(start=split, end=len(P) - 1).predicted_mean
    nxt = float(full.forecast(1)[0])
    return np.asarray(pred), nxt, {"order": order, "aic": float(aic), "adf": adf,
                                   "runner_up": sorted(grid, key=lambda g: g[1])[1:3]}


def fit_garch(P, split):
    r = 100 * np.diff(np.log(P))  # r[i] = return from day i to i+1
    am = arch_model(r[:split - 1], mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    fixed = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fix(res.params.values)
    mu = res.params["mu"]
    pred = P[split - 1:-1] * np.exp(mu / 100)
    sig = np.asarray(fixed.conditional_volatility)  # sig[i]: sd of r[i] given info to day i
    return pred, float(P[-1] * np.exp(mu / 100)), {k: float(v) for k, v in res.params.items()}, r, sig


def fit_lstm(P, split, seed, W=60, epochs=100):
    """Stacked LSTM in Keras 3 (JAX backend); same network/preprocessing as the deployed platform."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import keras
    keras.utils.set_random_seed(seed)
    win = lambda e: P[e - W:e] / P[e - 1] - 1
    X0 = np.array([win(i) for i in range(W, split)]); Y0 = np.array([P[i] / P[i - 1] - 1 for i in range(W, split)])
    lo, hi = min(X0.min(), Y0.min()), max(X0.max(), Y0.max())  # MinMax fitted on training windows only
    sc = lambda v: (v - lo) / (hi - lo); un = lambda v: v * (hi - lo) + lo
    m = keras.Sequential([keras.Input((W, 1)), keras.layers.LSTM(50, return_sequences=True), keras.layers.Dropout(0.2),
                          keras.layers.LSTM(50), keras.layers.Dropout(0.2), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    t0 = time.time()
    h = m.fit(sc(X0)[..., None], sc(Y0), epochs=epochs, batch_size=32, validation_split=0.1, shuffle=True, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    ends = list(range(split, len(P) + 1))
    Xt = np.array([sc(win(e)) for e in ends])[..., None]
    prices = np.array([P[e - 1] for e in ends]) * (un(m.predict(Xt, verbose=0).ravel()) + 1)
    return prices[:-1], float(prices[-1]), {"epochs": len(h.history["loss"]), "best_val": float(min(h.history["val_loss"])), "sec": time.time() - t0}


def bs(S, K, T, r, s, cp):
    d1 = (math.log(S / K) + (r + s * s / 2) * T) / (s * math.sqrt(T)); d2 = d1 - s * math.sqrt(T)
    if cp == "C":
        return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def iv_study():
    opt = json.load(open(os.path.join(HERE, "snapshot", "opt.json"), encoding="utf-8-sig"))
    tx = [json.load(open(os.path.join(os.path.join(HERE, "snapshot"), f), encoding="utf-8-sig"))
          for f in sorted(os.listdir(os.path.join(HERE, "snapshot"))) if f.startswith("TAIEX-")]
    tx = [r for j in tx for r in j["data"]]
    rows = [o for o in opt if o["Contract"] == "TXO" and o["TradingSession"] == "一般" and re.fullmatch(r"\d{6}", o["ContractMonth(Week)"])]
    exps = sorted({o["ContractMonth(Week)"] for o in rows})[:3]
    trade = pd.Timestamp(rows[0]["Date"])
    roc = f"{trade.year - 1911}/{trade.month:02d}/{trade.day:02d}"
    hit = [t for t in tx if t[0].strip() == roc]  # spot on the same trading day as the option settlement prices
    S = float(hit[0][4].replace(",", "")); sdate = hit[0][0]
    txo_all = [o for o in opt if o["Contract"] == "TXO"]
    etl["txo_rows"] = len(txo_all)
    etl["txo_null_fields"] = sum(1 for o in txo_all for v in o.values() if str(v).strip() in ("-", "--", ""))
    def expiry(ym):
        d = pd.Timestamp(int(ym[:4]), int(ym[4:]), 1)
        weds = [d + pd.Timedelta(days=i) for i in range(31) if (d + pd.Timedelta(days=i)).month == d.month and (d + pd.Timedelta(days=i)).weekday() == 2]
        return weds[2]
    r = 0.0175
    res = {"brent_it": [], "bis_it": [], "brent_us": [], "bis_us": [], "solved": 0, "no_root": 0, "skipped": 0}
    for o in rows:
        if o["ContractMonth(Week)"] not in exps:
            continue
        px = o["SettlementPrice"]
        if px in ("", "-", "--") or float(px) <= 0:
            res["skipped"] += 1; continue
        px = float(px); K = float(o["StrikePrice"]); cp = "C" if o["CallPut"] == "買權" else "P"
        T = (expiry(o["ContractMonth(Week)"]) - trade).days / 365
        f = lambda s: bs(S, K, T, r, s, cp) - px
        if f(0.001) * f(5.0) > 0:
            res["no_root"] += 1; continue
        t0 = time.perf_counter(); x, rb = optimize.brentq(f, 0.001, 5.0, xtol=1e-4, full_output=True); t1 = time.perf_counter()
        y, rs = optimize.bisect(f, 0.001, 5.0, xtol=1e-4, full_output=True); t2 = time.perf_counter()
        res.setdefault("points", []).append([o["ContractMonth(Week)"], K, cp, x, int(o["OpenInterest"] or 0)])
        res["solved"] += 1; res["brent_it"].append(rb.iterations); res["bis_it"].append(rs.iterations)
        res["brent_us"].append((t1 - t0) * 1e6); res["bis_us"].append((t2 - t1) * 1e6)
    summ = {k: v for k, v in res.items() if not isinstance(v, list)}
    for k in ("brent_it", "bis_it", "brent_us", "bis_us"):
        summ[k + "_mean"] = float(np.mean(res[k])); summ[k + "_max"] = float(np.max(res[k]))
    summ["points"] = res["points"]
    summ.update({"S": S, "S_date": sdate, "trade_date": str(trade.date()), "expiries": exps,
                 "txo_rows_all_sessions": sum(1 for o in opt if o["Contract"] == "TXO")})
    return summ


def main():
    results = {"period": [START, END], "stocks": {}}
    preds_store = {}
    for code, name in STOCKS.items():
        s = load(code); P = s.values.astype(float); dates = list(s.index); n = len(P); split = int(n * 0.8)
        act, prev = P[split:], P[split - 1:-1]
        t0 = time.time(); ar, ar_next, ar_info = fit_arima(P, split); t_ar = time.time() - t0
        t0 = time.time(); ga, ga_next, ga_par, r, sig = fit_garch(P, split); t_ga = time.time() - t0
        lstm_runs = []
        for sd in SEEDS:
            ls, ls_next, ls_info = fit_lstm(P, split, sd)
            assert len(ls) == len(act)
            lstm_runs.append((ls, ls_next, ls_info)); print(code, "seed", sd, ls_info, flush=True)
        rw_m = metrics(prev, act, prev); e_rw = prev - act
        row = {"RW": {**rw_m, "u": 1.0}}
        def add(key, p):
            m = metrics(p, act, prev); m["u"] = m["rmse"] / rw_m["rmse"]; m["dm"], m["dm_p"] = dm_test(p - act, e_rw); return m
        row["ARIMA"] = add("ARIMA", ar); row["GARCH"] = add("GARCH", ga)
        per = {"LSTM": [], "ENS3": [], "ENS2": []}
        for ls, _, _ in lstm_runs:
            per["LSTM"].append(add("LSTM", ls)); per["ENS3"].append(add("ENS3", (ar + ga + ls) / 3)); per["ENS2"].append(add("ENS2", (ar + ls) / 2))
        for k, lst in per.items():
            agg = {}
            for f in ("rmse", "mae", "mape", "da", "u", "dm"):
                v = np.array([x[f] for x in lst], float); agg[f] = float(v.mean()); agg[f + "_sd"] = float(v.std(ddof=1))
            agg["dm_p_max"] = float(max(x["dm_p"] for x in lst)); agg["dm_p_min"] = float(min(x["dm_p"] for x in lst))
            agg["hits"] = [x["hits"] for x in lst]; agg["n_dir"] = lst[0]["n_dir"]
            row[k] = agg
        # GARCH volatility accuracy on test: variance proxy r^2 vs sigma^2 (GARCH) and HV20^2 (rolling, info to t-1)
        rt = r[split - 1:]; sg = sig[split - 1:]
        hv = pd.Series(r).rolling(20).std(ddof=1).shift(1).values[split - 1:]
        qlike = lambda s2: float(np.mean(np.log(s2) + rt ** 2 / s2))
        vol = {"mse_garch": float(np.mean((rt ** 2 - sg ** 2) ** 2)), "mse_hv20": float(np.mean((rt ** 2 - hv ** 2) ** 2)),
               "qlike_garch": qlike(sg ** 2), "qlike_hv20": qlike(hv ** 2),
               "next_ann_vol": float(np.sqrt(ga_par["omega"] + ga_par["alpha[1]"] * (r[-1] - ga_par["mu"]) ** 2 + ga_par["beta[1]"] * sig[-1] ** 2) * np.sqrt(252) / 100)}
        lsn = [x[1] for x in lstm_runs]
        results["stocks"][code] = {
            "name": name, "n": n, "split": split, "train": [dates[0], dates[split - 1]], "test": [dates[split], dates[-1]],
            "n_test": n - split, "last_close": float(P[-1]), "price_train_minmax": [float(P[:split].min()), float(P[:split].max())],
            "price_test_minmax": [float(act.min()), float(act.max())],
            "arima": ar_info, "garch": ga_par, "vol": vol, "metrics": row,
            "lstm_runs": [x[2] for x in lstm_runs], "sec": {"arima": t_ar, "garch": t_ga, "lstm_mean": float(np.mean([x[2]["sec"] for x in lstm_runs]))},
            "next": {"ARIMA": ar_next, "GARCH": ga_next, "LSTM": float(np.mean(lsn))},
        }
        preds_store[code] = {"dates": dates[split:], "act": act.tolist(), "rw": prev.tolist(), "arima": ar.tolist(), "garch": ga.tolist(),
                             "lstm0": lstm_runs[0][0].tolist(), "ens3_0": ((ar + ga + lstm_runs[0][0]) / 3).tolist(),
                             "train_dates": dates[:split], "train": P[:split].tolist()}
        print(code, json.dumps({k: {f: (round(v, 4) if isinstance(v, float) else v) for f, v in m.items() if f in ("rmse", "mae", "mape", "da", "u", "dm_p", "dm_p_max")} for k, m in row.items()}), flush=True)
    results["iv"] = iv_study()
    results["etl"] = dict(etl)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    json.dump(preds_store, open(os.path.join(OUT, "preds.json"), "w"))
    print("ETL", etl); print("IV", {k: v for k, v in results["iv"].items() if k != "points"})


if __name__ == "__main__":
    main()
