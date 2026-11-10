"""Put-call-parity implied forward per TXO expiry (strikes within +/-2% of spot), to quantify the spot-vs-forward basis
behind the call/put IV gap in Fig. 4. Also re-solves ATM IV with Black-76 on the implied forward."""
import json, math, os, re
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
R = json.load(open(os.path.join(OUT, "results.json")))
S, r = R["iv"]["S"], 0.0175
opt = json.load(open(os.path.join(HERE, "snapshot", "opt.json"), encoding="utf-8-sig"))
rows = [o for o in opt if o["Contract"] == "TXO" and o["TradingSession"] == "一般" and o["ContractMonth(Week)"] in R["iv"]["expiries"]]
trade = pd.Timestamp(R["iv"]["trade_date"])
def expiry(ym):
    d = pd.Timestamp(int(ym[:4]), int(ym[4:]), 1)
    return [d + pd.Timedelta(days=i) for i in range(31) if (d + pd.Timedelta(days=i)).weekday() == 2][2]
def b76(F, K, T, s, cp):
    d1 = (math.log(F / K) + s * s * T / 2) / (s * math.sqrt(T)); d2 = d1 - s * math.sqrt(T); df = math.exp(-r * T)
    return df * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)) if cp == "C" else df * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))
out = {}
for e in R["iv"]["expiries"]:
    T = (expiry(e) - trade).days / 365
    px = {}
    for o in rows:
        if o["ContractMonth(Week)"] == e and o["SettlementPrice"] not in ("", "-", "--"):
            px.setdefault(float(o["StrikePrice"]), {})["C" if o["CallPut"] == "買權" else "P"] = float(o["SettlementPrice"])
    Fs = sorted(K + math.exp(r * T) * (v["C"] - v["P"]) for K, v in px.items() if "C" in v and "P" in v and abs(K / S - 1) < 0.02)
    F = Fs[len(Fs) // 2]
    Katm = min((K for K, v in px.items() if "C" in v and "P" in v), key=lambda K: abs(K - F))
    iv = {cp: optimize.brentq(lambda s: b76(F, Katm, T, s, cp) - px[Katm][cp], 1e-3, 5, xtol=1e-4) for cp in ("C", "P")}
    out[e] = {"T_days": round(T * 365), "F": F, "basis_pct": (F / S - 1) * 100, "n_strikes": len(Fs), "K_atm": Katm,
              "b76_iv_call": iv["C"], "b76_iv_put": iv["P"]}
    print(e, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[e].items()})
json.dump(out, open(os.path.join(OUT, "forward.json"), "w"), indent=1)
