"""Figures for the paper (grayscale-safe, 300 dpi PNG)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
R = json.load(open(os.path.join(OUT, "results.json")))
PR = json.load(open(os.path.join(OUT, "preds.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

# Fig 1: architecture
fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
def box(x, y, w, h, title, body, fc="#f2f2f2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#222", lw=0.9))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=8.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=7.2, linespacing=1.35)
def arrow(x1, y1, x2, y2, label="", dashed=False):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222", linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=6.8, style="italic")
box(0.1, 3.3, 2.1, 1.9, "Data sources", "TWSE STOCK_DAY\nTAIEX index history\nTAIFEX OpenAPI\n(TXO daily report)")
box(2.8, 3.3, 2.1, 1.9, "ETL pipeline", "ROC → ISO 8601\nBig5 → UTF-8\n'--' → NULL\nthousand separators")
box(5.5, 3.3, 2.0, 1.9, "MySQL 8.0", "equities\noption contracts\noption prices + Greeks\nprediction logs")
box(8.0, 3.3, 1.9, 1.9, "Laravel 10 API", "REST endpoints\nscheduler\nRepository pattern\nRedis cache")
box(5.3, 0.3, 2.4, 2.2, "Python workers", "ARIMA (ADF+AIC)\nGARCH(1,1)\nstacked LSTM\nBlack–Scholes + Brent IV", fc="#e4e4e4")
box(2.8, 0.3, 2.1, 2.2, "Redis 7 queue", "job dispatch\nretry (3×, back-off)\nSupervisor workers")
box(8.0, 0.3, 1.9, 2.2, "Vue 3 UI", "price & volume\nT-quote, IV smile\nforecast panel\nChart.js")
arrow(2.2, 4.25, 2.8, 4.25, "daily"); arrow(4.9, 4.25, 5.5, 4.25); arrow(7.5, 4.25, 8.0, 4.25)
ax.plot([8.4, 8.4, 3.85], [3.3, 2.9, 2.9], color="#222", lw=0.9, ls="--")  # dispatch path routed under the top row
arrow(3.85, 2.9, 3.85, 2.5, "", dashed=True); ax.text(7.75, 2.95, "dispatch job", fontsize=6.8, style="italic", ha="right", va="bottom")
arrow(4.9, 1.4, 5.3, 1.4, "")
arrow(6.5, 2.5, 6.5, 3.3, ""); ax.text(6.58, 2.65, "JSON results", fontsize=6.8, style="italic", ha="left", va="center")
arrow(9.3, 3.3, 9.3, 2.5, ""); ax.text(9.4, 2.9, "REST", fontsize=6.8, style="italic")
fig.savefig(os.path.join(OUT, "fig1_architecture.png")); plt.close(fig)

# Fig 2: TSMC forecasts on the test set
d = PR["2330"]; dt = pd.to_datetime(d["dates"])
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [2.2, 1]})
for ax, sl in ((axs[0], slice(None)), (axs[1], slice(-30, None))):
    ax.plot(dt[sl], np.array(d["act"])[sl], color="black", lw=1.6, label="Actual close")
    ax.plot(dt[sl], np.array(d["arima"])[sl], color="#555", lw=0.9, ls="--", label="ARIMA")
    ax.plot(dt[sl], np.array(d["lstm0"])[sl], color="#888", lw=0.9, ls=":", label="LSTM (seed 0)")
    ax.plot(dt[sl], np.array(d["ens3_0"])[sl], color="#333", lw=0.9, ls="-.", label="Ensemble")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m" if sl.start is None else "%m-%d"))
    ax.tick_params(axis="x", labelrotation=30); ax.grid(axis="y", lw=0.3, color="#ccc")
axs[0].set_ylabel("Close (TWD)"); axs[0].set_title("(a) TSMC 2330, full test period", fontsize=9)
axs[1].set_title("(b) last 30 trading days", fontsize=9); axs[0].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_tsmc_forecasts.png")); plt.close(fig)

# Fig 3: Theil's U by stock and model
models = [("ARIMA", "ARIMA", "//"), ("GARCH", "GARCH(1,1)", ".."), ("LSTM", "LSTM", "xx"), ("ENS3", "Ensemble (3)", ""), ("ENS2", "ARIMA+LSTM", "\\\\")]
fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = 0.16; x = np.arange(3)
for i, (k, lab, h) in enumerate(models):
    u = [R["stocks"][c]["metrics"][k]["u"] for c in ("2330", "2317", "2454")]
    e = [R["stocks"][c]["metrics"][k].get("u_sd", 0) for c in ("2330", "2317", "2454")]
    ax.bar(x + (i - 2) * w, u, w, yerr=e, capsize=2, label=lab, color=["#bbb", "#ddd", "#888", "#444", "#fff"][i], edgecolor="black", hatch=h, lw=0.6)
ax.axhline(1, color="black", lw=0.8, ls="--"); ax.text(-0.48, 1.002, "random walk (U = 1)", fontsize=7, va="bottom", ha="left")
lo = min(R["stocks"][c]["metrics"][k]["u"] for c in ("2330", "2317", "2454") for k, _, _ in models)
hi = max(R["stocks"][c]["metrics"][k]["u"] + R["stocks"][c]["metrics"][k].get("u_sd", 0) for c in ("2330", "2317", "2454") for k, _, _ in models)
ax.set_ylim(min(0.95, lo - 0.02), hi + 0.03)
ax.set_xticks(x, ["TSMC (2330)", "Foxconn (2317)", "MediaTek (2454)"]); ax.set_ylabel("Theil's U (lower is better)")
ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.grid(axis="y", lw=0.3, color="#ccc")
fig.savefig(os.path.join(OUT, "fig3_theil_u.png")); plt.close(fig)

# Fig 4: IV smile from the live TXO chain
iv = R["iv"]; pts = iv["points"]; S = iv["S"]
fig, ax = plt.subplots(figsize=(7.2, 2.15)); marks = ["o", "s", "^"]
for i, e in enumerate(iv["expiries"]):
    # out-of-the-money side of each expiry (puts below spot, calls above) with open interest > 0
    q = sorted((k, v) for ee, k, cp, v, oi in pts if ee == e and oi > 0 and ((cp == "P" and k < S) or (cp == "C" and k >= S)) and abs(k / S - 1) < 0.15)
    if q:
        ax.plot([k / S for k, _ in q], [v * 100 for _, v in q], marker=marks[i], ms=2.6, lw=0.8, color=["#000", "#555", "#999"][i],
                label=f"{e[:4]}/{e[4:]} expiry", mfc="white" if i == 1 else None)
ax.axvline(1, color="black", lw=0.6, ls=":"); ax.set_xlabel("Moneyness K / S"); ax.set_ylabel("Implied volatility (%)")
ax.legend(frameon=False, fontsize=7); ax.grid(lw=0.3, color="#ddd")
ax.set_title(f"TXO settlement prices, trade date {iv['trade_date']}, TAIEX S = {S:,.2f}", fontsize=8.5)
fig.savefig(os.path.join(OUT, "fig4_iv_smile.png")); plt.close(fig)
print("figures written")
