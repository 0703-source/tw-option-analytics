# Experiments for the ICCAI 2027 paper

Reproduces every number, table and figure in Section 7 of
"A FinTech Analytics and Machine Learning Platform for Taiwan Stock and Index Option Markets".

| Script | Output |
|---|---|
| `fetch_data.py` | (optional) re-downloads any missing TWSE month into `raw/` |
| `run_experiments.py` | `out/results.json`, `out/preds.json` — forecasts for 2330/2317/2454, 2023-01-03..2026-10-30, 80/20 split, ARIMA (ADF+AIC), GARCH(1,1), stacked LSTM ×5 seeds, ensembles, Diebold–Mariano tests, GARCH volatility losses, Brent vs bisection IV on the 2026-11-09 TXO chain, ETL counts |
| `forward.py` | `out/forward.json` — put–call-parity implied forwards and Black-76 ATM IV |
| `figures.py` | `out/fig*.png` |
| `h3_latency.py` | `out/h3.json` — main-thread frame gaps on the live site (Web Worker vs main thread), needs Playwright + Microsoft Edge |

`raw/` holds the TWSE responses and `snapshot/` the TAIFEX daily report and TAIEX history for the option experiments, so results do not change as new data arrive.

```
pip install numpy pandas scipy statsmodels arch matplotlib jax "keras>=3"
set KERAS_BACKEND=jax        (PowerShell: $env:KERAS_BACKEND="jax")
python run_experiments.py && python forward.py && python figures.py
```
Tested with Python 3.12, statsmodels 0.15.0, arch 8.0.0, Keras 3.15.1 on JAX 0.12 (CPU).
