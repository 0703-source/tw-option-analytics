# Taiwan Stock & Option Analytics

Live demo of the platform described in *A FinTech Analytics and Machine Learning Platform for Taiwan Stock and Index Option Markets* (ICCAI 2027, Special Session 2) and the TCSE 2026 poster *基於機器學習之台灣股票與選擇權分析系統*, by Chia-Chun Kang, Yu-Cheng Li and Yu-De He.

The page runs in the browser: the ETL rules (ROC dates, Big5, `--` → NULL), Black–Scholes pricing and Greeks, Brent's implied-volatility solver, the TXO T-quote, Put/Call ratios, and ARIMA / GARCH(1,1) / stacked LSTM / equal-weighted ensemble forecasts.

## Data refresh

`.github/workflows/update.yml` runs every weekday at 16:07 and 18:37 Taipei time. It downloads the latest TWSE and TAIFEX data (`scripts/update.py`), tests the build (`scripts/test.js`), commits the raw responses to `data/raw/`, and deploys `site/` to GitHub Pages. Run it manually from the **Actions** tab with **Run workflow**.

Local build: `python scripts/update.py` (or `--build-only` to skip downloading), then open `site/index.html`.

Research demonstration, not investment advice.
