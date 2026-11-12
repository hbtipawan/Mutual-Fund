# PKC Fund Desk

Evidence-first research desk for Indian mutual funds (every open-ended Direct-Growth scheme), built the same way as PKCSCREEN: a Python job on GitHub Actions refreshes static JSON every evening, and GitHub Pages serves `site/`.

## Views
- **Pulse** — category heat-map (1M–10Y medians, TER, AUM), active vs index-fund scoreboard, evidence summary
- **Fund Finder** — horizon + loss tolerance + tax slab → category mix → gated shortlist per category
- **Screener** — 1,700+ schemes, filters, 6 column presets, sortable
- **Fund X-Ray** — PKC Score with pillars, flags, growth vs peers vs investable benchmark, rolling 3Y, drawdown, calendar years, capture/alpha/beta/MRAR, returns-based style (large/mid/small/gold/debt) with drift check, TER/BER split, direct-vs-regular cost, credit-event scan
- **Compare** — up to 4 funds, common-period growth, side-by-side metrics, return correlation
- **Portfolio Doctor** — real asset mix, weighted TER & ₹ fees, near-duplicate funds, weak links with better peers, concentration
- **Calculators** — real-NAV SIP back-test, goal planner on category-realised returns, FY 2026-27 redemption tax
- **Evidence & Method** — point-in-time backtest (what predicts future returns and risk), score build, sources

## Run locally
```
pip install pandas numpy scipy requests
python pipeline/fetch.py      # ~5 min first run; TER (slow AMFI endpoint, ~20 min) and AUM refresh weekly
python pipeline/compute.py    # ~1 min
cd site && python -m http.server 8000
```
Set `MF_FORCE=1` to force a TER/AUM refresh.

## Deploy
Push this folder as a repo, enable Pages from the `site/` folder (or move `site/*` to root / `docs/`), and the workflow in `.github/workflows/daily.yml` refreshes data Mon–Sat at 23:10 IST.

Not investment advice.
