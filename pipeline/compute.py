"""PKC Fund Desk - analytics engine.
Reads the cache written by fetch.py and writes the static site's data files:
  site/data/meta.json      as-of date, categories, benchmarks, heat-map, active-vs-passive, sources
  site/data/funds.json     one record per Direct-Growth scheme (returns, risk, consistency, cost, style, score, flags)
  site/data/s/<cat>.json   weekly NAV series per category (+ category median + benchmark) for charts
  site/data/backtest.json  point-in-time validation of the score vs plain trailing returns
Method notes are in METHOD.md.
"""
import json, os, re, math, datetime as dt, warnings
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.optimize import nnls

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE = os.environ.get("MF_CACHE", os.path.join(ROOT, ".cache"))
OUT = os.path.join(ROOT, "site", "data")
os.makedirs(os.path.join(OUT, "s"), exist_ok=True)
W = 52  # weeks per year

# ---------------------------------------------------------------- categories
CAT_RULES = [  # (regex on AMFI category, normalized category, group)
    (r"large & mid", "Large & Mid Cap", "Equity"), (r"large cap", "Large Cap", "Equity"),
    (r"mid cap", "Mid Cap", "Equity"), (r"small cap", "Small Cap", "Equity"),
    (r"multi cap", "Multi Cap", "Equity"), (r"flexi cap", "Flexi Cap", "Equity"),
    (r"focused", "Focused", "Equity"), (r"contra", "Value / Contra", "Equity"), (r"value", "Value / Contra", "Equity"),
    (r"dividend yield", "Dividend Yield", "Equity"), (r"elss", "ELSS (Tax Saver)", "Equity"),
    (r"equity scheme.*sectoral|equity schemes - sectoral", "Sectoral", "Equity"),
    (r"thematic", "Thematic", "Equity"),
    (r"aggressive hybrid", "Aggressive Hybrid", "Hybrid"), (r"balanced hybrid", "Balanced Hybrid", "Hybrid"),
    (r"conservative hybrid", "Conservative Hybrid", "Hybrid"),
    (r"balanced advantage|dynamic asset", "Balanced Advantage", "Hybrid"),
    (r"multi asset", "Multi Asset", "Hybrid"), (r"arbitrage", "Arbitrage", "Hybrid"),
    (r"equity savings", "Equity Savings", "Hybrid"),
    (r"overnight", "Overnight", "Debt"), (r"liquid", "Liquid", "Debt"),
    (r"ultra short to short", "Low Duration", "Debt"), (r"low duration", "Low Duration", "Debt"),
    (r"ultra short", "Ultra Short Duration", "Debt"), (r"money market", "Money Market", "Debt"),
    (r"medium to long", "Medium to Long Duration", "Debt"), (r"medium", "Medium Duration", "Debt"),
    (r"short", "Short Duration", "Debt"), (r"long term|long duration", "Long Duration", "Debt"),
    (r"dynamic", "Dynamic Bond", "Debt"), (r"corporate bond", "Corporate Bond", "Debt"),
    (r"credit risk", "Credit Risk", "Debt"), (r"banking and psu", "Banking & PSU", "Debt"),
    (r"10-year constant", "Gilt 10Y Constant", "Debt"), (r"gilt", "Gilt", "Debt"),
    (r"floating|floater", "Floater", "Debt"), (r"income/debt.*sectoral", "Sectoral Debt", "Debt"),
    (r"retirement", "Retirement (Solution)", "Solution"), (r"children", "Children (Solution)", "Solution"),
    (r"life cycle", "Life Cycle", "Solution"),
    (r"fof overseas|investing overseas", "FoF Overseas", "FoF"), (r"fof domestic|fund of funds scheme \(domestic\)", "FoF Domestic", "FoF"),
]
DEBT_IDX = re.compile(r"sdl|g-sec|gsec|gilt|psu bond|bond|crisil|t-bill|tbill|money market|aaa|corporate|liquid|debt|g sec|treasury", re.I)


def norm_cat(amfi_cat, name):
    a = amfi_cat.lower(); n = name.lower()
    if "index fund" in a or "etf" in a:
        if "gold" in a or "gold" in n or "silver" in n:
            return "Gold & Silver (Passive)", "Passive"
        if "hybrid" in a:
            return "Index (Hybrid)", "Passive"
        if "debt" in a or DEBT_IDX.search(n):
            return "Index (Debt)", "Passive"
        return "Index (Equity)", "Passive"
    if "gold" in n and ("fof" in a or "fund of fund" in a or "gold" in a or "other" in a):
        return "Gold & Silver (Passive)", "Passive"
    if "silver" in n and ("fof" in a or "fund of fund" in a):
        return "Gold & Silver (Passive)", "Passive"
    for rx, c, g in CAT_RULES:
        if re.search(rx, a):
            if c == "Value / Contra" and "contra" in n:
                return "Value / Contra", g
            return c, g
    return "Other", "Other"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def short_name(n):
    n = re.sub(r"\s*[-–]\s*direct plan.*$", "", n, flags=re.I)
    n = re.sub(r"\s*\(?direct\)?\s*(plan)?\s*[-–]?\s*growth.*$", "", n, flags=re.I)
    n = re.sub(r"\s*-\s*growth.*$", "", n, flags=re.I)
    return n.strip(" -")


def base_key(house, name):
    """Plan-agnostic scheme key used to sum AUM over all plans/options and to match TER rows."""
    n = name.lower().replace("&", " and ")
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"\b(direct|regular|plan|option|growth|idcw|payout|reinvestment|dividend|monthly|quarterly|weekly|daily|annual|half yearly|bonus|fund|scheme|the|of|mf|mutual|retail|institutional|super|cumulative|gr|g)\b", " ", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def house_key(h):
    h = (h or "").lower()
    h = re.sub(r"mutual fund|asset management.*|amc|ltd|limited|india|\.|mf", " ", h)
    return re.sub(r"[^a-z0-9]+", " ", h).strip().split(" ")[0] if h.strip() else ""


# ---------------------------------------------------------------- load
def load_nav(code):
    fp = os.path.join(CACHE, "nav", f"{code}.json")
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    rows = d.get("data") or []
    if len(rows) < 20:
        return None
    s = pd.Series({pd.Timestamp(dt.datetime.strptime(r["date"], "%d-%m-%Y")): float(r["nav"]) for r in rows if r["nav"] not in ("", "0", "0.0")})
    s = s[s > 0].sort_index()
    s = s[~s.index.duplicated()]
    # remove one-day data spikes that fully revert (mfapi has occasional bad prints)
    r = s.pct_change()
    bad = (r.abs() > 0.15) & (r.shift(-1).abs() > 0.12) & (np.sign(r) != np.sign(r.shift(-1)))
    s = s[~bad]
    return s


def xirr(cfs):
    """cfs: list of (date, amount). Newton on NPV."""
    if not cfs:
        return None
    t0 = cfs[0][0]
    ts = np.array([(d - t0).days / 365.25 for d, _ in cfs]); a = np.array([c for _, c in cfs])
    r = 0.1
    for _ in range(60):
        f = np.sum(a / (1 + r) ** ts); df = np.sum(-ts * a / (1 + r) ** (ts + 1))
        if df == 0:
            break
        nr = r - f / df
        if abs(nr - r) < 1e-8:
            r = nr; break
        r = max(nr, -0.95)
    return float(r)


def sip_xirr(s, years):
    end = s.index[-1]; start = end - pd.DateOffset(years=years)
    if s.index[0] > start + pd.Timedelta(days=10):
        return None
    m = s[s.index >= start]
    firsts = m.groupby([m.index.year, m.index.month]).head(1)
    firsts = firsts.iloc[:years * 12]
    units = (1000 / firsts).sum()
    cfs = [(d, -1000.0) for d in firsts.index] + [(end, float(units * s.iloc[-1]))]
    return xirr(cfs)


def cagr(s, years):
    end = s.index[-1]; start = end - pd.DateOffset(years=years)
    if s.index[0] > start + pd.Timedelta(days=7):
        return None
    p0 = s.asof(start)
    return float((s.iloc[-1] / p0) ** (1 / years) - 1)


def absret(s, months):
    start = s.index[-1] - pd.DateOffset(months=months)
    if s.index[0] > start + pd.Timedelta(days=5):
        return None
    return float(s.iloc[-1] / s.asof(start) - 1)


def r4(x, n=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


# ---------------------------------------------------------------- main build
def main():
    uni = json.load(open(os.path.join(CACHE, "universe.json")))
    funds = {}
    for u in uni:
        c, g = norm_cat(u["amfi_cat"], u["name"])
        u["cat"], u["grp"] = c, g
        funds[u["code"]] = u
    navs = {}
    for code in list(funds):
        s = load_nav(code)
        if s is not None and len(s) > 40:
            # mfapi can lag AMFI by a day or two: append AMFI's latest NAV if newer
            try:
                d = pd.Timestamp(dt.datetime.strptime(funds[code]["date"], "%d-%b-%Y")); v = float(funds[code]["nav"])
                if d > s.index[-1] and v > 0 and abs(v / s.iloc[-1] - 1) < 0.15:
                    s.loc[d] = v
            except Exception:
                pass
            navs[code] = s
        else:
            funds.pop(code)
    print("funds with history", len(navs))

    # weekly panel (Friday close, ffill <= 2 weeks)
    start = pd.Timestamp("2013-01-04")
    widx = pd.date_range(start, max(s.index[-1] for s in navs.values()), freq="W-FRI")
    WK = pd.DataFrame({c: s.reindex(s.index.union(widx)).ffill(limit=10).reindex(widx) for c, s in navs.items()})
    # don't let ffill extend past each fund's real start
    for c, s in navs.items():
        WK.loc[WK.index < s.index[0], c] = np.nan
    WR = WK.pct_change(fill_method=None)
    asof = max(s.index[-1] for s in navs.values())
    print("as of", asof.date(), "weekly rows", len(WK))

    bycat = defaultdict(list)
    for c, f in funds.items():
        bycat[f["cat"]].append(c)

    # ------------------------------------------------ investable benchmarks (median of index funds tracking same index)
    def idx_pick(rx, neg=None):
        out = []
        for c, f in funds.items():
            if f["cat"] not in ("Index (Equity)", "Gold & Silver (Passive)"):
                continue
            n = f["name"].lower()
            if re.search(rx, n) and not (neg and re.search(neg, n)):
                out.append(c)
        return out
    BSPEC = {
        "N50": ("Nifty 50 (index funds)", idx_pick(r"nifty 50 index|nifty50 index", r"next|equal|value|low vol|momentum|quality|alpha|shariah|elss|tax")),
        "N100": ("Nifty 100", idx_pick(r"nifty 100 index", r"equal|low vol|quality|esg|alpha")),
        "NN50": ("Nifty Next 50", idx_pick(r"next 50", r"etf\b")),
        "M150": ("Nifty Midcap 150", idx_pick(r"midcap 150 index", r"momentum|quality|value|alpha")),
        "S250": ("Nifty Smallcap 250", idx_pick(r"smallcap 250 index", r"momentum|quality|value|alpha")),
        "N500": ("Nifty 500", idx_pick(r"nifty 500 index|total market index", r"momentum|quality|value|multicap|equal|alpha|shariah")),
        "LM250": ("Nifty LargeMidcap 250", idx_pick(r"largemidcap 250|large midcap 250|large mid ?cap 250", r"elss|tax")),
        "GOLD": ("Gold (gold funds)", [c for c, f in funds.items() if f["cat"] == "Gold & Silver (Passive)" and "gold" in f["name"].lower() and "silver" not in f["name"].lower()]),
    }

    def median_chain(codes, min_n=1):
        if not codes:
            return None
        r = WR[codes].median(axis=1, skipna=True)
        cnt = WR[codes].notna().sum(axis=1)
        r[cnt < min_n] = np.nan
        return r

    BR = {}
    for k, (lab, codes) in BSPEC.items():
        BR[k] = median_chain(codes)
    # peer medians for debt-type legs
    def catmed(cat):
        cs = bycat.get(cat, [])
        return WR[cs].median(axis=1, skipna=True) if cs else None
    BR["LIQ"] = catmed("Liquid"); BR["ON"] = catmed("Overnight"); BR["SD"] = catmed("Short Duration")
    BR["ARB"] = catmed("Arbitrage"); BR["CB"] = catmed("Corporate Bond")
    rf_w = BR["ON"].fillna(BR["LIQ"]).fillna(0.065 / W)  # risk-free proxy = overnight fund median weekly return

    def chain(*legs):
        """first non-null of alternative return series (e.g. N100 then N50)."""
        out = None
        for l in legs:
            if l is None:
                continue
            out = l if out is None else out.fillna(l)
        return out

    def comp(parts):
        tot = None
        for w, r in parts:
            tot = (r * w) if tot is None else tot + r * w
        return tot
    EQ = chain(BR["N500"], BR["N50"]); LG = chain(BR["N100"], BR["N50"])
    BMAP = {
        "Large Cap": ("Nifty 100 (index funds; Nifty 50 before 2019)", LG),
        "Large & Mid Cap": ("Nifty LargeMidcap 250 (index funds)", chain(BR["LM250"], comp([(.5, LG), (.5, BR["M150"])]) if BR["M150"] is not None else None, BR["N50"])),
        "Mid Cap": ("Nifty Midcap 150 (index funds)", BR["M150"]),
        "Small Cap": ("Nifty Smallcap 250 (index funds)", BR["S250"]),
        "Multi Cap": ("50/25/25 Nifty100/Midcap150/Smallcap250", chain(comp([(.5, LG), (.25, BR["M150"]), (.25, BR["S250"])]), EQ)),
        "Flexi Cap": ("Nifty 500 (index funds)", EQ), "Focused": ("Nifty 500 (index funds)", EQ),
        "Value / Contra": ("Nifty 500 (index funds)", EQ), "Dividend Yield": ("Nifty 500 (index funds)", EQ),
        "ELSS (Tax Saver)": ("Nifty 500 (index funds)", EQ), "Sectoral": ("Nifty 500 (index funds)", EQ),
        "Thematic": ("Nifty 500 (index funds)", EQ),
        "Aggressive Hybrid": ("70% Nifty 50 + 30% short-duration debt", comp([(.7, BR["N50"]), (.3, BR["SD"])])),
        "Balanced Hybrid": ("50% Nifty 50 + 50% short-duration debt", comp([(.5, BR["N50"]), (.5, BR["SD"])])),
        "Balanced Advantage": ("50% Nifty 50 + 50% short-duration debt", comp([(.5, BR["N50"]), (.5, BR["SD"])])),
        "Conservative Hybrid": ("20% Nifty 50 + 80% short-duration debt", comp([(.2, BR["N50"]), (.8, BR["SD"])])),
        "Equity Savings": ("35% Nifty 50 + 35% arbitrage + 30% debt", comp([(.35, BR["N50"]), (.35, BR["ARB"]), (.3, BR["SD"])])),
        "Multi Asset": ("50% Nifty 500 + 15% gold + 35% debt", comp([(.5, EQ), (.15, BR["GOLD"]), (.35, BR["SD"])])),
        "Arbitrage": ("Liquid fund median", BR["LIQ"]),
        "Retirement (Solution)": ("Nifty 500 (index funds)", EQ), "Children (Solution)": ("70% Nifty 50 + 30% debt", comp([(.7, BR["N50"]), (.3, BR["SD"])])),
    }

    # category median weekly returns and indexes
    CMED = {cat: WR[cs].median(axis=1, skipna=True).where(WR[cs].notna().sum(axis=1) >= 3) for cat, cs in bycat.items()}

    def idx_from_ret(r):
        r = r.copy(); first = r.first_valid_index()
        if first is None:
            return None
        r[r.index < first] = np.nan
        lv = (1 + r.fillna(0)).cumprod(); lv[r.index < first] = np.nan
        return lv

    # daily category-median returns for debt categories (credit-event detector needs same-day peer moves)
    DMED = {}
    for cat, cs in bycat.items():
        if funds[cs[0]]["grp"] == "Debt" or cat == "Index (Debt)":
            D = pd.DataFrame({c: navs[c].pct_change() for c in cs if c in navs})
            DMED[cat] = D.median(axis=1, skipna=True)
    # ------------------------------------------------ per-fund metrics
    def window(c, yrs):
        s = WK[c].dropna()
        if len(s) == 0 or s.index[0] > asof - pd.DateOffset(years=yrs) + pd.Timedelta(days=10):
            return None
        return WR[c][WR.index > asof - pd.DateOffset(years=yrs)]

    def roll_cagr(lv, yrs):
        n = int(round(W * yrs))
        return (lv / lv.shift(n)) ** (1 / yrs) - 1

    def capture(fr, br, up=True):
        m = pd.concat([fr, br], axis=1).dropna()
        if len(m) < 12:
            return None
        m = (1 + m).resample("ME").prod() - 1  # monthly
        sel = m[m.iloc[:, 1] > 0] if up else m[m.iloc[:, 1] < 0]
        if len(sel) < 4:
            return None
        f = (1 + sel.iloc[:, 0]).prod() ** (12 / len(sel)) - 1
        b = (1 + sel.iloc[:, 1]).prod() ** (12 / len(sel)) - 1
        return float(f / b * 100) if b != 0 else None

    def mrar(fr, rf, gamma=2):
        m = pd.concat([fr, rf], axis=1).dropna()
        if len(m) < 26:
            return None
        mm = (1 + m).resample("ME").prod() - 1
        er = (1 + mm.iloc[:, 0]) / (1 + mm.iloc[:, 1]) - 1
        return float(np.mean((1 + er) ** (-gamma)) ** (-12 / gamma) - 1)

    def rbsa(fr, legs):
        m = pd.concat([fr] + [l for _, l in legs], axis=1).dropna()
        if len(m) < 40:
            return None
        y = m.iloc[:, 0].values; X = m.iloc[:, 1:].values
        big = 1e3  # enforce sum(w)=1 via augmented row
        Xa = np.vstack([X, big * np.ones(X.shape[1])]); ya = np.append(y, big)
        w, _ = nnls(Xa, ya)
        fit = X @ w; r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
        return {k: round(float(x) * 100, 1) for (k, _), x in zip(legs, w)}, round(float(r2), 3)

    style_legs = [("Large", BR["N50"]), ("Mid", BR["M150"]), ("Small", BR["S250"]), ("Gold", BR["GOLD"]), ("Debt/Cash", BR["SD"])]

    # TER and AUM maps
    ter_rows = json.load(open(os.path.join(CACHE, "ter.json"))) if os.path.exists(os.path.join(CACHE, "ter.json")) else []
    TER = {}
    for t in ter_rows:
        TER.setdefault(base_key("", t["Scheme_Name"]), []).append(t)
    aumj = json.load(open(os.path.join(CACHE, "aum.json"))) if os.path.exists(os.path.join(CACHE, "aum.json")) else {"rows": []}
    AUMK = defaultdict(float); AUMC = {}
    for a in aumj["rows"]:
        AUMC[a["code"]] = a
    # scheme total AUM: group every plan/option code by plan-agnostic key within the same AMC
    for a in aumj["rows"]:
        AUMK[(house_key(a["house"]), base_key("", a["name"]))] += a["aum_lakh"] or 0

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return None

    recs = []
    for c, f in funds.items():
        s = navs[c]; sw = WK[c].dropna()
        cat = f["cat"]; grp = f["grp"]
        rec = dict(code=c, name=short_name(f["name"]), full=f["name"], amc=(f.get("house") or "").replace(" Mutual Fund", ""), cat=cat, grp=grp,
                   amfi=f["amfi_cat"], nav=r4(s.iloc[-1]), date=s.index[-1].strftime("%Y-%m-%d"),
                   start=s.index[0].strftime("%Y-%m-%d"), age=r4((asof - s.index[0]).days / 365.25, 1))
        for k, m in (("r1m", 1), ("r3m", 3), ("r6m", 6), ("r1y", 12)):
            rec[k] = r4(absret(s, m))
        for k, y in (("r3y", 3), ("r5y", 5), ("r7y", 7), ("r10y", 10)):
            rec[k] = r4(cagr(s, y))
        yrs_all = (s.index[-1] - s.index[0]).days / 365.25
        rec["rsi"] = r4((s.iloc[-1] / s.iloc[0]) ** (1 / yrs_all) - 1) if yrs_all > 1 else None
        rec["sip3"] = r4(sip_xirr(s, 3)); rec["sip5"] = r4(sip_xirr(s, 5))
        # calendar years
        cy = {}
        for y in range(max(2013, s.index[0].year), asof.year + 1):
            a = s[s.index.year == y]; prev = s[s.index.year == y - 1]
            if len(a) < 20:
                continue
            p0 = prev.iloc[-1] if len(prev) else (a.iloc[0] if a.index[0] <= pd.Timestamp(y, 1, 15) else None)
            if p0 is None:
                continue
            cy[str(y)] = r4(a.iloc[-1] / p0 - 1)
        rec["cy"] = cy
        # drawdowns (daily)
        dd = s / s.cummax() - 1
        rec["mdd"] = r4(dd.min()); rec["cdd"] = r4(dd.iloc[-1])
        rec["mdd_date"] = dd.idxmin().strftime("%Y-%m-%d")
        trough = dd.idxmin(); peak_val = s[:trough].max()
        rec_after = s[trough:][s[trough:] >= peak_val]
        rec["mdd_rec_days"] = int((rec_after.index[0] - trough).days) if len(rec_after) else None
        s3 = s[s.index > asof - pd.DateOffset(years=3)]
        rec["mdd3"] = r4((s3 / s3.cummax() - 1).min()) if len(s3) > 100 and s.index[0] < asof - pd.DateOffset(years=3) else None
        # risk on weekly returns
        bname, bret = BMAP.get(cat, (None, None))
        if grp in ("Debt",) or cat in ("Index (Debt)",):
            bname, bret = f"{cat} category median", CMED.get(cat)
        if grp == "Passive" and cat == "Index (Equity)":
            bname, bret = None, None
        if cat == "Gold & Silver (Passive)":
            bname, bret = "Gold funds median", BR["GOLD"]
        rec["bench"] = bname
        for yrs, tag in ((3, "3"), (5, "5")):
            fr = window(c, yrs)
            if fr is None:
                continue
            fr = fr.dropna()
            rfw = rf_w.reindex(fr.index).fillna(0.065 / W)
            ex = fr - rfw
            vol = fr.std() * math.sqrt(W)
            dn = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * math.sqrt(W)
            ann = (1 + fr).prod() ** (W / len(fr)) - 1
            annrf = (1 + rfw).prod() ** (W / len(rfw)) - 1
            rec["vol" + tag] = r4(vol)
            rec["sharpe" + tag] = r4((ann - annrf) / vol if vol > 0 else None, 3)
            rec["sortino" + tag] = r4((ann - annrf) / dn if dn > 0 else None, 3)
            rec["mrar" + tag] = r4(mrar(fr, rfw))
            # Value-Research style downside: avg monthly shortfall vs risk-free
            mm = (1 + pd.concat([fr, rfw], axis=1)).resample("ME").prod() - 1
            sh = (mm.iloc[:, 0] - mm.iloc[:, 1]).clip(upper=0)
            rec["shortfall" + tag] = r4(float(sh.mean()) * 12)
            if bret is not None:
                br = bret.reindex(fr.index)
                m = pd.concat([fr, br], axis=1).dropna()
                if len(m) > 0.8 * len(fr):
                    x = m.iloc[:, 1] - rfw.reindex(m.index); y = m.iloc[:, 0] - rfw.reindex(m.index)
                    beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1) if np.var(x) > 0 else None
                    alpha = (y.mean() - beta * x.mean()) * W if beta is not None else None
                    corr = np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1]
                    te = (m.iloc[:, 0] - m.iloc[:, 1]).std() * math.sqrt(W)
                    bann = (1 + m.iloc[:, 1]).prod() ** (W / len(m)) - 1
                    fann = (1 + m.iloc[:, 0]).prod() ** (W / len(m)) - 1
                    rec["beta" + tag] = r4(beta, 3); rec["alpha" + tag] = r4(alpha)
                    rec["r2_" + tag] = r4(corr ** 2, 3); rec["te" + tag] = r4(te)
                    rec["ir" + tag] = r4((fann - bann) / te if te > 0 else None, 3)
                    rec["xs" + tag] = r4(fann - bann)
                    rec["upc" + tag] = r4(capture(m.iloc[:, 0], m.iloc[:, 1], True), 1)
                    rec["dnc" + tag] = r4(capture(m.iloc[:, 0], m.iloc[:, 1], False), 1)
        # rolling-return consistency (weekly, windows ending in last 5y)
        lv = WK[c]
        cm = idx_from_ret(CMED[cat]) if cat in CMED else None
        bl = idx_from_ret(bret) if bret is not None else None
        for yrs, tag in ((1, "1"), (3, "3"), (5, "5")):
            rr = roll_cagr(lv, yrs)
            rr = rr[rr.index > asof - pd.DateOffset(years=5)].dropna()
            if len(rr) < 26:
                continue
            rec[f"rr{tag}_med"] = r4(rr.median()); rec[f"rr{tag}_min"] = r4(rr.min()); rec[f"rr{tag}_max"] = r4(rr.max())
            rec[f"rr{tag}_pos"] = r4((rr > 0).mean(), 3)
            if cm is not None:
                cr = roll_cagr(cm, yrs).reindex(rr.index)
                ok = cr.notna()
                if ok.sum() >= 26:
                    rec[f"rr{tag}_bcat"] = r4((rr[ok] > cr[ok]).mean(), 3)
            if bl is not None:
                b = roll_cagr(bl, yrs).reindex(rr.index)
                ok = b.notna()
                if ok.sum() >= 26:
                    rec[f"rr{tag}_bb"] = r4((rr[ok] > b[ok]).mean(), 3)
                    up = ok & (b > 0); dn_ = ok & (b <= 0)
        # style (returns-based) for equity & hybrid
        if grp in ("Equity", "Hybrid", "Solution", "FoF") and cat not in ("Arbitrage",):
            fr3 = window(c, 3)
            legs = [(k, l) for k, l in style_legs if l is not None]
            if fr3 is not None:
                st = rbsa(fr3, legs)
                if st:
                    rec["style"], rec["style_r2"] = st
            fr1 = window(c, 1)
            if fr1 is not None:
                st1 = rbsa(fr1, legs)
                if st1:
                    rec["style1"] = st1[0]
        # debt credit-event detector: sharp one-day NAV falls far outside the fund's normal noise
        if (grp == "Debt" or cat == "Index (Debt)") and cat not in ("Gilt", "Gilt 10Y Constant", "Long Duration", "Dynamic Bond", "Medium to Long Duration"):
            # sovereign / duration categories skipped: their sharp falls are rate moves, not credit events
            dr = s.pct_change().dropna()
            dr = dr[dr.index >= "2014-01-01"]
            sd = dr[dr.index > dr.index[-1] - pd.DateOffset(years=10)].std()
            ev = dr[(dr < -max(0.006, 6 * sd))]
            # keep only idiosyncratic falls: market-wide rate shocks (e.g. 19-Mar-2020, 4-May-2022) hit the whole category
            if len(ev):
                cmd = DMED.get(cat)
                if cmd is not None:
                    m = cmd.reindex(ev.index)
                    ev = ev[(ev - m.fillna(0)) < -0.005]
            rec["events"] = [[d.strftime("%Y-%m-%d"), r4(v)] for d, v in ev.items()][-6:]
        # TER
        bk = base_key("", f["name"]); cands = TER.get(bk, [])
        if not cands:  # loose match: AMC first word + key containment
            hk = house_key(f.get("house"))
            for k, v in TER.items():
                if v and k.startswith(hk) and (k == bk or (len(bk) > 8 and (bk in k or k in bk)) ):
                    cands = v; break
        if cands:
            t = sorted(cands, key=lambda x: x["TER_Date"])[-1]
            rec["ter"] = fnum(t.get("D_TER")); rec["ber"] = fnum(t.get("D_BER"))
            rec["ter_reg"] = fnum(t.get("R_TER")); rec["levies"] = fnum(t.get("D_StatutoryLevies"))
            rec["ter_date"] = t["TER_Date"][:10]
        # AUM (₹ crore, all plans of the scheme) — prefer exact code match to locate the AMC key
        a = AUMC.get(c)
        hk = house_key(a["house"]) if a else house_key(f.get("house"))
        tot = AUMK.get((hk, base_key("", a["name"] if a else f["name"])))
        if tot:
            rec["aum"] = round(tot / 100, 1)
        elif a:
            rec["aum"] = round(a["aum_lakh"] / 100, 1)
        recs.append(rec)
    print("records", len(recs))

    # ------------------------------------------------ scoring (peer percentiles within category)
    df = pd.DataFrame(recs).set_index("code")

    def pct(col, higher=True, sub=None):
        x = df.loc[sub, col] if sub is not None else df[col]
        r = x.rank(pct=True, ascending=higher)
        return r

    # Weights follow the backtest (Evidence tab): persistent traits (cost, downside behaviour) weigh most; raw past return least.
    SCORE_EQ = [  # (pillar, column, weight, higher_is_better)
        ("Cost", "ter", 25, False),
        ("Downside", "dnc3", 12, False), ("Downside", "mdd3", 8, True), ("Downside", "shortfall3", 5, True),
        ("Consistency", "cons3", 12, True), ("Consistency", "cons5", 8, True),
        ("Returns", "mrar3", 10, True), ("Returns", "ir3", 10, True),
        ("Robustness", "aum_log", 5, True), ("Robustness", "age_c", 5, True),
    ]
    SCORE_DEBT = [
        ("Cost", "ter", 25, False),
        ("Downside", "mdd3", 10, True), ("Downside", "vol3", 10, False), ("Downside", "ev_pen", 10, False),
        ("Consistency", "rr1_bcat", 15, True), ("Consistency", "rr1_min", 5, True),
        ("Returns", "r3y", 10, True), ("Returns", "rr1_med", 10, True),
        ("Robustness", "aum_log", 5, True),
    ]
    SCORE_IDX = [  # passive: tracking quality & cost within same underlying index
        ("Tracking", "td1", 35, True), ("Tracking", "tev", 25, False), ("Cost", "ter", 30, False), ("Robustness", "aum_log", 10, True),
    ]
    df["aum_log"] = np.log10(df["aum"].clip(lower=1)) if "aum" in df else np.nan
    df["aum_log"] = df["aum_log"].clip(upper=np.log10(20000))  # beyond ₹20k cr size isn't a plus
    for t in ("3", "5"):
        bb, bc = f"rr{t}_bb", f"rr{t}_bcat"
        df["cons" + t] = (df[bb] if bb in df else pd.Series(np.nan, index=df.index)).fillna(df[bc] if bc in df else np.nan)
    df["age_c"] = df["age"].clip(upper=10)
    df["ev_pen"] = df["events"].apply(lambda e: len(e) if isinstance(e, list) else 0) if "events" in df else 0

    # passive: underlying index key & tracking vs same-index median
    def idx_key(n):
        n = n.lower()
        n = re.sub(r"\b(direct|plan|growth|index|fund|fof|etf|scheme|of|the|option|mutual)\b", " ", n)
        n = re.sub(r"^\S+\s+(sun life\s+|prudential\s+|mutual\s+|india\s+|asset\s+|blackrock\s+|finserv\s+|case\s+|bnp paribas\s+|one\s+|oswal\s+|mf\s+)?", "", n)
        return re.sub(r"[^a-z0-9]+", " ", n).strip()
    pas = df[df["cat"].isin(["Index (Equity)", "Index (Debt)", "Gold & Silver (Passive)"])].copy()
    pas["ikey"] = pas["name"].apply(idx_key)
    df["ikey"] = pas["ikey"]
    for k, grpdf in pas.groupby("ikey"):
        codes = list(grpdf.index)
        if len(codes) < 2:
            continue
        r1 = WR[codes][WR.index > asof - pd.DateOffset(years=1)]
        med = r1.median(axis=1)
        for cc in codes:
            x = r1[cc].dropna()
            if len(x) < 40:
                continue
            m = med.reindex(x.index)
            df.loc[cc, "td1"] = ((1 + x).prod() - (1 + m).prod())  # 1y return vs same-index peer median
            df.loc[cc, "tev"] = (x - m).std() * math.sqrt(W)
            df.loc[cc, "ipeers"] = len(codes)

    df["score"] = np.nan; df["pillars"] = None
    for cat, sub in df.groupby("cat"):
        grp = sub["grp"].iloc[0]
        if cat.startswith("Index") or cat.startswith("Gold"):
            spec = SCORE_IDX; elig = sub.index[sub["td1"].notna()] if "td1" in sub else []
            # rank inside the same underlying index
            for k, g2 in sub.loc[elig].groupby("ikey"):
                if len(g2) < 3:
                    continue
                tot = pd.Series(0.0, index=g2.index); wsum = pd.Series(0.0, index=g2.index); pil = defaultdict(dict)
                for p, col, w, hi in spec:
                    if col not in g2 or g2[col].notna().sum() < 3:
                        continue
                    r = g2[col].rank(pct=True, ascending=hi)
                    for cc in g2.index:
                        if pd.notna(r[cc]):
                            tot[cc] += r[cc] * w; wsum[cc] += w
                            pil[cc].setdefault(p, []).append((r[cc], w))
                for cc in g2.index:
                    if wsum[cc] >= 60:
                        df.at[cc, "score"] = round(100 * tot[cc] / wsum[cc], 1)
                        df.at[cc, "pillars"] = {p: round(100 * sum(a * b for a, b in v) / sum(b for _, b in v)) for p, v in pil[cc].items()}
            continue
        spec = SCORE_DEBT if grp == "Debt" else SCORE_EQ
        elig = sub.index[(sub["age"] >= 3) & sub["r3y"].notna()]
        if len(elig) < 5:
            continue
        e = sub.loc[elig]
        tot = pd.Series(0.0, index=e.index); wsum = pd.Series(0.0, index=e.index); pil = defaultdict(dict)
        for p, col, w, hi in spec:
            if col not in e or e[col].notna().sum() < 4:
                continue
            r = e[col].rank(pct=True, ascending=hi)
            for cc in e.index:
                if pd.notna(r[cc]):
                    tot[cc] += r[cc] * w; wsum[cc] += w
                    pil[cc].setdefault(p, []).append((r[cc], w))
        for cc in e.index:
            if wsum[cc] >= 60:
                df.at[cc, "score"] = round(100 * tot[cc] / wsum[cc], 1)
                df.at[cc, "pillars"] = {p: round(100 * sum(a * b for a, b in v) / sum(b for _, b in v)) for p, v in pil[cc].items()}
        df.loc[elig, "crank"] = df.loc[elig, "score"].rank(ascending=False, method="min")
        df.loc[elig, "cn"] = len(elig)

    # ------------------------------------------------ flags (evidence-based red / amber / green notes)
    catstats = {}
    for cat, sub in df.groupby("cat"):
        catstats[cat] = dict(ter_q3=sub["ter"].quantile(.75) if "ter" in sub else None,
                             ter_med=sub["ter"].median() if "ter" in sub else None)

    def flags(r):
        out = []
        cs = catstats.get(r["cat"], {})
        if pd.notna(r.get("age")) and r["age"] < 3:
            out.append(["amber", "Less than 3 years of history — score withheld, numbers are unproven."])
        if pd.notna(r.get("aum")) and r["aum"] < 100 and r["grp"] != "Passive":
            out.append(["amber", f"Small fund (₹{r['aum']:.0f} cr) — closure/merger risk and lumpy flows."])
        if r["cat"] == "Small Cap" and pd.notna(r.get("aum")) and r["aum"] > 25000:
            out.append(["amber", f"Very large small-cap fund (₹{r['aum']:,.0f} cr) — liquidity can limit agility."])
        if pd.notna(r.get("ter")) and cs.get("ter_q3") is not None and r["ter"] > cs["ter_q3"]:
            out.append(["amber", f"Costlier than 75% of peers (TER {r['ter']:.2f}% vs median {cs['ter_med']:.2f}%)."])
        if pd.notna(r.get("ter")) and pd.notna(r.get("ter_reg")) and r["ter_reg"] - r["ter"] >= 0.75:
            out.append(["info", f"Regular plan costs {r['ter_reg'] - r['ter']:.2f}% a year more than this Direct plan."])
        if isinstance(r.get("events"), list) and r["events"]:
            d, v = r["events"][-1]
            recent = d >= (pd.Timestamp(r["date"]) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
            out.append(["red" if recent else "amber", f"{'Recent' if recent else 'Past'} sharp one-day NAV fall of {v * 100:.2f}% on {d}, far worse than peers that day — typical of a credit event/write-down. Check the factsheet."])
        if pd.notna(r.get("rr3_bb")) and r["rr3_bb"] < 0.3 and r["grp"] in ("Equity", "Hybrid"):
            out.append(["amber", f"Beat its index-fund benchmark in only {r['rr3_bb'] * 100:.0f}% of 3-year windows — an index fund would usually have done better."])
        if pd.notna(r.get("rr3_bb")) and r["rr3_bb"] >= 0.8 and r["grp"] in ("Equity", "Hybrid"):
            out.append(["green", f"Beat its index-fund benchmark in {r['rr3_bb'] * 100:.0f}% of 3-year windows."])
        if pd.notna(r.get("dnc3")) and r["dnc3"] < 85 and r["grp"] in ("Equity", "Hybrid"):
            out.append(["green", f"Falls less than the market: downside capture {r['dnc3']:.0f}%."])
        if pd.notna(r.get("dnc3")) and r["dnc3"] > 110 and r["grp"] in ("Equity",):
            out.append(["amber", f"Falls harder than the market: downside capture {r['dnc3']:.0f}%."])
        st, st1 = r.get("style"), r.get("style1")
        if isinstance(st, dict) and isinstance(st1, dict):
            drift = sum(abs(st.get(k, 0) - st1.get(k, 0)) for k in st) / 2
            if drift >= 25:
                out.append(["amber", f"Style drift: effective large/mid/small mix moved ~{drift:.0f} pts in the last year vs 3-year average."])
        if r["cat"] == "Large Cap" and isinstance(st, dict) and st.get("Large", 100) < 60:
            out.append(["amber", f"Behaves only ~{st.get('Large', 0):.0f}% like large-caps — more mid/small risk than the label suggests."])
        if pd.notna(r.get("r1y")) and pd.notna(r.get("rr3_bcat")) and r["rr3_bcat"] > 0.7:
            pass
        return out
    df["flags"] = df.apply(flags, axis=1)

    def verdict(r):
        s = r.get("score")
        if pd.isna(s):
            return "Unrated"
        return "Strong" if s >= 75 else "Good" if s >= 60 else "Average" if s >= 40 else "Weak"
    df["verdict"] = df.apply(verdict, axis=1)

    # ------------------------------------------------ category heat-map & active vs passive
    heat = []
    for cat, sub in df.groupby("cat"):
        row = dict(cat=cat, grp=sub["grp"].iloc[0], n=len(sub), slug=slug(cat))
        for k in ("r1m", "r3m", "r6m", "r1y", "r3y", "r5y", "r10y"):
            row[k] = r4(sub[k].median()) if sub[k].notna().sum() >= 3 else None
        row["ter_med"] = r4(sub["ter"].median(), 3) if "ter" in sub and sub["ter"].notna().any() else None
        row["aum"] = r4(sub["aum"].sum(), 0) if "aum" in sub else None
        row["bench"] = BMAP.get(cat, (None,))[0]
        heat.append(row)
    avp = []
    for cat, sub in df.groupby("cat"):
        if sub["grp"].iloc[0] not in ("Equity", "Hybrid") or cat in ("Sectoral", "Thematic"):
            continue
        row = dict(cat=cat)
        for t in ("3", "5"):
            x = sub["xs" + t].dropna() if ("xs" + t) in sub else pd.Series(dtype=float)
            row["n" + t] = int(len(x)); row["beat" + t] = r4((x > 0).mean(), 3) if len(x) else None
            row["med_xs" + t] = r4(x.median()) if len(x) else None
        x = sub["rr3_bb"].dropna() if "rr3_bb" in sub else pd.Series(dtype=float)
        row["roll3_med"] = r4(x.median(), 3) if len(x) else None
        row["bench"] = BMAP.get(cat, (None,))[0]
        avp.append(row)

    # ------------------------------------------------ series files per category (weekly)
    cats_meta = []
    for cat, cs in bycat.items():
        cs = [c for c in cs if c in WK.columns]
        sub = WK[cs]
        first = sub.first_valid_index() or WK.index[0]
        sub = sub[sub.index >= first]
        out = dict(cat=cat, start=first.strftime("%Y-%m-%d"), n=len(sub), f={})
        for c in cs:
            v = sub[c]
            out["f"][str(c)] = [None if pd.isna(x) else float(f"{x:.5g}") for x in v.values]
        cmi = idx_from_ret(CMED[cat]) if cat in CMED else None
        if cmi is not None:
            out["cat_med"] = [None if pd.isna(x) else round(float(x), 5) for x in cmi.reindex(sub.index).values]
        bn, br = BMAP.get(cat, (None, None))
        if cat in ("Gold & Silver (Passive)",):
            bn, br = "Gold funds median", BR["GOLD"]
        if br is not None:
            bi = idx_from_ret(br)
            out["bench"] = [None if pd.isna(x) else round(float(x), 5) for x in bi.reindex(sub.index).values]
            out["bench_name"] = bn
        # Nifty 50 always included for context on equity-type categories
        if BR["N50"] is not None:
            ni = idx_from_ret(BR["N50"])
            out["n50"] = [None if pd.isna(x) else round(float(x), 5) for x in ni.reindex(sub.index).values]
        json.dump(out, open(os.path.join(OUT, "s", slug(cat) + ".json"), "w"), separators=(",", ":"))
        cats_meta.append(dict(cat=cat, slug=slug(cat), grp=funds[cs[0]]["grp"] if cs else "", n=len(cs)))

    # ------------------------------------------------ backtest: does the score pick future winners better than trailing return?
    bt = backtest(WK, WR, df, bycat, BMAP, rf_w, asof)

    # ------------------------------------------------ write
    keep = [c for c in df.columns if c not in ("full",)]
    out_f = []
    for code, r in df.iterrows():
        d = {"code": int(code)}
        for k in keep:
            v = r[k]
            if isinstance(v, (list, dict)):
                d[k] = v
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            elif isinstance(v, (np.floating, float)):
                d[k] = round(float(v), 5)
            elif isinstance(v, (np.integer,)):
                d[k] = int(v)
            else:
                d[k] = v
        out_f.append(d)
    json.dump(out_f, open(os.path.join(OUT, "funds.json"), "w"), separators=(",", ":"), ensure_ascii=False)
    meta = dict(asof=asof.strftime("%Y-%m-%d"), built=dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M IST"),
                n=len(out_f), cats=sorted(cats_meta, key=lambda x: (x["grp"], x["cat"])), heat=heat, avp=avp,
                aum_period=aumj.get("period"), ter_date=max([r.get("ter_date", "") for r in out_f] or [""]),
                bench={k: dict(name=v[0], n=len(v[1])) for k, v in BSPEC.items()}, rf=r4((1 + rf_w[rf_w.index > asof - pd.DateOffset(years=1)]).prod() - 1))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), separators=(",", ":"), ensure_ascii=False)
    json.dump(bt, open(os.path.join(OUT, "backtest.json"), "w"), separators=(",", ":"))
    print("written", len(out_f), "funds; scored", int(df["score"].notna().sum()))


def backtest(WK, WR, df, bycat, BMAP, rf_w, asof):
    """Point-in-time test. At each test date T (every 6 months, 2016-01 .. asof-3y), using ONLY data up to T:
       build a NAV-only version of the score (cost & AUM excluded because history of those isn't available),
       then measure each fund's NEXT-3-YEAR CAGR minus its category median. Compare with ranking by plain 3Y trailing return.
       Caveat: universe = funds alive today (survivorship bias), categories as of today."""
    cats = ["Large Cap", "Large & Mid Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Multi Cap", "Focused", "Value / Contra",
            "ELSS (Tax Saver)", "Aggressive Hybrid", "Balanced Advantage"]
    dates = pd.date_range("2016-01-01", asof - pd.DateOffset(years=3), freq="6MS")
    rows = []

    def snap_idx(i, d):
        return WK.index.searchsorted(d) - 1
    for T in dates:
        iT = WK.index.searchsorted(T)
        if iT >= len(WK.index):
            continue
        T = WK.index[iT]; iF = WK.index.searchsorted(T + pd.DateOffset(years=3))
        if iF >= len(WK.index):
            continue
        F = WK.index[iF]
        for cat in cats:
            cs = [c for c in bycat.get(cat, []) if c in WK.columns]
            P = WK[cs]
            hist = P.iloc[max(0, iT - 3 * W):iT + 1]
            ok = [c for c in cs if hist[c].notna().sum() >= 3 * W - 2 and pd.notna(P[c].iloc[iF])]
            if len(ok) < 6:
                continue
            h = hist[ok]; r = h.pct_change(fill_method=None).iloc[1:]
            cm = r.median(axis=1)
            bret = BMAP.get(cat, (None, None))[1]
            br = bret.reindex(r.index) if bret is not None else None
            rf = rf_w.reindex(r.index).fillna(0.065 / W)
            feat = pd.DataFrame(index=ok)
            feat["tr3"] = (h.iloc[-1] / h.iloc[0]) ** (1 / 3) - 1
            # consistency vs category median: % of 1y windows (inside the 3y lookback) beating cat median
            lv = (1 + r).cumprod(); cml = (1 + cm).cumprod()
            r1 = lv / lv.shift(W) - 1; c1 = cml / cml.shift(W) - 1
            feat["cons"] = (r1.gt(c1, axis=0)).iloc[W:].mean()
            mm = (1 + r).resample("ME").prod() - 1; rfm = (1 + rf).resample("ME").prod() - 1
            er = (1 + mm).div(1 + rfm, axis=0) - 1
            feat["mrar"] = (np.mean((1 + er) ** (-2), axis=0)) ** (-6) - 1
            ex = r.sub(rf, axis=0)
            dn = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * math.sqrt(W)
            ann = (1 + r).prod() ** (W / len(r)) - 1
            feat["sortino"] = (ann - ((1 + rf).prod() ** (W / len(rf)) - 1)) / dn
            dd = (h / h.cummax() - 1).min(); feat["mdd"] = dd
            if br is not None and br.notna().mean() > 0.9:
                bm = (1 + br).resample("ME").prod() - 1
                down = bm < 0
                if down.sum() >= 4:
                    fd = (1 + mm[down]).prod() ** (12 / down.sum()) - 1; bd = (1 + bm[down]).prod() ** (12 / down.sum()) - 1
                    feat["dnc"] = fd / bd
            feat["tr1"] = h.iloc[-1] / h.iloc[-W - 1] - 1
            feat["vol"] = r.std() * math.sqrt(W)
            if br is not None and br.notna().mean() > 0.9:
                bmm = (1 + br).resample("ME").prod() - 1; up = bmm > 0
                if up.sum() >= 4:
                    fu = (1 + mm[up]).prod() ** (12 / up.sum()) - 1; bu = (1 + bmm[up]).prod() ** (12 / up.sum()) - 1
                    feat["upc"] = fu / bu
                xr = r.sub(br, axis=0); feat["ir"] = (xr.mean() * W) / (xr.std() * math.sqrt(W))
            feat["ter"] = [df.at[c, "ter"] if ("ter" in df.columns and c in df.index and pd.notna(df.at[c, "ter"])) else np.nan for c in ok]
            # forward risk outcomes
            fh = P.iloc[iT:iF + 1][ok]; frr = fh.pct_change(fill_method=None).iloc[1:]
            feat["fvol"] = frr.std() * math.sqrt(W); feat["fmdd"] = (fh / fh.cummax() - 1).min()
            if br is not None:
                fbr = bret.reindex(frr.index)
                if fbr.notna().mean() > 0.9:
                    fmm = (1 + frr).resample("ME").prod() - 1; fbm = (1 + fbr).resample("ME").prod() - 1; dwn = fbm < 0
                    if dwn.sum() >= 4:
                        feat["fdnc"] = ((1 + fmm[dwn]).prod() ** (12 / dwn.sum()) - 1) / ((1 + fbm[dwn]).prod() ** (12 / dwn.sum()) - 1)
            sc = (feat["tr3"].rank(pct=True) * 20 + feat["mrar"].rank(pct=True) * 15 + feat["cons"].rank(pct=True) * 25 +
                  feat["sortino"].rank(pct=True) * 15 + feat["mdd"].rank(pct=True) * 10)
            wsum = 85
            if "dnc" in feat:
                sc = sc + feat["dnc"].rank(pct=True, ascending=False) * 15; wsum += 15
            feat["score"] = sc / wsum
            fwd = (P.iloc[iF][ok] / P.iloc[iT][ok]) ** (1 / 3) - 1
            feat["fxs"] = fwd - fwd.median()
            for c in ok:
                d = dict(T=T.strftime("%Y-%m"), cat=cat, code=int(c))
                for k in feat.columns:
                    v = feat.at[c, k]
                    d[k] = float(v) if pd.notna(v) else np.nan
                rows.append(d)
    if not rows:
        return {}
    B = pd.DataFrame(rows)

    def quint(col):
        q = B.groupby(["T", "cat"])[col].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
        return q
    B["q_score"] = quint("score"); B["q_tr3"] = quint("tr3")
    res = dict(n_obs=len(B), n_dates=B["T"].nunique(), dates=[B["T"].min(), B["T"].max()], cats=sorted(B["cat"].unique()))
    for m in ("score", "tr3"):
        g = B.groupby("q_" + m)["fxs"]
        res[m] = dict(q_mean=[r4(x) for x in g.mean().values], q_hit=[r4(x, 3) for x in B.groupby("q_" + m)["fxs"].apply(lambda x: (x > 0).mean()).values],
                      ic=r4(B.groupby(["T", "cat"]).apply(lambda x: x[m].corr(x["fxs"], method="spearman")).mean(), 3),
                      top_bottom=r4(g.mean().iloc[-1] - g.mean().iloc[0]))
    # per category top-quintile hit-rate for the score
    pc = []
    for cat, x in B.groupby("cat"):
        t = x[x["q_score"] == 5]; b = x[x["q_score"] == 1]; t2 = x[x["q_tr3"] == 5]
        pc.append(dict(cat=cat, n=len(x), top_hit=r4((t["fxs"] > 0).mean(), 3), top_xs=r4(t["fxs"].mean()), bot_xs=r4(b["fxs"].mean()),
                       tr_top_hit=r4((t2["fxs"] > 0).mean(), 3), tr_top_xs=r4(t2["fxs"].mean())))
    res["by_cat"] = pc
    # information coefficient of every feature vs each forward outcome (mean Spearman across date x category snapshots)
    feats = [f for f in ("tr1", "tr3", "cons", "mrar", "sortino", "mdd", "vol", "dnc", "upc", "ir", "ter", "score") if f in B]
    outs = [o for o in ("fxs", "fvol", "fmdd", "fdnc") if o in B]
    ics = {}
    for f in feats:
        ics[f] = {}
        for o in outs:
            v = B.groupby(["T", "cat"]).apply(lambda x: x[f].corr(x[o], method="spearman") if x[[f, o]].dropna().shape[0] >= 6 else np.nan).dropna()
            ics[f][o] = dict(ic=r4(v.mean(), 3), t=r4(v.mean() / v.std() * math.sqrt(len(v)) if v.std() > 0 else None, 2), n=int(len(v)))
    res["ic"] = ics
    B.to_pickle(os.path.join(CACHE, "bt.pkl"))
    # persistence: chance a top-quartile (by trailing 3y) fund stays top-quartile next 3y
    B["q4_tr"] = B.groupby(["T", "cat"])["tr3"].transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False) + 1)
    B["q4_f"] = B.groupby(["T", "cat"])["fxs"].transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False) + 1)
    top = B[B["q4_tr"] == 4]
    res["persist"] = dict(stay_top=r4((top["q4_f"] == 4).mean(), 3), fall_bottom=r4((top["q4_f"] == 1).mean(), 3), n=len(top))
    return res


if __name__ == "__main__":
    main()
