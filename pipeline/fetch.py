"""PKC Fund Desk - data fetch.
Pulls (1) AMFI NAVAll (universe + SEBI category), (2) full NAV history per scheme from mfapi.in,
(3) AMFI daily TER with BER split, (4) AMFI average AUM (latest quarter, scheme-wise).
Everything is cached under CACHE so a re-run only refreshes what changed.
"""
import json, os, re, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests

CACHE = os.environ.get("MF_CACHE", os.path.join(os.path.dirname(__file__), "..", ".cache"))
os.makedirs(os.path.join(CACHE, "nav"), exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (PKC Fund Desk; personal research)"


def get(url, tries=4, timeout=40):
    for i in range(tries):
        try:
            r = S.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError("failed " + url)


def parse_navall(text):
    cat = None; house = None; rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(Open Ended Schemes|Close Ended Schemes|Interval Fund Schemes)\((.*)\)$", line)
        if m:
            cat = (m.group(1), m.group(2)); continue
        if ";" not in line:
            house = line; continue
        p = line.split(";")
        if p[0] == "Scheme Code" or len(p) < 8:
            continue
        rows.append(dict(code=int(p[0]), isin=p[1], name=p[3].strip(), plan=p[4].strip(), opt=p[5].strip(),
                         nav=p[6], date=p[7], type=cat[0] if cat else "", amfi_cat=cat[1] if cat else "", house=house))
    return rows


def is_direct_growth(r):
    n = r["name"].lower(); o = r["opt"].lower(); pl = r["plan"].lower()
    direct = pl == "direct plan" or (pl == "" and "direct" in n)
    if not direct:
        return False
    if any(k in o for k in ("idcw", "dividend", "bonus", "payout", "reinvest")):
        return False
    growth = "growth" in o or (o == "" and "growth" in n)
    return growth


def fetch_nav(code):
    fp = os.path.join(CACHE, "nav", f"{code}.json")
    today = dt.date.today().isoformat()
    if os.path.exists(fp):
        try:
            d = json.load(open(fp))
            if d.get("_fetched") == today:
                return code, True
        except Exception:
            pass
    try:
        r = get(f"https://api.mfapi.in/mf/{code}")
        d = r.json(); d["_fetched"] = today
        json.dump(d, open(fp, "w"))
        return code, True
    except Exception:
        return code, False


def fetch_ter():
    out = []
    first = get("https://www.amfiindia.com/api/populate-te-rdata-revised?page=1&pageSize=100").json()
    pages = first["meta"]["pageCount"]; out += first["data"]

    def pg(i):
        try:
            return get(f"https://www.amfiindia.com/api/populate-te-rdata-revised?page={i}&pageSize=100", tries=7).json()["data"]
        except Exception:
            print("ter page miss", i, flush=True); return []
    with ThreadPoolExecutor(3) as ex:
        for d in ex.map(pg, range(2, pages + 1)):
            out += d
    # AMCs that have not yet filed this month (e.g. Nippon in early month): take last month's filing for them
    seen = {x.get("MF_ID") for x in out}
    prev = (dt.date.today().replace(day=1) - dt.timedelta(days=1)).strftime("%m-%Y")

    def amc_prev(mid):
        if mid in seen:
            return []
        try:
            j = get(f"https://www.amfiindia.com/api/populate-te-rdata-revised?MF_ID={mid}&Month={prev}&page=1&pageSize=100", tries=3).json()
        except Exception:
            return []
        rows = j.get("data", []); n = j.get("meta", {}).get("pageCount", 0)
        for i in range(2, n + 1):
            try:
                rows += get(f"https://www.amfiindia.com/api/populate-te-rdata-revised?MF_ID={mid}&Month={prev}&page={i}&pageSize=100", tries=5).json()["data"]
            except Exception:
                pass
        return rows
    with ThreadPoolExecutor(3) as ex:
        for rows in ex.map(amc_prev, range(1, 100)):
            if rows:
                print("ter fallback", rows[0].get("Scheme_Name", "")[:30], len(rows), flush=True)
            out += rows
    latest = {}
    for x in out:
        k = x["NSDLSchemeCode"]
        if k not in latest or x["TER_Date"] > latest[k]["TER_Date"]:
            latest[k] = x
    json.dump(list(latest.values()), open(os.path.join(CACHE, "ter.json"), "w"))
    return len(latest)


def fetch_aum():
    base = "https://www.amfiindia.com/api/average-aum-schemewise?strType=Categorywise"
    yrs = get(base + "&fyId=1").json()
    periods = yrs["data"]["periods"]
    fy, pid = 1, periods[0]["id"]
    d = get(base + f"&fyId={fy}&periodId={pid}", timeout=90).json()
    rows = []
    for g in d["data"]:
        for s in g["schemes"]:
            a = s["AverageAumForTheMonth"]
            rows.append(dict(code=s["AMFI_Code"], name=s["SchemeNAVName"], house=g["Mfname"], cat=g["SchemeCat_Desc"],
                             aum_lakh=(a.get("ExcludingFundOfFundsDomesticButIncludingFundOfFundsOverseas") or 0) + (a.get("FundOfFundsDomestic") or 0)))
    json.dump(dict(period=periods[0]["period"], fy=yrs["data"]["financial_year"], rows=rows),
              open(os.path.join(CACHE, "aum.json"), "w"))
    return len(rows)


def main():
    t = get("https://portal.amfiindia.com/spages/NAVAll.txt").content.decode("utf-8", errors="replace")
    open(os.path.join(CACHE, "NAVAll.txt"), "w").write(t)
    rows = parse_navall(t)
    cutoff = dt.date.today() - dt.timedelta(days=12)
    uni = []
    for r in rows:
        if r["type"] != "Open Ended Schemes" or not is_direct_growth(r):
            continue
        try:
            d = dt.datetime.strptime(r["date"], "%d-%b-%Y").date()
        except Exception:
            continue
        if d < cutoff:
            continue
        uni.append(r)
    json.dump(uni, open(os.path.join(CACHE, "universe.json"), "w"))
    print("universe", len(uni), flush=True)
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("MF_THREADS", "12"))) as ex:
        for i, (c, s) in enumerate(ex.map(fetch_nav, [r["code"] for r in uni])):
            ok += s
            if i % 200 == 0:
                print("nav", i, ok, flush=True)
    print("nav ok", ok, flush=True)
    for nm, fn in (("ter", fetch_ter), ("aum", fetch_aum)):
        fp = os.path.join(CACHE, nm + ".json")
        if os.path.exists(fp) and time.time() - os.path.getmtime(fp) < 6 * 86400 and not os.environ.get("MF_FORCE"):
            print(nm, "cache fresh, skipped", flush=True); continue
        try:
            print(nm, fn(), flush=True)
        except Exception as e:
            print(nm, "FAILED (previous cache kept)", e, flush=True)


if __name__ == "__main__":
    main()
