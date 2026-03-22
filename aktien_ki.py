import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="KI-Aktien-Terminal Pro", page_icon="📈", layout="wide")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ── WKN → TICKER ─────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0"}

def _yahoo_search(query):
    for host in ["query1", "query2"]:
        try:
            url = (
                f"https://{host}.finance.yahoo.com/v1/finance/search"
                f"?q={query}&lang=de-DE&region=DE&quotesCount=8&newsCount=0"
            )
            r = requests.get(url, timeout=5, headers=HEADERS)
            quotes = r.json().get("quotes", [])
            if quotes:
                return quotes
        except Exception:
            continue
    return []

def _best_ticker(quotes):
    for suffix in [".DE", ".F", ".MU", ".BE"]:
        for q in quotes:
            if q.get("symbol", "").endswith(suffix):
                return q.get("symbol"), q.get("longname") or q.get("shortname")
    if quotes:
        q = quotes[0]
        return q.get("symbol"), q.get("longname") or q.get("shortname")
    return None, None

@st.cache_data(ttl=86400)
def wkn_zu_ticker(wkn):
    """
    Löst eine WKN in ein Ticker-Symbol auf.
    Strategie 1: Direkte WKN-Suche bei Yahoo Finance
    Strategie 2: OpenFIGI API (kostenfrei, kein Key nötig)
    Strategie 3: ISIN-Ableitung (DE000 + WKN)
    """
    wkn = wkn.strip().upper()

    # Strategie 1: Yahoo direkt
    quotes = _yahoo_search(wkn)
    ticker, name = _best_ticker(quotes)
    if ticker:
        return ticker, name

    # Strategie 2: OpenFIGI
    try:
        r = requests.post(
            "https://api.openfigi.com/v3/mapping",
            json=[{"idType": "ID_WERTPAPIER", "idValue": wkn, "exchCode": "GR"}],
            timeout=6, headers=HEADERS
        )
        results = r.json()
        if results and isinstance(results, list) and results[0].get("data"):
            d    = results[0]["data"][0]
            name = d.get("name", wkn)
            raw  = d.get("ticker", "")
            if raw:
                candidate = raw + ".DE"
                test = yf.Ticker(candidate).history(period="5d")
                if not test.empty:
                    return candidate, name
                quotes2 = _yahoo_search(raw)
                t2, n2  = _best_ticker(quotes2)
                if t2:
                    return t2, n2 or name
    except Exception:
        pass

    # Strategie 3: ISIN-Ableitung
    for isin in ["DE000" + wkn, "DE" + wkn]:
        try:
            quotes3 = _yahoo_search(isin)
            t3, n3  = _best_ticker(quotes3)
            if t3:
                return t3, n3
        except Exception:
            pass

    return None, None

# ── DATEN ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def lade_daten(ticker, zeitraum):
    try:
        obj = yf.Ticker(ticker)
        df = obj.history(period=zeitraum, auto_adjust=True)
        if df.empty or len(df) < 100:
            return None, None, {}
        df.index = df.index.tz_localize(None)
        divs = obj.dividends
        if not divs.empty:
            divs.index = divs.index.tz_localize(None)
        info = obj.info
        meta = {
            "name": info.get("longName", ticker),
            "div_yield": info.get("dividendYield", 0) or 0,
            "sector": info.get("sector", "-"),
        }
        return df, divs, meta
    except Exception:
        return None, None, {}

# ── ANALYSTEN ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def lade_analysten(ticker):
    try:
        obj  = yf.Ticker(ticker)
        info = obj.info
        # Kursziel & Empfehlung
        ziel      = info.get("targetMeanPrice")
        ziel_low  = info.get("targetLowPrice")
        ziel_high = info.get("targetHighPrice")
        empf      = info.get("recommendationKey", "")  # buy, hold, sell, ...
        n_analyst = info.get("numberOfAnalystOpinions", 0)

        # Ratings-Verteilung
        ratings = {
            "Starker Kauf": info.get("recommendationMean", None),
        }

        # Aktuelle Ratings-Tabelle (letzte Einträge)
        try:
            rec_df = obj.recommendations
            if rec_df is not None and not rec_df.empty:
                rec_df = rec_df.tail(10).copy()
            else:
                rec_df = None
        except Exception:
            rec_df = None

        return {
            "ziel":      ziel,
            "ziel_low":  ziel_low,
            "ziel_high": ziel_high,
            "empf":      empf,
            "n":         n_analyst,
            "rec_df":    rec_df,
        }
    except Exception:
        return {}

# ── INDIKATOREN ───────────────────────────────────────────────────────────────
def indikatoren(df):
    c = df["Close"].copy()
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_Sig"]
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_U"] = sma20 + 2 * std20
    df["BB_L"] = sma20 - 2 * std20
    df["BB_W"] = (df["BB_U"] - df["BB_L"]) / sma20
    df["BB_P"] = (c - df["BB_L"]) / (df["BB_U"] - df["BB_L"] + 1e-9)
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-c.shift()).abs(),
                    (df["Low"]-c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(com=13, adjust=False).mean()
    df["VR"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)
    df["R1"] = c.pct_change(1)
    df["R5"] = c.pct_change(5)
    df["Target"] = np.where(c.shift(-1) > c, 1, 0)
    return df.dropna()

FCOLS = ["RSI","MACD_H","BB_P","BB_W","VR","R1","R5","TR"]

def features(df):
    df = df.copy()
    df["TR"] = (df["SMA20"] > df["SMA50"]).astype(int)
    for col in FCOLS:
        df[col+"_l"] = df[col].shift(1)
    lag = [c+"_l" for c in FCOLS]
    return df.dropna(), lag

# ── MODELL ────────────────────────────────────────────────────────────────────
def modell(df, lag):
    X, y = df[lag], df["Target"]
    split = int(len(X) * 0.8)
    base = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    m = CalibratedClassifierCV(base, cv=3, method="isotonic")
    m.fit(X.iloc[:split], y.iloc[:split])
    acc = accuracy_score(y.iloc[split:], m.predict(X.iloc[split:]))
    prob = m.predict_proba(X.iloc[[-1]])[0][1]
    imp = None
    try:
        imp = dict(zip(FCOLS, m.calibrated_classifiers_[0].estimator.feature_importances_))
    except Exception:
        pass
    return prob, acc, imp, m, X

# ── TREFFERQUOTE ──────────────────────────────────────────────────────────────
def trefferquote(m, X, df, h=20):
    probs = m.predict_proba(X)[:, 1]
    sigs = np.where(probs > 0.55, "KAUFEN", np.where(probs < 0.45, "VERKAUFEN", "HALTEN"))
    closes = df["Close"].values
    res = {"KAUFEN": [], "VERKAUFEN": []}
    for i, sig in enumerate(sigs):
        if sig == "HALTEN": continue
        if i + h >= len(closes): continue
        r = (closes[i+h] - closes[i]) / closes[i] * 100
        res[sig].append(r)
    out = {}
    for sig in ["KAUFEN", "VERKAUFEN"]:
        w = res[sig]
        if not w:
            out[sig] = {"n": 0, "pct": 0, "ret": 0}
            continue
        tr = sum(1 for r in w if (r > 0 if sig == "KAUFEN" else r < 0))
        out[sig] = {"n": len(w), "pct": tr/len(w)*100, "ret": float(np.mean(w))}
    return out

# ── DIVIDENDEN ────────────────────────────────────────────────────────────────
def div_info(divs, preis, n_aktien, meta_yield, manuell_div=0.0):
    heute = pd.Timestamp.now()
    if manuell_div > 0:
        pa  = manuell_div
        yld = pa / preis * 100 if preis > 0 else 0
        iv  = 12
        naechste_str = "unbekannt"
        if divs is not None and not divs.empty:
            nz_hist = len(divs[divs.index >= heute - pd.DateOffset(years=1)])
            iv = 12 if nz_hist <= 1 else (6 if nz_hist <= 2 else (3 if nz_hist <= 4 else 1))
            letzte = divs.index[-1]
            naechste = letzte + pd.DateOffset(months=iv)
            naechste_str = naechste.strftime("%d.%m.%Y")
        dpz = pa
        z6  = 1 if iv >= 6 else (2 if iv == 3 else 6)
        z1  = 12 // iv if iv > 0 else 1
        g6  = dpz * (z6 / z1) * n_aktien
        g1  = dpz * n_aktien
        return {
            "yield": yld, "pa": pa, "g6": g6, "g1": g1,
            "r6": g6 / (preis * n_aktien) * 100 if preis > 0 and n_aktien > 0 else 0,
            "r1": yld, "z6": z6, "z1": z1,
            "iv": {1:"monatlich",3:"quartalsweise",6:"halbjährlich",12:"jährlich"}.get(iv,"-"),
            "next": naechste_str, "quelle": "⚠️ Manuell eingegeben",
        }
    if divs is not None and not divs.empty:
        recent = divs[divs.index >= heute - pd.DateOffset(years=1)]
        pa  = float(recent.sum()) if not recent.empty else 0
        yld = pa / preis * 100 if preis > 0 else meta_yield * 100
        nz  = len(recent)
        letzte  = divs.index[-1]
        iv  = 12 if nz <= 1 else (6 if nz <= 2 else (3 if nz <= 4 else 1))
        naechste = letzte + pd.DateOffset(months=iv)
        dpz = pa / nz if nz > 0 else 0
        z6, z1 = 0, 0
        t = naechste
        while t <= heute + pd.DateOffset(months=6):
            z6 += 1; t += pd.DateOffset(months=iv)
        t = naechste
        while t <= heute + pd.DateOffset(years=1):
            z1 += 1; t += pd.DateOffset(months=iv)
        return {
            "yield": yld, "pa": pa,
            "g6": dpz * z6 * n_aktien, "g1": dpz * z1 * n_aktien,
            "r6": dpz * z6 / preis * 100 if preis > 0 else 0,
            "r1": dpz * z1 / preis * 100 if preis > 0 else 0,
            "z6": z6, "z1": z1,
            "iv": {1:"monatlich",3:"quartalsweise",6:"halbjährlich",12:"jährlich"}.get(iv,"-"),
            "next": naechste.strftime("%d.%m.%Y"), "quelle": "📡 yfinance (kann veraltet sein)",
        }
    yld = meta_yield * 100
    pa  = preis * meta_yield
    return {"yield": yld, "pa": pa, "g6": 0, "g1": pa * n_aktien,
            "r6": 0, "r1": yld, "z6": 0, "z1": 1 if yld > 0 else 0,
            "iv": "jährlich", "next": "unbekannt", "quelle": "📡 yfinance (kann veraltet sein)"}

# ── MONTE CARLO ───────────────────────────────────────────────────────────────
def monte_carlo(df, tage, n=1000, seed=42, div_pa=0):
    np.random.seed(seed)
    lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    mu = float(lr.mean()) + div_pa / 252
    sigma = float(lr.std())
    S0 = float(df["Close"].iloc[-1])
    r = np.random.normal(mu - 0.5*sigma**2, sigma, (n, tage))
    p = S0 * np.exp(np.cumsum(r, axis=1))
    p = np.hstack([np.full((n, 1), S0), p])
    return p, mu, sigma

def kz(paths, kauf):
    ep = paths[:, -1]
    return {
        "p5":   float(np.percentile(ep, 5)),
        "p25":  float(np.percentile(ep, 25)),
        "p50":  float(np.percentile(ep, 50)),
        "p75":  float(np.percentile(ep, 75)),
        "p95":  float(np.percentile(ep, 95)),
        "gwkt": float(np.mean(ep > kauf)),
        "ret":  float((np.percentile(ep, 50) - kauf) / kauf * 100),
    }

# ── BOOTSTRAP SIMULATION ─────────────────────────────────────────────────────
def bootstrap(df, tage, n=1000, seed=42, div_pa=0):
    """
    Zieht echte historische Tagesrenditen mit Zurücklegen neu zusammen.
    Bildet Fat Tails und Crashs realistischer ab als Normalverteilung.
    """
    np.random.seed(seed)
    lr = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
    div_daily = div_pa / 252
    S0 = float(df["Close"].iloc[-1])
    idx = np.random.randint(0, len(lr), size=(n, tage))
    r   = lr[idx] + div_daily
    p   = S0 * np.exp(np.cumsum(r, axis=1))
    p   = np.hstack([np.full((n, 1), S0), p])
    return p

# ── GARCH(1,1) SIMULATION ─────────────────────────────────────────────────────
def garch(df, tage, n=1000, seed=42, div_pa=0):
    """
    GARCH(1,1): Volatilität ist zeitvariabel — nach Schocks steigt sie, in
    ruhigen Phasen sinkt sie wieder. Realistischer als konstante Vola.
    """
    np.random.seed(seed)
    lr   = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
    mu   = float(lr.mean()) + div_pa / 252
    # GARCH-Parameter via Momentenschätzung
    var  = float(lr.var())
    omega = var * 0.05
    alpha = 0.10   # Gewicht letztes Schock²
    beta  = 0.85   # Gewicht letzte Varianz
    S0   = float(df["Close"].iloc[-1])
    paths = np.zeros((n, tage + 1))
    paths[:, 0] = S0
    h = np.full(n, var)   # Startvarianz für alle Pfade
    for t in range(tage):
        eps  = np.random.standard_normal(n)
        ret  = mu - 0.5 * h + np.sqrt(h) * eps
        paths[:, t+1] = paths[:, t] * np.exp(ret)
        h = omega + alpha * (ret - mu)**2 + beta * h
        h = np.clip(h, 1e-9, None)
    return paths

# ── SIMULATIONS-VERGLEICH ─────────────────────────────────────────────────────
def sim_vergleich(mc_paths, bs_paths, gc_paths, kauf, preis):
    """Erstellt einen Vergleichs-Chart aller drei Simulationsmethoden."""
    methods = [
        ("Monte Carlo", mc_paths, "#4a9eff", "rgba(74,158,255,0.12)"),
        ("Bootstrap",   bs_paths, "#f0a000", "rgba(240,160,0,0.12)"),
        ("GARCH(1,1)",  gc_paths, "#00c87a", "rgba(0,200,122,0.12)"),
    ]
    fig = go.Figure()
    for name, paths, color, fill_color in methods:
        ep = paths[:, -1]
        gwkt = float(np.mean(ep > kauf)) * 100
        med  = float(np.percentile(ep, 50))
        ret  = (med - kauf) / kauf * 100
        label = f"{name}: {gwkt:.0f}% Gewinn | Median {ret:+.1f}%"
        x_days = list(range(paths.shape[1]))
        p5  = list(np.percentile(paths, 5,  axis=0))
        p95 = list(np.percentile(paths, 95, axis=0))
        p50 = list(np.percentile(paths, 50, axis=0))
        fig.add_trace(go.Scatter(
            x=x_days, y=p95, showlegend=False,
            line=dict(color="rgba(0,0,0,0)", width=0)))
        fig.add_trace(go.Scatter(
            x=x_days, y=p5, name=label,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty", fillcolor=fill_color))
        fig.add_trace(go.Scatter(
            x=x_days, y=p50, name=label + " (Median)",
            line=dict(color=color, width=2, dash="dot")))
    fig.add_hline(y=kauf, line_dash="dash", line_color="rgba(255,77,106,0.6)",
                  annotation_text="Kaufkurs")
    fig.update_layout(
        template="plotly_dark", height=420,
        title="Simulationsvergleich — 90%-Band + Median",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        xaxis_title="Handelstage", yaxis_title="Kurs (EUR)",
    )
    return fig

# ── SIGNAL ────────────────────────────────────────────────────────────────────
def signal(prob, rsi, sma20, sma50, macd_h):
    g, s = [], 0.0
    s += (prob - 0.5) * 2 * 0.5
    g.append(f"ML: {prob*100:.1f}%")
    if rsi < 30:   s += 0.25; g.append(f"RSI {rsi:.0f} - ueberverkauft")
    elif rsi > 70: s -= 0.25; g.append(f"RSI {rsi:.0f} - ueberkauft")
    elif rsi < 45: s -= 0.10; g.append(f"RSI {rsi:.0f} - leicht baerisch")
    elif rsi > 55: s += 0.10; g.append(f"RSI {rsi:.0f} - leicht bullisch")
    else:          g.append(f"RSI {rsi:.0f} - neutral")
    if sma20 > sma50: s += 0.15; g.append("SMA20 > SMA50 - Aufwaertstrend")
    else:             s -= 0.15; g.append("SMA20 < SMA50 - Abwaertstrend")
    if macd_h > 0: s += 0.10; g.append("MACD positiv")
    else:          s -= 0.10; g.append("MACD negativ")
    sig = "KAUFEN" if s > 0.15 else ("VERKAUFEN" if s < -0.15 else "HALTEN")
    return sig, min(abs(s)/0.55, 1.0), g, s

# ── GESAMTFAZIT ───────────────────────────────────────────────────────────────
def gesamtfazit(name, ticker, preis, inv, nakt, gwkt6, gwkt1,
                g6tot, g1tot, sl, rp, rvmax, k6, k1, sig, ana=None):
    xdown6_pct = abs(round((k6["p5"] - preis) / preis * 100))
    xdown1_pct = abs(round((k1["p5"] - preis) / preis * 100))
    xup6_pct   = round((k6["p95"] - preis) / preis * 100)
    xup1_pct   = round((k1["p95"] - preis) / preis * 100)

    sig_text = {
        "KAUFEN":    "spricht das Gesamtbild klar für einen Einstieg",
        "HALTEN":    "empfiehlt sich aktuell eine abwartende Haltung",
        "VERKAUFEN": "deuten die Signale eher auf einen Ausstieg hin",
    }.get(sig, "ist das Bild gemischt")

    if gwkt1 >= 80:
        risiko_einschaetzung = "Das Chance-Risiko-Verhältnis fällt deutlich positiv aus — die Simulation zeigt ein klares Übergewicht profitabler Szenarien."
    elif gwkt1 >= 65:
        risiko_einschaetzung = "Das Chance-Risiko-Verhältnis ist insgesamt ausgeglichen, mit einem leichten Vorteil auf der Gewinnseite."
    else:
        risiko_einschaetzung = "Das Chance-Risiko-Verhältnis verdient besondere Aufmerksamkeit — mehr als ein Drittel der Szenarien endet im Minus."

    # ── Analysten-Divergenz ──────────────────────────────────────────────────
    analyst_block = ""
    if ana and ana.get("ziel") and preis > 0:
        ziel    = ana["ziel"]
        upside  = (ziel - preis) / preis * 100
        sim_ret = g1tot  # Simulation 1-Jahres-Erwartung

        differenz = abs(upside - sim_ret)

        if differenz >= 15:
            richtung = "deutlich optimistischer" if upside > sim_ret else "deutlich pessimistischer"
            empf_map = {
                "strong_buy": "Starker Kauf", "buy": "Kaufen",
                "hold": "Halten", "underperform": "Unterperformen", "sell": "Verkaufen",
            }
            empf_txt = empf_map.get(ana.get("empf", ""), "keine klare Empfehlung")

            if upside > sim_ret:
                # Analysten bullisher als Simulation
                if gwkt1 < 55:
                    trendwende = (
                        f"\n\n**⚠️ Hinweis — Analysten vs. Simulation:** "
                        f"Die {ana.get('n', '')} befragten Analysten sehen ein durchschnittliches Kursziel "
                        f"von **{ziel:.0f} EUR (+{upside:.0f} %)** und empfehlen **{empf_txt}** — "
                        f"die Simulation hingegen zeigt nur **{sim_ret:+.1f} %** Gesamtertrag bei "
                        f"lediglich **{gwkt1} % Gewinnwahrscheinlichkeit**. "
                        f"Diese Diskrepanz von **{differenz:.0f} Prozentpunkten** entsteht oft, wenn "
                        f"eine Aktie fundamental unterbewertet ist, der Markt den Analysten aber (noch) nicht folgt. "
                        f"Ein technischer Trendwechsel — SMA20 über SMA50 — wäre das erste Signal, "
                        f"dass sich Kurs und Analystenmeinung annähern. Bis dahin ist Abwarten ratsam."
                    )
                else:
                    trendwende = (
                        f"\n\n**ℹ️ Hinweis — Analysten vs. Simulation:** "
                        f"Die Analysten sehen mit **{ziel:.0f} EUR** noch deutlich mehr Potenzial "
                        f"als die historische Simulation ({sim_ret:+.1f} %). "
                        f"Das kann bedeuten, dass fundamentale Katalysatoren bevorstehen, "
                        f"die in den Kursdaten noch nicht sichtbar sind."
                    )
            else:
                # Simulation bullisher als Analysten
                trendwende = (
                    f"\n\n**⚠️ Hinweis — Analysten vs. Simulation:** "
                    f"Interessanterweise ist hier die Simulation mit **{sim_ret:+.1f} %** "
                    f"optimistischer als das Analysten-Kursziel von **{ziel:.0f} EUR ({upside:+.0f} %)**. "
                    f"Das deutet auf eine Aktie hin, die historisch stark gelaufen ist — "
                    f"Analysten sehen möglicherweise weniger Spielraum nach oben als die reine Trendfortschreibung."
                )
            analyst_block = trendwende

    return f"""
**Gesamtfazit – {name} ({ticker})**

Wer heute **{nakt} Aktien** zu je **{preis:.2f} EUR** kauft — also **{inv:,.0f} EUR** investiert — \
hält laut Monte-Carlo-Simulation in **{gwkt6} % aller Szenarien** nach sechs Monaten mehr als heute. \
Der erwartete Gesamtertrag (Kurs + Dividende) liegt bei **+{g6tot} %**. \
Im günstigen Fall wächst das Investment um **+{xup6_pct} %**, im schlechten Szenario \
liegt der Extremverlust ohne Absicherung bei **–{xdown6_pct} %**.

Hältst du ein volles Jahr durch, steigt die Gewinnwahrscheinlichkeit weiter auf **{gwkt1} %**. \
Der erwartete Gesamtertrag klettert auf **+{g1tot} %**. \
Selbst im ungünstigsten Extremfall ohne Stop-Loss wären maximal **–{xdown1_pct} %** \
möglich — nach oben hingegen bis zu **+{xup1_pct} %**.

Mit dem gesetzten Stop-Loss bei **{sl:.2f} EUR (–{rp} %)** \
ist dein maximaler Verlust auf **{rvmax:,.0f} EUR** begrenzt — unabhängig davon, was der Markt macht.

**Fazit:** Bei {name} {sig_text}. {risiko_einschaetzung} \
Solange der Stop-Loss konsequent sitzt, bleibt das Risiko kalkulierbar.{analyst_block}
""".strip()

# ── CHARTS ────────────────────────────────────────────────────────────────────
def chart_prognose(df, p6, p1, kauf, ticker):
    heute = df.index[-1]
    d6 = [heute + timedelta(days=i) for i in range(p6.shape[1])]
    d1 = [heute + timedelta(days=i) for i in range(p1.shape[1])]
    fig = go.Figure()
    hist = df["Close"].tail(252)
    fig.add_trace(go.Scatter(x=list(hist.index), y=list(hist.values),
                             name="Historisch", line=dict(color="#c8cdd8", width=1.8)))
    pc = lambda p, q: list(np.percentile(p, q, axis=0))
    fig.add_trace(go.Scatter(x=d1, y=pc(p1, 95), showlegend=False,
                             line=dict(color="rgba(0,0,0,0)", width=0)))
    fig.add_trace(go.Scatter(x=d1, y=pc(p1, 5), name="90%-Band",
                             line=dict(color="rgba(0,0,0,0)", width=0),
                             fill="tonexty", fillcolor="rgba(74,158,255,0.10)"))
    fig.add_trace(go.Scatter(x=d1, y=pc(p1, 75), showlegend=False,
                             line=dict(color="rgba(0,0,0,0)", width=0)))
    fig.add_trace(go.Scatter(x=d1, y=pc(p1, 25), name="50%-Band",
                             line=dict(color="rgba(0,0,0,0)", width=0),
                             fill="tonexty", fillcolor="rgba(74,158,255,0.18)"))
    fig.add_trace(go.Scatter(x=d1, y=pc(p1, 50), name="Median 1J",
                             line=dict(color="#4a9eff", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=d6, y=pc(p6, 50), name="Median 6M",
                             line=dict(color="#f0a000", width=2, dash="dot")))
    fig.add_hline(y=kauf, line_dash="dash", line_color="rgba(255,77,106,0.7)")
    fig.update_layout(template="plotly_dark", height=450,
                      title=ticker + " - Monte Carlo Prognose",
                      paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14")
    return fig

def chart_tech(df, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=list(df.index), open=list(df["Open"]),
                                 high=list(df["High"]), low=list(df["Low"]),
                                 close=list(df["Close"]), name="Kurs",
                                 increasing_line_color="#00c87a",
                                 decreasing_line_color="#ff4d6a"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df["SMA20"]),
                             name="SMA20", line=dict(color="#f0a000", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df["SMA50"]),
                             name="SMA50", line=dict(color="#4a9eff", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df["RSI"]),
                             name="RSI", line=dict(color="#c084fc", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,100,100,0.4)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(100,200,100,0.4)", row=2, col=1)
    colors = ["#00c87a" if v >= 0 else "#ff4d6a" for v in df["MACD_H"]]
    fig.add_trace(go.Bar(x=list(df.index), y=list(df["MACD_H"]),
                         name="MACD Hist", marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df["MACD"]),
                             name="MACD", line=dict(color="#f0a000", width=1.2)), row=3, col=1)
    fig.update_layout(template="plotly_dark", height=560,
                      xaxis_rangeslider_visible=False,
                      paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14")
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📈 KI-Aktien-Terminal Pro")
st.caption("Technische Analyse · KI-Signal · Monte-Carlo-Prognose · Dividenden")
st.warning("Nur für Bildungszwecke — keine Finanzberatung.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Einstellungen")

wkn_input = st.sidebar.text_input(
    "WKN eingeben",
    placeholder="z.B. 843002",
    max_chars=6,
    help="6-stellige Wertpapierkennnummer, z.B. 843002 für Munich Re"
).strip().upper()

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation**")
n_sim = st.sidebar.select_slider("Simulationen (Monte Carlo)", [500, 1000, 2000, 5000], value=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risiko**")
kapital = st.sidebar.number_input("Kapital (EUR)", min_value=100, value=5000, step=100)
rp      = st.sidebar.number_input("Max. Verlust (%)", min_value=1, max_value=50, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dividende (optional)**")
st.sidebar.caption("Falls yfinance veraltet ist: aktuelle Dividende/Aktie pro Jahr eingeben.")
manuell_div = st.sidebar.number_input(
    "Dividende p.a. (EUR/Aktie)",
    min_value=0.0, value=0.0, step=0.10, format="%.2f",
    help="z.B. 24.00 für Munich Re 2025. Leer lassen = automatisch."
)

st.sidebar.markdown("---")
start = st.sidebar.button("Analyse starten", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Watchlist**")
for w in st.session_state.watchlist:
    st.sidebar.markdown(f"- {w}")
if st.sidebar.button("+ Aktuelle WKN merken") and wkn_input:
    if wkn_input not in st.session_state.watchlist:
        st.session_state.watchlist.append(wkn_input)

# ── Analyse ───────────────────────────────────────────────────────────────────
if start:
    if not wkn_input or len(wkn_input) < 4:
        st.error("Bitte eine gültige WKN eingeben (z.B. 843002 für Munich Re).")
        st.stop()

    with st.spinner(f"Suche Ticker für WKN {wkn_input} ..."):
        ticker, gefundener_name = wkn_zu_ticker(wkn_input)

    if not ticker:
        st.error(f"Kein Ticker für WKN **{wkn_input}** gefunden. Bitte WKN prüfen.")
        st.stop()

    st.info(f"WKN **{wkn_input}** → **{ticker}** ({gefundener_name})")

    with st.spinner("Lade Marktdaten..."):
        df_raw, divs, meta = lade_daten(ticker, "5y")

    if df_raw is None:
        st.error(f"Keine Kursdaten für {ticker} verfügbar.")
        st.stop()

    with st.spinner("Berechne Indikatoren..."):
        df = indikatoren(df_raw.copy())

    with st.spinner("Trainiere KI-Modell..."):
        df_f, lag = features(df)
        prob, acc, imp, m, X = modell(df_f, lag)

    with st.spinner("Trefferquote..."):
        tq = trefferquote(m, X, df_f)

    with st.spinner("Monte Carlo Simulation..."):
        preis  = float(df["Close"].iloc[-1])
        rsi    = float(df["RSI"].iloc[-1])
        sma20  = float(df["SMA20"].iloc[-1])
        sma50  = float(df["SMA50"].iloc[-1])
        macd_h = float(df["MACD_H"].iloc[-1])
        bb_u   = float(df["BB_U"].iloc[-1])
        bb_l   = float(df["BB_L"].iloc[-1])

        sig, konf, gruende, score = signal(prob, rsi, sma20, sma50, macd_h)
        emoji = {"KAUFEN": "🟢", "VERKAUFEN": "🔴", "HALTEN": "🟡"}[sig]

        sl    = preis * (1 - rp / 100)
        vpa   = preis - sl
        mveur = kapital * (rp / 100)
        nris  = int(mveur / vpa) if vpa > 0 else 0
        nkap  = int(kapital / preis)
        nakt  = min(nris, nkap)
        inv   = nakt * preis
        rvmax = nakt * vpa

        dv = div_info(divs, preis, nakt, meta.get("div_yield", 0), manuell_div)
        p6, mu, sigma = monte_carlo(df, 126, n_sim, div_pa=dv["yield"]/100)
        p1, _,  _     = monte_carlo(df, 252, n_sim, div_pa=dv["yield"]/100)
        k6 = kz(p6, preis)
        k1 = kz(p1, preis)
        # Bootstrap
        bs6 = bootstrap(df, 126, n_sim, seed=7,  div_pa=dv["yield"]/100)
        bs1 = bootstrap(df, 252, n_sim, seed=7,  div_pa=dv["yield"]/100)
        kb6 = kz(bs6, preis)
        kb1 = kz(bs1, preis)
        # GARCH
        gc6 = garch(df, 126, n_sim, seed=13, div_pa=dv["yield"]/100)
        gc1 = garch(df, 252, n_sim, seed=13, div_pa=dv["yield"]/100)
        kg6 = kz(gc6, preis)
        kg1 = kz(gc1, preis)

    # ── Header ────────────────────────────────────────────────────────────────
    st.subheader(meta.get("name", ticker) + " (" + ticker + ")")
    st.caption("Sektor: " + meta.get("sector", "-"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Preis",        f"{preis:.2f}")
    c2.metric("RSI",          f"{rsi:.1f}")
    c3.metric("KI-Konfidenz", f"{konf*100:.0f}%")
    c4.metric("Accuracy",     f"{acc*100:.0f}%")
    c5.metric("Div.-Rendite", f"{dv['yield']:.1f}%", help=dv.get("quelle",""))

    if manuell_div > 0:
        st.info(f"⚠️ Dividende manuell: **{manuell_div:.2f} EUR/Aktie** "
                f"({dv['yield']:.2f}% bei aktuellem Kurs). yfinance-Wert ignoriert.")

    st.markdown("---")

    # ── Signal + Risiko ───────────────────────────────────────────────────────
    col_sig, col_ris = st.columns([3, 2])
    with col_sig:
        st.subheader(emoji + " Signal: " + sig)
        st.caption(f"Konfidenz {konf*100:.0f}%  |  ML {prob*100:.0f}%")
        t = tq.get(sig, {})
        if t and t.get("n", 0) > 5:
            st.markdown(f"**Trefferquote:** {t['pct']:.0f}% ({t['n']} Faelle)  "
                        f"  |  Avg. Rendite: **{t['ret']:+.1f}%**")
        else:
            st.info("Zu wenige Signale fuer Trefferquote.")
        st.markdown("**Indikatoren:**")
        for g in gruende:
            bull = any(k in g for k in ["ueberverkauft","bullisch","Aufwaerts","positiv"])
            bear = any(k in g for k in ["ueberkauft","baerisch","Abwaerts","negativ"])
            ico  = "🟢" if bull else ("🔴" if bear else "🟡")
            st.markdown(ico + " " + g)

    with col_ris:
        st.markdown("**Risiko-Plan**")
        r1c, r2c = st.columns(2)
        r1c.metric("Aktien",       str(nakt) + " Stk")
        r2c.metric("Investition",  f"{inv:.0f} EUR")
        r1c.metric("Stop-Loss",    f"{sl:.2f}")
        r2c.metric("Max. Verlust", f"{rvmax:.0f} EUR")

    st.markdown("---")

    # ── Dividenden ────────────────────────────────────────────────────────────
    if dv["yield"] > 0:
        st.subheader("💰 Dividenden")
        d1c, d2c, d3c, d4c = st.columns(4)
        d1c.metric("Rendite p.a.",  f"{dv['yield']:.1f}%")
        d2c.metric("Pro Aktie",     f"{dv['pa']:.2f} EUR")
        d3c.metric("Deine Div. 6M", f"{dv['g6']:.0f} EUR")
        d4c.metric("Deine Div. 1J", f"{dv['g1']:.0f} EUR")
        st.info(f"Rhythmus: {dv['iv']}  |  Nächste Zahlung ca. {dv['next']}  |  "
                f"6M: {dv['z6']}x  |  1J: {dv['z1']}x  |  Quelle: {dv.get('quelle','')}")

    st.markdown("---")

    # ── Analysten ─────────────────────────────────────────────────────────────
    ana = lade_analysten(ticker)
    if ana and ana.get("ziel"):
        st.subheader("🎯 Analystenbewertungen & Kursziele")

        # Empfehlung leserlich machen
        empf_map = {
            "strong_buy":  ("Starker Kauf",  "🟢"),
            "buy":         ("Kaufen",         "🟢"),
            "hold":        ("Halten",         "🟡"),
            "underperform":("Unterperformen", "🔴"),
            "sell":        ("Verkaufen",      "🔴"),
        }
        empf_txt, empf_ico = empf_map.get(ana.get("empf",""), ("Keine Angabe", "⚪"))
        upside = ((ana["ziel"] - preis) / preis * 100) if preis > 0 else 0

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Konsensus", f"{empf_ico} {empf_txt}")
        a2.metric("Analysten", f"{ana['n']} Meinungen")
        a3.metric("Ø Kursziel", f"{ana['ziel']:.2f} EUR",
                  f"{upside:+.1f}% zum aktuellen Kurs")
        a4.metric("Kursziel-Spanne",
                  f"{ana['ziel_low']:.0f} – {ana['ziel_high']:.0f} EUR")

        # Fortschrittsbalken: Kurs vs. Kursziel
        if ana.get("ziel_low") and ana.get("ziel_high"):
            spanne = ana["ziel_high"] - ana["ziel_low"]
            if spanne > 0:
                pos_pct = max(0.0, min(1.0, (preis - ana["ziel_low"]) / spanne))
                st.caption(f"Kursziel-Spanne: {ana['ziel_low']:.0f} EUR ◀─── aktuell: {preis:.2f} EUR ───▶ {ana['ziel_high']:.0f} EUR")
                st.progress(pos_pct)

        # Letzte Einzelratings
        if ana.get("rec_df") is not None:
            with st.expander("Letzte Einzelratings anzeigen"):
                st.dataframe(ana["rec_df"], use_container_width=True)
    else:
        st.info("Keine Analystendaten für diese Aktie verfügbar.")

    st.markdown("---")

    # ── Prognose ──────────────────────────────────────────────────────────────
    st.subheader("🔭 Langfrist-Prognose")
    st.caption(f"Drift: {mu*252*100:.1f}% p.a.  |  Vola: {sigma*252**0.5*100:.1f}% p.a.")

    g6tot = round(k6["ret"] + dv["r6"], 1)
    g1tot = round(k1["ret"] + dv["r1"], 1)
    gwkt6 = round(k6["gwkt"] * 100)
    gwkt1 = round(k1["gwkt"] * 100)

    pr1, pr2 = st.columns(2)
    with pr1:
        st.markdown("#### In 6 Monaten — kaufst du heute")
        g6col = "normal" if g6tot >= 0 else "inverse"
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("Erwarteter Kurs", f"{k6['p50']:.2f}",
                   f"{k6['ret']:+.1f}% Kursgewinn",
                   help="Median — in 50% der Szenarien liegt er höher")
        m2c.metric("Günstiges Szenario", f"{k6['p75']:.2f}",
                   f"{(k6['p75']-preis)/preis*100:+.1f}%")
        m3c.metric("Schlechtes Szenario", f"{k6['p25']:.2f}",
                   f"{(k6['p25']-preis)/preis*100:+.1f}%")
        st.metric("Dein Gesamtgewinn (Kurs + Dividende)", f"+{g6tot}%", delta_color=g6col)
        st.metric(
            label=f"In {gwkt6}% der Szenarien machst du Gewinn",
            value=f"Stop-Loss bei {sl:.2f} EUR  (-{rp}%)",
            delta=f"Maximal {rvmax:.0f} EUR Verlust", delta_color="inverse"
        )
        st.caption(f"Extremfall ohne Stop-Loss: unten {k6['p5']:.0f} "
                   f"({(k6['p5']-preis)/preis*100:+.0f}%) | "
                   f"oben {k6['p95']:.0f} ({(k6['p95']-preis)/preis*100:+.0f}%)")

    with pr2:
        st.markdown("#### In 1 Jahr — kaufst du heute")
        g1col = "normal" if g1tot >= 0 else "inverse"
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("Erwarteter Kurs", f"{k1['p50']:.2f}",
                   f"{k1['ret']:+.1f}% Kursgewinn")
        m2c.metric("Günstiges Szenario", f"{k1['p75']:.2f}",
                   f"{(k1['p75']-preis)/preis*100:+.1f}%")
        m3c.metric("Schlechtes Szenario", f"{k1['p25']:.2f}",
                   f"{(k1['p25']-preis)/preis*100:+.1f}%")
        st.metric("Dein Gesamtgewinn (Kurs + Dividende)", f"+{g1tot}%", delta_color=g1col)
        st.metric(
            label=f"In {gwkt1}% der Szenarien machst du Gewinn",
            value=f"Stop-Loss bei {sl:.2f} EUR  (-{rp}%)",
            delta=f"Maximal {rvmax:.0f} EUR Verlust", delta_color="inverse"
        )
        st.caption(f"Extremfall ohne Stop-Loss: unten {k1['p5']:.0f} "
                   f"({(k1['p5']-preis)/preis*100:+.0f}%) | "
                   f"oben {k1['p95']:.0f} ({(k1['p95']-preis)/preis*100:+.0f}%)")

    st.plotly_chart(chart_prognose(df, p6, p1, preis, ticker), use_container_width=True)

    # ── Simulationsvergleich ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Simulationsvergleich")
    st.caption("Alle drei Methoden auf einen Blick — stimmen sie überein, erhöht das die Verlässlichkeit.")

    sim_tab = st.tabs(["6 Monate", "1 Jahr"])
    for tab, horizon_label, kmc, kbs, kgc in [
        (sim_tab[0], "6 Monate", k6, kb6, kg6),
        (sim_tab[1], "1 Jahr",   k1, kb1, kg1),
    ]:
        with tab:
            sc1, sc2, sc3 = st.columns(3)
            for col, label, k, color in [
                (sc1, "📐 Monte Carlo", kmc, "normal"),
                (sc2, "🎲 Bootstrap",   kbs, "normal"),
                (sc3, "📈 GARCH(1,1)",  kgc, "normal"),
            ]:
                with col:
                    st.markdown(f"**{label}**")
                    gwkt_val = round(k["gwkt"]*100)
                    ret_val  = round(k["ret"] + dv["r6" if horizon_label=="6 Monate" else "r1"], 1)
                    col.metric("Gewinn-Wkt.", f"{gwkt_val} %")
                    col.metric("Erw. Ertrag", f"+{ret_val} %",
                               delta_color="normal" if ret_val >= 0 else "inverse")
                    col.metric("Median-Kurs", f"{k['p50']:.2f} EUR")
                    col.metric("Worst Case",  f"{k['p5']:.0f} EUR",
                               f"{(k['p5']-preis)/preis*100:+.0f}%",
                               delta_color="inverse")

            # Übereinstimmungs-Hinweis
            gwkts = [round(kmc["gwkt"]*100), round(kbs["gwkt"]*100), round(kgc["gwkt"]*100)]
            spanne = max(gwkts) - min(gwkts)
            if spanne <= 5:
                st.success(f"✅ Alle drei Methoden stimmen überein (Spanne: {spanne} %) — hohes Vertrauen in die Prognose.")
            elif spanne <= 15:
                st.warning(f"⚠️ Leichte Abweichungen zwischen den Methoden (Spanne: {spanne} %) — Prognose mit Vorsicht interpretieren.")
            else:
                st.error(f"❌ Große Abweichungen (Spanne: {spanne} %) — die Aktie verhält sich unregelmäßig, Prognosen unsicher.")

    # Vergleichs-Chart (1 Jahr)
    st.plotly_chart(sim_vergleich(p1, bs1, gc1, preis, preis), use_container_width=True)

    # ── Gesamtfazit ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Gesamtfazit")
    st.markdown(gesamtfazit(
        name=meta.get("name", ticker), ticker=ticker,
        preis=preis, inv=inv, nakt=nakt,
        gwkt6=gwkt6, gwkt1=gwkt1,
        g6tot=g6tot, g1tot=g1tot,
        sl=sl, rp=rp, rvmax=rvmax,
        k6=k6, k1=k1, sig=sig, ana=ana,
    ))

    st.markdown("---")

    # ── Technische Analyse ────────────────────────────────────────────────────
    st.subheader("📊 Technische Analyse")
    st.plotly_chart(chart_tech(df, ticker), use_container_width=True)

    with st.expander("KI-Details"):
        e1, e2 = st.columns([3, 2])
        with e1:
            sp = int((score + 0.55) / 1.10 * 100)
            sp = max(0, min(100, sp))
            st.markdown(f"**Score:** {score:+.2f}  |  {sig}")
            st.progress(sp / 100)
            st.caption(f"Bollinger: Oben {bb_u:.2f}  Unten {bb_l:.2f}")
        with e2:
            if imp:
                lbl = list(imp.keys())
                val = list(imp.values())
                idx = sorted(range(len(val)), key=lambda i: val[i])
                fig_i = go.Figure(go.Bar(
                    x=[val[i] for i in idx], y=[lbl[i] for i in idx],
                    orientation="h", marker_color="#4a9eff"))
                fig_i.update_layout(height=220, template="plotly_dark",
                                    paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                                    margin=dict(l=5,r=5,t=5,b=5))
                st.plotly_chart(fig_i, use_container_width=True)

    with st.expander("Rohdaten (letzte 30 Tage)"):
        cols = ["Open","High","Low","Close","Volume","SMA20","SMA50","RSI","MACD","ATR"]
        st.dataframe(df[cols].tail(30).style.format("{:.2f}"), use_container_width=True)
