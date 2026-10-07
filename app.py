"""
STOCKROOM — multi-agent inventory decisions (Streamlit + Ollama)

Run:
    python -m pip install streamlit pandas ollama
    ollama pull llama3.2
    python -m streamlit run app.py
"""
import io
import json
import math
from html import escape

import ollama
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stockroom", page_icon="▣", layout="wide")

# --------------------------------------------------------------- Styling ----
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');

:root{
  --ink:#12130f; --panel:#1b1d17; --line:#34372c;
  --paper:#efe9da; --muted:#9a9783; --signal:#ff5a1f;
  --restock:#7bd88f; --reduce:#f2c14e; --disc:#ef6461; --keep:#8f9bb3;
}
html, body, [class*="css"], .stApp { font-family:'DM Sans',sans-serif; color:var(--paper); }
.stApp { background:var(--ink);
  background-image: radial-gradient(var(--line) 1px, transparent 1px);
  background-size: 26px 26px; }
header[data-testid="stHeader"], footer, #MainMenu { visibility:hidden; height:0; }
.block-container { padding-top:2.2rem; max-width:1200px; }
section[data-testid="stSidebar"] { background:#0c0d0a; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:var(--paper); }

.brand { font-family:'Fraunces',serif; font-weight:800; font-size:4.2rem; line-height:.95;
  letter-spacing:-2px; margin:0; }
.brand span { color:var(--signal); }
.tag { font-family:'IBM Plex Mono',monospace; color:var(--muted); font-size:.85rem;
  letter-spacing:2px; text-transform:uppercase; margin:.6rem 0 1.6rem; }

.label { font-family:'IBM Plex Mono',monospace; font-size:.75rem; letter-spacing:3px;
  text-transform:uppercase; color:var(--signal); margin:2rem 0 .7rem;
  border-bottom:1px solid var(--line); padding-bottom:.4rem; }

.stButton > button { background:var(--signal); color:#12130f; border:none; border-radius:2px;
  font-family:'IBM Plex Mono',monospace; font-weight:500; letter-spacing:2px;
  text-transform:uppercase; padding:.7rem 1.6rem; }
.stButton > button:hover { background:var(--paper); color:#12130f; }
.stDownloadButton > button { background:transparent; color:var(--paper); border:1px solid var(--line);
  border-radius:2px; font-family:'IBM Plex Mono',monospace; letter-spacing:1px; }
.stDownloadButton > button:hover { border-color:var(--signal); color:var(--signal); }

.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--c);
  padding:14px 16px; }
.kpi b { font-family:'Fraunces',serif; font-size:2.6rem; display:block; line-height:1; }
.kpi small { font-family:'IBM Plex Mono',monospace; color:var(--muted); letter-spacing:2px;
  text-transform:uppercase; font-size:.7rem; }

.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--c);
  padding:16px 18px; }
.card .top { display:flex; justify-content:space-between; align-items:center; }
.card .sku { font-family:'IBM Plex Mono',monospace; color:var(--muted); font-size:.75rem; }
.card .pill { font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:2px;
  background:var(--c); color:#12130f; padding:2px 8px; font-weight:500; }
.card h4 { font-family:'Fraunces',serif; font-size:1.3rem; margin:.5rem 0 .2rem; }
.card .qty { font-family:'IBM Plex Mono',monospace; color:var(--c); font-size:.85rem; }
.card p { color:#cfc9b6; font-size:.92rem; margin:.5rem 0 0; line-height:1.45; }

.agent { background:var(--panel); border:1px solid var(--line); padding:10px 14px; margin-bottom:8px; }
.agent b { font-family:'IBM Plex Mono',monospace; color:var(--signal); font-size:.8rem;
  letter-spacing:1px; text-transform:uppercase; }
div[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:2px; }
div[data-testid="stExpander"] summary { font-family:'IBM Plex Mono',monospace; }
.stProgress > div > div > div > div { background:var(--signal); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SAMPLE_CSV = """sku,product,stock,unit_cost,price,lead_time_days,weeks_1_ago,weeks_2_ago,weeks_3_ago,weeks_4_ago,days_since_last_sale
A101,Wireless Mouse,12,8,20,10,30,28,25,22,1
A102,USB-C Cable,400,1,6,7,5,6,4,5,3
A103,Laptop Stand,35,12,35,14,10,12,11,13,2
A104,Old VGA Adapter,150,2,7,21,0,0,1,0,64
A105,Mechanical Keyboard,8,30,80,14,9,8,7,6,1
A106,Phone Case X5,220,2,10,10,2,3,5,8,9
"""

# ---------------------------------------------------------------- Agents ----
AGENTS = {
    "Demand Agent": (
        "You are a demand analyst for a retail store. Look at the sales trend "
        "and weeks of stock cover. Say whether demand is rising, stable, or "
        "falling and whether stock will run out before the supplier lead time."
    ),
    "Finance Agent": (
        "You are a finance analyst. Look at margin, and the cash tied up in "
        "stock (stock * unit_cost). Say whether holding or buying more stock "
        "is financially wise."
    ),
    "Risk Agent": (
        "You are a risk analyst. Look at stockout risk (selling fast, low "
        "cover vs lead time) and dead-stock risk (no recent sales, huge "
        "cover). Flag the biggest risk."
    ),
}

DECISION_AGENT = (
    "You are the inventory manager. You receive the product data (including a "
    "stock_status field computed by code) and reports from the Demand, Finance "
    "and Risk agents. Decide ONE action: RESTOCK, REDUCE, DISCONTINUE or KEEP.\n"
    "RULES:\n"
    "- RESTOCK only if weeks_of_cover is LESS than lead_time_weeks (stock runs out "
    "before a new order arrives). Quantity must be greater than 0.\n"
    "- REDUCE if weeks_of_cover is very high (over 12 weeks) but the product still sells.\n"
    "- DISCONTINUE if avg_weekly_sales is 1 or less and days_since_last_sale is 45+.\n"
    "- KEEP if stock is balanced.\n"
    "- High weeks_of_cover or falling demand NEVER means RESTOCK.\n"
    "Your reason must agree with your action. Reply ONLY with JSON: "
    '{"action": "...", "quantity": <int, units to order or 0>, "reason": "<one sentence>"}'
)

COLORS = {"RESTOCK": "#7bd88f", "REDUCE": "#f2c14e", "DISCONTINUE": "#ef6461", "KEEP": "#8f9bb3"}


def ask(model, system, user, as_json=False):
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format="json" if as_json else "",
        options={"temperature": 0.2},
    )
    return resp["message"]["content"].strip()


def compute_metrics(row):
    sales = [row.weeks_1_ago, row.weeks_2_ago, row.weeks_3_ago, row.weeks_4_ago]
    avg = sum(sales) / 4
    recent, older = (sales[0] + sales[1]) / 2, (sales[2] + sales[3]) / 2
    trend = "rising" if recent > older * 1.15 else "falling" if recent < older * 0.85 else "stable"
    return {
        "sku": row.sku,
        "product": row["product"],
        "stock": int(row.stock),
        "avg_weekly_sales": round(avg, 1),
        "sales_last_4_weeks_newest_first": [int(s) for s in sales],
        "trend": trend,
        "weeks_of_cover": round(row.stock / avg, 1) if avg > 0 else 999,
        "lead_time_weeks": round(row.lead_time_days / 7, 1),
        "margin_pct": round((row.price - row.unit_cost) / row.price * 100),
        "cash_tied_up": round(row.stock * row.unit_cost, 2),
        "days_since_last_sale": int(row.days_since_last_sale),
    }


def stock_status(m):
    """Plain-language flag computed in code so no agent has to guess."""
    cover, lead = m["weeks_of_cover"], m["lead_time_weeks"]
    if m["avg_weekly_sales"] <= 1 and m["days_since_last_sale"] >= 45:
        return "DEAD STOCK: almost no sales for a long time"
    if cover < lead:
        return f"LOW: stock lasts {cover} weeks but resupply takes {lead} weeks"
    if cover > 12:
        return f"OVERSTOCKED: {cover} weeks of cover vs {lead} week lead time"
    return "BALANCED"


def rule_based(m):
    """Deterministic fallback decision, used only if the agent keeps contradicting itself."""
    if m["avg_weekly_sales"] <= 1 and m["days_since_last_sale"] >= 45:
        return {"action": "DISCONTINUE", "quantity": 0}
    if m["weeks_of_cover"] < m["lead_time_weeks"]:
        qty = math.ceil(m["avg_weekly_sales"] * (m["lead_time_weeks"] + 2)) - m["stock"]
        return {"action": "RESTOCK", "quantity": max(qty, 1)}
    if m["weeks_of_cover"] > 12:
        return {"action": "REDUCE", "quantity": 0}
    return {"action": "KEEP", "quantity": 0}


def validate(d, m):
    """Reviewer: returns a problem description if the decision contradicts the data."""
    a = d["action"]
    if a == "RESTOCK" and m["weeks_of_cover"] >= m["lead_time_weeks"]:
        return f"RESTOCK is wrong: stock lasts {m['weeks_of_cover']} weeks, longer than the {m['lead_time_weeks']} week lead time."
    if a == "RESTOCK" and int(d.get("quantity") or 0) <= 0:
        return "RESTOCK needs a quantity above 0."
    if a == "DISCONTINUE" and m["avg_weekly_sales"] > 1:
        return "DISCONTINUE is wrong: the product still sells."
    if a == "REDUCE" and m["weeks_of_cover"] < m["lead_time_weeks"]:
        return "REDUCE is wrong: stock is running low."
    return None


def parse_decision(raw):
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = {}
    d["action"] = str(d.get("action", "KEEP")).upper()
    if d["action"] not in COLORS:
        d["action"] = "KEEP"
    return d


def run_pipeline(model, row):
    m = compute_metrics(row)
    m["stock_status"] = stock_status(m)
    data = json.dumps(m)
    reports = {n: ask(model, p + " Reply in max 2 sentences.", data) for n, p in AGENTS.items()}
    briefing = f"Product data: {data}\n\n" + "\n".join(f"{k}: {v}" for k, v in reports.items())

    decision = parse_decision(ask(model, DECISION_AGENT, briefing, as_json=True))
    problem = validate(decision, m)
    if problem:  # reviewer sends it back once
        retry = briefing + f"\n\nREVIEWER REJECTED YOUR DECISION {json.dumps(decision)}. Problem: {problem} Decide again."
        decision = parse_decision(ask(model, DECISION_AGENT, retry, as_json=True))
        if validate(decision, m):  # still wrong -> rules take over
            fixed = rule_based(m)
            decision = {**fixed, "reason": f"Auto-corrected by rules: {m['stock_status']}."}
    return {"metrics": m, "reports": reports, "decision": decision}


# ------------------------------------------------------------------- UI -----
with st.sidebar:
    st.markdown("<div class='label' style='margin-top:0'>Settings</div>", unsafe_allow_html=True)
    model = st.text_input("Ollama model", "llama3.2")
    uploaded = st.file_uploader("Inventory CSV", type="csv")
    st.caption("No file? A small sample set is used.")

st.markdown("<h1 class='brand'>Stock<span>room</span></h1>", unsafe_allow_html=True)
st.markdown("<div class='tag'>Four local AI agents · one call per SKU · restock / reduce / discontinue</div>",
            unsafe_allow_html=True)

df = pd.read_csv(uploaded) if uploaded else pd.read_csv(io.StringIO(SAMPLE_CSV))

st.markdown("<div class='label'>01 / Inventory</div>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True, hide_index=True, height=240)

st.markdown("<div class='label'>02 / Run agents</div>", unsafe_allow_html=True)
if st.button("Run analysis"):
    results = []
    bar = st.progress(0.0, text="Starting agents...")
    try:
        for i, (_, row) in enumerate(df.iterrows()):
            bar.progress(i / len(df), text=f"Agents reviewing {row['product']} ({i + 1}/{len(df)})")
            results.append(run_pipeline(model, row))
    except Exception as e:
        st.error(f"Could not reach Ollama: {e}")
        st.stop()
    bar.empty()
    st.session_state["results"] = results

if "results" in st.session_state:
    results = st.session_state["results"]
    counts = {k: sum(r["decision"]["action"] == k for r in results) for k in COLORS}

    st.markdown("<div class='label'>03 / Decisions</div>", unsafe_allow_html=True)
    kpis = "".join(
        f"<div class='kpi' style='--c:{COLORS[k]}'><b>{counts[k]}</b><small>{k}</small></div>"
        for k in ["RESTOCK", "REDUCE", "DISCONTINUE", "KEEP"]
    )
    st.markdown(f"<div class='kpis'>{kpis}</div><br>", unsafe_allow_html=True)

    cards = ""
    for r in results:
        d, m = r["decision"], r["metrics"]
        qty = f"order {d.get('quantity', 0)} units" if d["action"] == "RESTOCK" else f"{m['stock']} in stock"
        cards += (
            f"<div class='card' style='--c:{COLORS[d['action']]}'>"
            f"<div class='top'><span class='sku'>{escape(str(m['sku']))}</span>"
            f"<span class='pill'>{d['action']}</span></div>"
            f"<h4>{escape(str(m['product']))}</h4><div class='qty'>{qty}</div>"
            f"<p>{escape(str(d.get('reason', '')))}</p></div>"
        )
    st.markdown(f"<div class='grid'>{cards}</div>", unsafe_allow_html=True)

    out = pd.DataFrame([
        {"sku": r["metrics"]["sku"], "product": r["metrics"]["product"],
         "action": r["decision"]["action"], "quantity": r["decision"].get("quantity", 0),
         "reason": r["decision"].get("reason", "")} for r in results
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("Download decisions CSV", out.to_csv(index=False), "decisions.csv")

    st.markdown("<div class='label'>04 / How each agent reasoned</div>", unsafe_allow_html=True)
    for r in results:
        with st.expander(f"{r['metrics']['sku']} — {r['metrics']['product']}"):
            for name, text in r["reports"].items():
                st.markdown(f"<div class='agent'><b>{name}</b><br>{escape(text)}</div>",
                            unsafe_allow_html=True)
