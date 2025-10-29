# dashboard.py — FinSight AI Variance Demo (Tableau-style, with Waterfall)
import os, calendar
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Brand (fallback if utils/brand.py isn't present) ----------
try:
    from utils.brand import BRAND
except Exception:
    BRAND = {
        "name": "FinSight AI",
        "tagline": "AI-powered intelligence for modern finance",
        "colors": {
            "primary": "#8B9D83",   # Sage
            "dark":    "#2C3E2A",   # Forest
            "cream":   "#F5F1E8",   # Background
            "charcoal":"#3A3D3A"    # Text
        },
        "font": "Inter",
    }

PRIMARY   = BRAND["colors"]["primary"]
CHARCOAL  = BRAND["colors"]["charcoal"]
CREAM     = BRAND["colors"]["cream"]
FOREST    = BRAND["colors"]["dark"]

st.set_page_config(page_title="FinSight AI — Variance", layout="wide")

# ---------- Minimalist theme tweaks ----------
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: {BRAND['font']}, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        color: {CHARCOAL};
        background: {CREAM};
    }}
    .metric-label {{ color:{CHARCOAL}; opacity:.8; font-size:.9rem; }}
    .metric-value {{ font-weight:700; font-size:1.5rem; }}
    .metric-warn {{ color:#E67E22; }}
    .metric-bad  {{ color:#C0392B; }}
    .metric-good {{ color:#2E7D32; }}
</style>
""", unsafe_allow_html=True)

# ---------- Login gate ----------
def login_gate():
    pw_needed = "APP_PASSWORD" in st.secrets
    if not pw_needed:
        return True
    if "auth_ok" in st.session_state and st.session_state["auth_ok"]:
        return True
    st.title("🔒 FinSight AI — Private Demo")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == st.secrets["APP_PASSWORD"]:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

login_gate()

# ---------- Helpers ----------
def month_label(yyyymm: str) -> str:
    # "2025-01" -> "Jan 25"
    try:
        y, m = yyyymm.split("-")
        m = int(m)
        return f"{calendar.month_abbr[m]} {str(y)[-2:]}"
    except Exception:
        return yyyymm

def fmt_money(x):
    try:
        return "£{:,.0f}".format(x)
    except Exception:
        return "£0"

def traffic_light(val, threshold_good=0, invert=False):
    """
    For EBITDA & Revenue (higher is better) invert=False.
    For Costs (lower is better) invert=True (so positive variance is 'bad').
    Returns CSS class.
    """
    if invert:
        if val > 0:  # over budget cost
            return "metric-bad"
        elif val < 0:
            return "metric-good"
        else:
            return "metric-warn"
    else:
        if val > 0:
            return "metric-good"
        elif val < 0:
            return "metric-bad"
        else:
            return "metric-warn"

# ---------- Data load & harmonisation ----------
REQUIRED = ["Date","Product","Department","Category","GL_Code","Account","Budget","Actual"]
OPTIONAL = ["Forecast","Accounting_Period","Direct_Indirect","Type"]

data_path = os.path.join(os.getcwd(), "data.csv")
if not os.path.exists(data_path):
    st.error("`data.csv` not found in the app folder. Place your demo data next to dashboard.py and refresh.")
    st.stop()

try:
    df = pd.read_csv(data_path, dtype={"GL_Code":"Int64"})
except Exception as e:
    st.error(f"Couldn’t read data.csv: {e}")
    st.stop()

# Ensure required cols exist
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"data.csv is missing required columns: {missing}")
    st.stop()

# Parse dates & types
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for num in ["Budget","Actual","Forecast"]:
    if num in df.columns:
        df[num] = pd.to_numeric(df[num], errors="coerce").fillna(0.0)

# Accounting_Period (YYYY-MM) + pretty label
if "Accounting_Period" not in df.columns:
    df["Accounting_Period"] = df["Date"].dt.strftime("%Y-%m").fillna("Unknown")
df["Period_Label"] = df["Accounting_Period"].astype(str).map(month_label)

# Direct/Indirect/Revenue typing if missing
# Simple NetSuite-ish rule-of-thumb:
#   <5000  -> Revenue
#   5000-5999 -> Direct Cost (COGS)
#   >=6000 -> Indirect (OpEx)
if "Type" not in df.columns:
    df["Type"] = np.where(df["GL_Code"].fillna(0) < 5000, "Revenue", "Cost")
if "Direct_Indirect" not in df.columns:
    di = np.where(df["GL_Code"].fillna(0) < 5000, "N/A",
         np.where(df["GL_Code"].between(5000,5999, inclusive="both"), "Direct", "Indirect"))
    df["Direct_Indirect"] = di

# Variances (GBP & %). We compute vs a chosen base (Budget or Forecast) later too.
df["VarianceGBP_Budget"] = df["Actual"] - df["Budget"]
df["VariancePct_Budget"] = df["VarianceGBP_Budget"] / df["Budget"].replace(0, np.nan)

if "Forecast" not in df.columns:
    df["Forecast"] = 0.0
df["VarianceGBP_Forecast"] = df["Actual"] - df["Forecast"]
df["VariancePct_Forecast"] = df["VarianceGBP_Forecast"] / df["Forecast"].replace(0, np.nan)

# ---------- Sidebar: filters ----------
st.sidebar.header("Filters")

# Product selector (single)
products = ["All"] + sorted(df["Product"].dropna().unique().tolist())
product_sel = st.sidebar.selectbox("Product P&L", products, index=0)

# Department / Category (multiselects, dropdown style)
dept_opts = sorted(df["Department"].dropna().unique().tolist())
dept_sel = st.sidebar.multiselect("Department", dept_opts, default=dept_opts)

cat_opts = sorted(df["Category"].dropna().unique().tolist())
cat_sel = st.sidebar.multiselect("Category", cat_opts, default=cat_opts)

# Period (single)
period_opts = df[["Accounting_Period","Period_Label"]].drop_duplicates().sort_values("Accounting_Period")
period_label_map = dict(zip(period_opts["Period_Label"], period_opts["Accounting_Period"]))
label_opts = period_opts["Period_Label"].tolist()
period_label_sel = st.sidebar.selectbox("Accounting Period", label_opts, index=len(label_opts)-1)
period_sel = period_label_map[period_label_sel]

# Scenario base
scenario = st.sidebar.selectbox("Scenario (base for variance)", ["Budget","Forecast"], index=0)

# Materiality slider (by %)
materiality = st.sidebar.slider("Materiality threshold (absolute %)", 0.0, 0.50, 0.05, 0.01)

# ---------- Apply filters ----------
mask = (df["Accounting_Period"] == period_sel)
if product_sel != "All":
    mask &= (df["Product"] == product_sel)
mask &= df["Department"].isin(dept_sel)
mask &= df["Category"].isin(cat_sel)

dff = df.loc[mask].copy()

# Choose variance vs base
if scenario == "Budget":
    dff["Base"] = dff["Budget"]
    dff["VarianceGBP"] = dff["VarianceGBP_Budget"]
    dff["VariancePct"] = dff["VariancePct_Budget"]
else:
    dff["Base"] = dff["Forecast"]
    dff["VarianceGBP"] = dff["VarianceGBP_Forecast"]
    dff["VariancePct"] = dff["VariancePct_Forecast"]

# ---------- Executive summary (Revenue / Direct / Indirect / EBITDA) ----------
grp = dff.groupby("Type")[["Base","Actual"]].sum()
rev_base  = grp.loc["Revenue","Base"]  if "Revenue"  in grp.index else 0.0
rev_act   = grp.loc["Revenue","Actual"] if "Revenue" in grp.index else 0.0

direct = dff[dff["Direct_Indirect"]=="Direct"][["Base","Actual"]].sum()
indir  = dff[dff["Direct_Indirect"]=="Indirect"][["Base","Actual"]].sum()

direct_base, direct_act = float(direct["Base"]), float(direct["Actual"])
ind_base,    ind_act    = float(indir["Base"]),  float(indir["Actual"])

ebitda_base = rev_base - direct_base - ind_base
ebitda_act  = rev_act  - direct_act  - ind_act

rev_var = rev_act - rev_base
dir_var = direct_act - direct_base
ind_var = ind_act - ind_base
ebitda_var = ebitda_act - ebitda_base

st.title("🎯 FinSight AI — Variance Dashboard")
st.caption(f"{BRAND['tagline']} • Period: **{period_label_sel}** • Scenario: **{scenario}**")

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"<div class='metric-label'>Revenue (vs {scenario})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value {traffic_light(rev_var, invert=False)}'>{fmt_money(rev_act)} "
                f"<span style='opacity:.6'>({fmt_money(rev_var)})</span></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"<div class='metric-label'>Direct Costs (vs {scenario})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value {traffic_light(dir_var, invert=True)}'>{fmt_money(direct_act)} "
                f"<span style='opacity:.6'>({fmt_money(dir_var)})</span></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='metric-label'>Indirect Costs (vs {scenario})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value {traffic_light(ind_var, invert=True)}'>{fmt_money(ind_act)} "
                f"<span style='opacity:.6'>({fmt_money(ind_var)})</span></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div class='metric-label'>EBITDA (vs {scenario})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value {traffic_light(ebitda_var, invert=False)}'>{fmt_money(ebitda_act)} "
                f"<span style='opacity:.6'>({fmt_money(ebitda_var)})</span></div>", unsafe_allow_html=True)

st.divider()

# ---------- Visual Analytics ----------
st.subheader("📊 Visual Analytics")

# Variance by Department (bar)
dept_var = (dff.groupby("Department")[["VarianceGBP"]]
            .sum()
            .reset_index()
            .sort_values("VarianceGBP", ascending=True))
fig_dept = px.bar(
    dept_var, x="VarianceGBP", y="Department", orientation="h",
    color="VarianceGBP", color_continuous_scale=["#2E7D32", PRIMARY, "#C0392B"],
    title="Variance by Department (GBP)", labels={"VarianceGBP":"Variance (GBP)"}
)
fig_dept.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=50,b=10))
st.plotly_chart(fig_dept, use_container_width=True)

# Budget vs Actual by Category (grouped bar)
cat_ba = (dff.groupby("Category")[["Base","Actual"]]
          .sum().reset_index().sort_values("Actual", ascending=False).head(12))
fig_cat = go.Figure()
fig_cat.add_trace(go.Bar(name="Base",   x=cat_ba["Category"], y=cat_ba["Base"]))
fig_cat.add_trace(go.Bar(name="Actual", x=cat_ba["Category"], y=cat_ba["Actual"]))
fig_cat.update_layout(barmode="group", title=f"Top Categories — Actual vs {scenario}",
                      margin=dict(l=10,r=10,t=50,b=10), xaxis_title="", yaxis_title="GBP")
st.plotly_chart(fig_cat, use_container_width=True)

# 6-month Variance % trend (if data present)
hist = df.copy()
hist["Period_Label"] = hist["Accounting_Period"].astype(str).map(month_label)
if product_sel != "All":
    hist = hist[hist["Product"] == product_sel]
hist = hist[hist["Department"].isin(dept_sel) & hist["Category"].isin(cat_sel)]
if scenario == "Budget":
    hist["VariancePct"] = (hist["Actual"] - hist["Budget"]) / hist["Budget"].replace(0, np.nan)
else:
    hist["VariancePct"] = (hist["Actual"] - hist["Forecast"]) / hist["Forecast"].replace(0, np.nan)

trend = (hist.groupby(["Accounting_Period","Period_Label"])["VariancePct"]
         .mean().reset_index().sort_values("Accounting_Period").tail(6))
if not trend.empty:
    fig_trend = px.line(trend, x="Period_Label", y="VariancePct",
                        title="Variance % Trend (last 6 periods)",
                        markers=True)
    fig_trend.update_layout(yaxis_tickformat=".1%", margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ---------- Detailed table ----------
st.subheader("📋 Detailed Variance Table")
table_cols = ["Product","Department","Category","GL_Code","Account","Base","Actual","VarianceGBP","VariancePct"]
tbl = dff[table_cols].copy()
tbl["VariancePct"] = tbl["VariancePct"].fillna(0.0)
# Materiality filter
tbl = tbl[ tbl["VariancePct"].abs() >= materiality ]
st.dataframe(tbl.sort_values("VarianceGBP", ascending=False), use_container_width=True)

st.divider()

# ---------- Waterfall: EBITDA bridge ----------
st.subheader("🪄 EBITDA Variance Bridge")

# Bridge components: deltas between Actual and Base for Revenue, Direct, Indirect
wf_labels  = ["EBITDA (Base)","Revenue Δ","Direct Costs Δ","Indirect Costs Δ","EBITDA (Actual)"]
wf_values  = [ebitda_base, rev_var, -dir_var, -ind_var, ebitda_act]  # costs deltas invert in display
wf_measures= ["absolute","relative","relative","relative","absolute"]

fig_wf = go.Figure(go.Waterfall(
    name="EBITDA Bridge",
    orientation="h",
    measure=wf_measures,
    y=wf_labels,
    x=wf_values,
    connector={"line":{"color":CHARCOAL,"width":1}},
    decreasing={"marker":{"color":"#C0392B"}},
    increasing={"marker":{"color":"#2E7D32"}},
    totals={"marker":{"color":PRIMARY}}
))
fig_wf.update_layout(title="EBITDA Variance Waterfall", margin=dict(l=40,r=20,t=50,b=20))
st.plotly_chart(fig_wf, use_container_width=True)

# ---------- Tiny export (Excel) ----------
st.subheader("Export")
def to_excel_bytes(df_in: pd.DataFrame) -> bytes:
    import io
    import xlsxwriter
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        df_in.to_excel(xw, sheet_name="Detailed", index=False)
        summary = df_in.groupby("Accounting_Period")[["Base","Actual"]].sum()
        summary["VarianceGBP"] = summary["Actual"] - summary["Base"]
        summary["VariancePct"] = summary["VarianceGBP"] / summary["Base"].replace(0, np.nan)
        summary.to_excel(xw, sheet_name="Summary")
    out.seek(0)
    return out.read()

st.download_button(
    "📥 Download Excel (filtered)",
    data=to_excel_bytes(dff),
    file_name=f"FinSight-Variance-{period_sel.replace('-','')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ====== Chat with your data (offline, grounded) ======
st.divider()
st.subheader("💬 Ask your data")

try:
    from utils.chatbot_engine import ask as qa_ask, QAResult
except Exception as e:
    st.info("Chatbot engine not found (utils/chatbot_engine.py). Add it to enable CFO chat.")
else:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.container():
        # Show history
        for role, msg in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.markdown(msg)

        # Input
        prompt = st.chat_input(
            "Ask things like: 'What is EBITDA this period?', "
            "'Top 5 overspends', 'Department Marketing', 'Category Software'"
        )
        if prompt:
            # Echo user
            st.session_state["chat_history"].append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            # Answer from the filtered data (ground truth)
            result: QAResult = qa_ask(dff, scenario, period_label_sel, product_sel, prompt)

            # Render assistant
            with st.chat_message("assistant"):
                st.markdown(result.answer)
                if result.table is not None and not result.table.empty:
                    st.dataframe(result.table, use_container_width=True)
                if result.figure is not None:
                    st.plotly_chart(result.figure, use_container_width=True)
                if result.debug:
                    with st.expander("Debug (schema)"):
                        st.code(result.debug)

            # Save answer to history
            st.session_state["chat_history"].append(("assistant", result.answer))


# ---------- Footer ----------
st.caption(f"© {datetime.now().year} {BRAND['name']} • Minimalist, board-ready visuals • {BRAND['tagline']}")
