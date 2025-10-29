# utils/chatbot_engine.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.express as px

@dataclass
class QAResult:
    answer: str
    table: Optional[pd.DataFrame] = None
    figure: Optional[Any] = None  # plotly figure
    debug: Optional[str] = None

# --- Helpers ---
def _money(v: float) -> str:
    try:
        return "£{:,.0f}".format(float(v))
    except Exception:
        return "£0"

def _pct(v: float) -> str:
    try:
        return f"{float(v):.1%}"
    except Exception:
        return "0.0%"

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _topn(df: pd.DataFrame, col: str, n: int, asc: bool) -> pd.DataFrame:
    return df.sort_values(col, ascending=asc).head(n)

def schema_brief() -> str:
    return (
        "Columns: Date, Accounting_Period, Period_Label, Product, Department, "
        "Category, GL_Code, Account, Budget, Actual, Forecast, Type (Revenue/Cost), "
        "Direct_Indirect (Direct/Indirect), Base, VarianceGBP, VariancePct.\n"
        "Scenario: Variance is vs Base (Budget or Forecast)."
    )

# --- Core: offline intent router over the *already filtered* df (dff) ---
def ask(dff: pd.DataFrame, scenario: str, period_label: str, product_sel: str, question: str) -> QAResult:
    q = _norm(question)

    if dff is None or dff.empty:
        return QAResult(answer="I don't have any rows for the current filters.")

    # Aggregate fundamentals
    by_type = dff.groupby("Type")[["Base","Actual","VarianceGBP"]].sum()
    rev_base  = by_type.loc["Revenue","Base"]  if "Revenue" in by_type.index else 0.0
    rev_act   = by_type.loc["Revenue","Actual"] if "Revenue" in by_type.index else 0.0
    rev_var   = by_type.loc["Revenue","VarianceGBP"] if "Revenue" in by_type.index else 0.0

    di = dff[dff["Direct_Indirect"]=="Direct"].agg({"Base":"sum","Actual":"sum","VarianceGBP":"sum"}).fillna(0)
    ii = dff[dff["Direct_Indirect"]=="Indirect"].agg({"Base":"sum","Actual":"sum","VarianceGBP":"sum"}).fillna(0)
    dir_base, dir_act, dir_var = float(di["Base"]), float(di["Actual"]), float(di["VarianceGBP"])
    ind_base, ind_act, ind_var = float(ii["Base"]), float(ii["Actual"]), float(ii["VarianceGBP"])

    ebitda_base = rev_base - dir_base - ind_base
    ebitda_act  = rev_act  - dir_act  - ind_act
    ebitda_var  = ebitda_act - ebitda_base

    # Quick direct questions
    if "revenue" in q and ("what" in q or "how" in q or "this" in q):
        ans = f"Revenue in {period_label} is {_money(rev_act)} ({'+' if rev_var>=0 else ''}{_money(rev_var)} vs {scenario})."
        if product_sel != "All":
            ans += f" (Product: {product_sel})"
        return QAResult(answer=ans)

    if ("direct" in q and "cost" in q) or "cogs" in q:
        ans = f"Direct costs are {_money(dir_act)} ({'+' if dir_var>=0 else ''}{_money(dir_var)} vs {scenario})."
        return QAResult(answer=ans)

    if ("indirect" in q) or ("opex" in q) or ("overhead" in q):
        ans = f"Indirect costs are {_money(ind_act)} ({'+' if ind_var>=0 else ''}{_money(ind_var)} vs {scenario})."
        return QAResult(answer=ans)

    if "ebitda" in q:
        ans = f"EBITDA is {_money(ebitda_act)} ({'+' if ebitda_var>=0 else ''}{_money(ebitda_var)} vs {scenario})."
        return QAResult(answer=ans)

    # Top/bottom drivers
    m_top = re.search(r"(top|biggest)\s+(\d+)?\s*(drivers|variances|overspends|underspends)", q)
    if m_top:
        n = int(m_top.group(2) or 5)
        # overspends = positive variance on costs OR negative on revenue.
        # Here we just take absolute variance rank by Category x Department.
        grp = (dff.groupby(["Department","Category"])[["VarianceGBP","Base","Actual"]]
               .sum().reset_index())
        overspends = grp.sort_values("VarianceGBP", ascending=False).head(n)
        table = overspends.rename(columns={"VarianceGBP":"Variance (GBP)"})
        lines = [f"{r.Department} › {r.Category}: {_money(r['Variance (GBP)'])}" for _, r in table.iterrows()]
        ans = "Top drivers this period:\n- " + "\n- ".join(lines)
        return QAResult(answer=ans, table=table)

    m_btm = re.search(r"(bottom|smallest)\s+(\d+)?\s*(drivers|variances)", q)
    if m_btm:
        n = int(m_btm.group(2) or 5)
        grp = (dff.groupby(["Department","Category"])[["VarianceGBP","Base","Actual"]]
               .sum().reset_index())
        unders = grp.sort_values("VarianceGBP", ascending=True).head(n)
        table = unders.rename(columns={"VarianceGBP":"Variance (GBP)"})
        lines = [f"{r.Department} › {r.Category}: {_money(r['Variance (GBP)'])}" for _, r in table.iterrows()]
        ans = "Bottom drivers this period:\n- " + "\n- ".join(lines)
        return QAResult(answer=ans, table=table)

    # Department or Category specific
    m_dep = re.search(r"(?:dept|department)\s+([a-z0-9 &/_-]+)", q)
    if m_dep:
        dep = m_dep.group(1).strip().title()
        sub = dff[dff["Department"].str.lower()==dep.lower()]
        if sub.empty:
            return QAResult(answer=f"I can't find any rows for Department '{dep}' in {period_label}.")
        v = sub["VarianceGBP"].sum()
        ans = f"Department {dep}: variance {'+' if v>=0 else ''}{_money(v)} vs {scenario}."
        return QAResult(answer=ans)

    m_cat = re.search(r"(?:cat|category)\s+([a-z0-9 &/_-]+)", q)
    if m_cat:
        cat = m_cat.group(1).strip().title()
        sub = dff[dff["Category"].str.lower()==cat.lower()]
        if sub.empty:
            return QAResult(answer=f"I can't find any rows for Category '{cat}' in {period_label}.")
        v = sub["VarianceGBP"].sum()
        ans = f"Category {cat}: variance {'+' if v>=0 else ''}{_money(v)} vs {scenario}."
        return QAResult(answer=ans)

    # Trend ask (show last up-to-6 periods variance%)
    if "trend" in q or "last" in q or "over time" in q:
        hist = dff.copy()
        # We only have one period in dff; caller should pass full df for trends.
        # So we return a hint instead.
        return QAResult(
            answer="For trends, switch to the Visual Analytics trend chart (last 6 periods). "
                   "We can wire a trend-aware chat later against the full, unfiltered data."
        )

    # Fallback: small synopsis
    ans = (
        f"In {period_label}, Revenue {_money(rev_act)} ({'+' if rev_var>=0 else ''}{_money(rev_var)} vs {scenario}), "
        f"Direct {_money(dir_act)} ({'+' if dir_var>=0 else ''}{_money(dir_var)}), "
        f"Indirect {_money(ind_act)} ({'+' if ind_var>=0 else ''}{_money(ind_var)}), "
        f"EBITDA {_money(ebitda_act)} ({'+' if ebitda_var>=0 else ''}{_money(ebitda_var)})."
    )
    return QAResult(answer=ans, debug=schema_brief())
