# utils/chatbot/engine.py
from __future__ import annotations
import re
from typing import List, Dict
import pandas as pd
import numpy as np

# -----------------------------
# Small helpers
# -----------------------------
def _money(x: float) -> str:
    try:
        return f"£{x:,.0f}"
    except Exception:
        return str(x)

def _ensure_variances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Actual" in df.columns and "Budget" in df.columns:
        df["Variance_GBP"] = df["Actual"] - df["Budget"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["Variance_Pct"] = np.where(
                df["Budget"] != 0, (df["Actual"] - df["Budget"]) / df["Budget"], 0.0
            )
    else:
        # If the source lacks Actual/Budget, keep safe columns
        if "Variance_GBP" not in df.columns:
            df["Variance_GBP"] = 0.0
        if "Variance_Pct" not in df.columns:
            df["Variance_Pct"] = 0.0
    return df

def _filter_text_list(values: List[str]) -> str:
    if not values:
        return "All"
    return ", ".join(values[:6]) + ("…" if len(values) > 6 else "")

# -----------------------------
# Public API
# -----------------------------
def append_history(history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
    """
    Maintain a simple chat history list of {role, content} dicts.
    """
    history = list(history) if history else []
    history.append({"role": role, "content": content})
    return history

def answer_query(df: pd.DataFrame, question: str, filters: Dict[str, List[str]]) -> str:
    """
    Offline-friendly Q&A over the filtered dataframe.
    Recognises a few common finance questions; otherwise returns a helpful default.
    """
    if df is None or len(df) == 0:
        return "I don’t see any rows after filtering. Try widening your selections."

    df = _ensure_variances(df)

    # Normalise text
    q = (question or "").strip().lower()

    # Context string for the reply header
    fctx = (
        f"Filters → Product: {_filter_text_list(filters.get('Product', []))} • "
        f"Department: {_filter_text_list(filters.get('Department', []))} • "
        f"Category: {_filter_text_list(filters.get('Category', []))} • "
        f"Periods: {_filter_text_list(filters.get('Accounting_Period', []))}"
    )

    # --- Patterns ---
    # 1) Top overspends / underspends
    if re.search(r"\b(top\s*\d+\s*overspends|top\s*overspends|top\s*5\s*overspends)\b", q):
        top_n = 5
        m = re.search(r"top\s*(\d+)", q)
        if m:
            try:
                top_n = max(1, min(20, int(m.group(1))))
            except Exception:
                pass

        by_cols = [c for c in ["Department", "Category", "Product"] if c in df.columns]
        if not by_cols:
            by_cols = ["Accounting_Period"]

        g = df.groupby(by_cols, as_index=False)["Variance_GBP"].sum()
        g = g.sort_values("Variance_GBP", ascending=False).head(top_n)

        lines = [f"**Top {len(g)} overspends (by {', '.join(by_cols)}):**"]
        for _, r in g.iterrows():
            key = " • ".join(str(r[c]) for c in by_cols)
            lines.append(f"- {key}: {_money(r['Variance_GBP'])} over budget")
        return f"{fctx}\n\n" + "\n".join(lines)

    # 2) Period with highest cost / revenue / variance
    if "period" in q and ("highest" in q or "max" in q):
        per_col = "Accounting_Period" if "Accounting_Period" in df.columns else None
        if per_col:
            if "revenue" in q and "Revenue" in df.columns:
                g = df.groupby(per_col, as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
                top = g.head(1)
                if not top.empty:
                    return f"{fctx}\n\nHighest revenue period: **{top.iloc[0][per_col]}** at **{_money(top.iloc[0]['Revenue'])}**."
            if ("cost" in q) and ({"Direct_Costs","Indirect_Costs"} <= set(df.columns)):
                df["Total_Costs"] = df["Direct_Costs"] + df["Indirect_Costs"]
                g = df.groupby(per_col, as_index=False)["Total_Costs"].sum().sort_values("Total_Costs", ascending=False)
                top = g.head(1)
                if not top.empty:
                    return f"{fctx}\n\nHighest total cost period: **{top.iloc[0][per_col]}** at **{_money(top.iloc[0]['Total_Costs'])}**."
            if "variance" in q and "Variance_GBP" in df.columns:
                g = df.groupby(per_col, as_index=False)["Variance_GBP"].sum().sort_values("Variance_GBP", ascending=False)
                top = g.head(1)
                if not top.empty:
                    return f"{fctx}\n\nLargest positive variance period: **{top.iloc[0][per_col]}** at **{_money(top.iloc[0]['Variance_GBP'])}**."

    # 3) EBITDA for a period or overall
    if "ebitda" in q:
        revenue = float(df["Revenue"].sum()) if "Revenue" in df.columns else 0.0
        direct  = float(df["Direct_Costs"].sum()) if "Direct_Costs" in df.columns else 0.0
        indirect= float(df["Indirect_Costs"].sum()) if "Indirect_Costs" in df.columns else 0.0
        ebitda  = revenue - direct - indirect
        return f"{fctx}\n\nEBITDA in the current selection is **{_money(ebitda)}**."

    # 4) Generic summary
    total_budget = float(df["Budget"].sum()) if "Budget" in df.columns else 0.0
    total_actual = float(df["Actual"].sum()) if "Actual" in df.columns else 0.0
    if total_budget or total_actual:
        var = total_actual - total_budget
        pct = (var / total_budget) if total_budget else 0.0
        direction = "over" if var > 0 else "under"
        note = (
            f"Actual is **{direction}** budget by **{_money(abs(var))} ({pct:.1%})**."
            if (total_budget or total_actual) else
            "No Actual/Budget columns were detected."
        )
    else:
        note = "No Actual/Budget columns were detected."

    parts = [fctx, "", note]

    # top driver by Category
    if "Revenue" in df.columns and "Category" in df.columns:
        by_cat = df.groupby("Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
        if not by_cat.empty:
            parts.append(f"Top revenue category: **{by_cat.iloc[0]['Category']}** at **{_money(by_cat.iloc[0]['Revenue'])}**.")

    return "\n".join(parts)
