import pandas as pd
import numpy as np

def material_flag(row, pct=0.05, abs_amt=10000):
    v_abs = abs(row.get("Variance_£", 0))
    v_pct = abs(row.get("Variance_%", 0))
    return (v_pct >= pct) or (v_abs >= abs_amt)

def generate_insights(df_period: pd.DataFrame):
    """
    Returns up to 4 short, CFO-friendly bullets about the current filtered slice.
    Computes Variance_£ and Variance_% if missing.
    """
    if df_period is None or df_period.empty:
        return ["No data for the current selection."]

    dfp = df_period.copy()

    if "Variance_£" not in dfp.columns:
        dfp["Variance_£"] = dfp["Actual"] - dfp["Budget"]

    if "Variance_%" not in dfp.columns:
        denom = dfp["Budget"].replace(0, np.nan)
        dfp["Variance_%"] = dfp["Variance_£"] / denom

    bullets = []

    # Biggest overspend
    over = dfp.sort_values("Variance_£", ascending=False)
    for _, r in over.head(8).iterrows():
        if r["Variance_£"] > 0 and material_flag(r):
            bullets.append(
                f"{r['Department']} › {r['Category']} exceeded budget by "
                f"£{r['Variance_£']:,.0f} ({r['Variance_%']:.1%})."
            )
            break

    # Biggest underspend
    under = dfp.sort_values("Variance_£", ascending=True)
    for _, r in under.head(8).iterrows():
        if r["Variance_£"] < 0 and material_flag(r):
            bullets.append(
                f"{r['Department']} › {r['Category']} came in under budget by "
                f"£{-r['Variance_£']:,.0f} ({-r['Variance_%']:.1%})."
            )
            break

    # Topline proxy (if any revenue-like category present)
    if "Category" in dfp.columns and "Accounting_Period" in dfp.columns:
        rev_like = dfp[dfp["Category"].str.contains("Revenue|Sales", case=False, na=False)]
        if not rev_like.empty:
            grp = rev_like.groupby("Accounting_Period")[["Budget", "Actual"]].sum().tail(1)
            if not grp.empty:
                latest = grp.iloc[0]
                delta = latest["Actual"] - latest["Budget"]
                pct = (delta / latest["Budget"]) if latest["Budget"] else 0
                bullets.append(
                    f"Topline in latest period is "
                    f"{'above' if delta>0 else 'below'} budget by £{abs(delta):,.0f} ({abs(pct):.1%})."
                )

    # Ops/IT pressure note
    needed = {"Department", "Category", "Variance_£"}
    if needed.issubset(dfp.columns):
        it_ops = dfp[dfp["Department"].isin(["IT", "Operations"])]
        if not it_ops.empty:
            top = it_ops.sort_values("Variance_£", ascending=False).head(1).iloc[0]
            bullets.append(
                f"Cost pressure in {top['Department']} ({top['Category']}), "
                f"variance £{top['Variance_£']:,.0f} vs budget."
            )

    return bullets[:4]
