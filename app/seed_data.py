import numpy as np, pandas as pd
rng = np.random.default_rng(42)

products = ["Product A","Product B","Product C"]
departments = ["Sales","Marketing","IT","HR","Finance","Operations","R&D","Admin"]
categories = {
    "Revenue": ["Product Revenue"],
    "COGS": ["Materials","Freight","Packaging"],
    "OPEX": ["Salaries","Digital Ads","Software","Travel","Rent","Utilities","Training"]
}
gl_map = {
    "Revenue": 4000,
    "COGS": 5000,
    "OPEX": 6000
}

periods = pd.period_range("2025-01", "2025-06", freq="M")
rows = []

for p in products:
    # revenue per product per month
    base_rev = rng.integers(250_000, 420_000)
    for per in periods:
        rev_budget = base_rev * (1 + rng.normal(0, 0.03))
        rev_actual = rev_budget * (1 + rng.normal(0, 0.04))
        rows.append({
            "Date": per.end_time.date().isoformat(),
            "Accounting_Period": per.strftime("%Y-%m"),
            "Accounting_Period_Label": per.strftime("%b %y").title(),  # e.g., Jan 25
            "Product": p,
            "Department": "Sales",
            "Category": "Product Revenue",
            "GL_Code": gl_map["Revenue"],
            "Account": "Product Revenue",
            "Type": "Revenue",
            "Cost_Class": "—",
            "Budget": round(rev_budget, 2),
            "Actual": round(rev_actual, 2),
            "Forecast": round(rev_budget * (1 + rng.normal(0, 0.02)), 2),
            "Notes": "Topline"
        })

    # costs (direct = COGS, indirect = OPEX)
    for kind, cats in categories.items():
        if kind == "Revenue":
            continue
        for dept in departments:
            for per in periods:
                for cat in cats:
                    base = {"COGS": rng.integers(20_000, 60_000),
                            "OPEX": rng.integers(5_000, 35_000)}[kind]
                    bud = base * (1 + 0.15 * (dept == "Marketing") + rng.normal(0,0.08))
                    act = bud * (1 + rng.normal(0,0.12))
                    rows.append({
                        "Date": per.end_time.date().isoformat(),
                        "Accounting_Period": per.strftime("%Y-%m"),
                        "Accounting_Period_Label": per.strftime("%b %y").title(),
                        "Product": p,
                        "Department": dept,
                        "Category": cat,
                        "GL_Code": gl_map[kind],
                        "Account": f"{cat}",
                        "Type": kind,  # Revenue / COGS / OPEX
                        "Cost_Class": "Direct" if kind=="COGS" else "Indirect",
                        "Budget": round(bud, 2),
                        "Actual": round(act, 2),
                        "Forecast": round(bud * (1 + rng.normal(0, 0.05)), 2),
                        "Notes": ""
                    })

df = pd.DataFrame(rows)
df.to_csv("data.csv", index=False)
print("Wrote data.csv with", len(df), "rows")
