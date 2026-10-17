"""
Exploratory analysis of the online-store data.

Input : data/processed/*.csv   (run clean_data.py first)
Output: reports/tables/*.csv   (every number used in the README)
        reports/figures/*.png

Revenue convention used everywhere:
    net revenue = quantity * unit_price * (1 - discount), counted only for
    orders with status "Delivered". Returned and cancelled orders are excluded.

Usage:
    python src/analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ACCENT, GREY, RED, DARK = "#1F6F8B", "#A9B3BD", "#C0392B", "#2B2B2B"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.edgecolor": "#888",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#E6E9ED", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def out(df, name, index=False):
    df.to_csv(TAB / name, index=index)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
customers = pd.read_csv(PROC / "customers.csv", parse_dates=["signup_date"])
orders = pd.read_csv(PROC / "orders.csv", parse_dates=["order_date"])
lines = pd.read_csv(PROC / "order_lines.csv", parse_dates=["order_date"])

order_value = lines.groupby("order_id")["net_revenue"].sum().rename("order_value")
orders = orders.merge(order_value, on="order_id", how="left")
cust_cols = ["customer_id", "region", "acquisition_channel", "age_group", "gender"]
orders = orders.merge(customers[cust_cols], on="customer_id", how="left")
lines = lines.merge(customers[["customer_id", "region"]], on="customer_id", how="left")

delivered_orders = orders[orders["status"] == "Delivered"]
delivered_lines = lines[lines["status"] == "Delivered"]


# --------------------------------------------------------------------------
# 1. Headline KPIs
# --------------------------------------------------------------------------
def kpi_block(o, l):
    d = o[o["status"] == "Delivered"]
    ld = l[l["status"] == "Delivered"]
    shipped = o["status"].isin(["Delivered", "Returned"]).sum()
    rev = ld["net_revenue"].sum()
    return {
        "net_revenue_uzs_m": rev / 1e6,
        "gross_profit_uzs_m": ld["gross_profit"].sum() / 1e6,
        "gross_margin_pct": ld["gross_profit"].sum() / rev * 100,
        "orders_placed": len(o),
        "orders_delivered": len(d),
        "avg_order_value_uzs": rev / len(d),
        "return_rate_pct": (o["status"] == "Returned").sum() / shipped * 100,
        "cancel_rate_pct": (o["status"] == "Cancelled").mean() * 100,
        "buying_customers": d["customer_id"].nunique(),
        "orders_per_customer": len(d) / d["customer_id"].nunique(),
    }


kpi_rows = {"All": kpi_block(orders, lines)}
for y in (2024, 2025):
    kpi_rows[str(y)] = kpi_block(orders[orders["year"] == y],
                                 lines[lines["year"] == y])
kpis = pd.DataFrame(kpi_rows).T
out(kpis.round(2), "kpis.csv", index=True)

per_cust = delivered_orders.groupby("customer_id").size()
repeat_rate = (per_cust >= 2).mean() * 100
print("=== KPIs ===")
print(kpis.round(1).T.to_string())
print(f"\nRepeat customer rate (>=2 delivered orders): {repeat_rate:.1f}%")


# --------------------------------------------------------------------------
# 2. Monthly trend and seasonality
# --------------------------------------------------------------------------
monthly = (delivered_lines.groupby("year_month")
           .agg(net_revenue=("net_revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
                orders=("order_id", "nunique"),
                customers=("customer_id", "nunique"))
           .reset_index())
monthly["year"] = monthly["year_month"].str[:4].astype(int)
monthly["month"] = monthly["year_month"].str[5:].astype(int)
monthly["mom_pct"] = monthly["net_revenue"].pct_change() * 100
prev = monthly.set_index(["year", "month"])["net_revenue"]
monthly["yoy_pct"] = [
    (r.net_revenue / prev.get((r.year - 1, r.month)) - 1) * 100
    if (r.year - 1, r.month) in prev.index else np.nan
    for r in monthly.itertuples()
]
out(monthly.round(2), "monthly_revenue.csv")

rev_by_year = monthly.groupby("year")["net_revenue"].sum()
yoy_total = (rev_by_year[2025] / rev_by_year[2024] - 1) * 100
piv = monthly.pivot(index="month", columns="year", values="net_revenue") / 1e6
peak_month = piv[2025].idxmax()

daily = delivered_lines.groupby("order_date")["net_revenue"].sum()
top_days = daily.sort_values(ascending=False).head(5)
print("\n=== Seasonality ===")
print(f"2025 vs 2024 revenue: {yoy_total:+.1f}%   peak 2025 month: {MONTHS[peak_month - 1]}")
print("Top 5 revenue days:")
print((top_days / 1e6).round(1).to_string())

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(piv.index, piv[2024], color=GREY, marker="o", lw=2, label="2024")
ax.plot(piv.index, piv[2025], color=ACCENT, marker="o", lw=2.6, label="2025")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTHS)
ax.set_ylabel("Net revenue, UZS million")
ax.set_title(f"Monthly net revenue: 2025 is {yoy_total:+.0f}% vs 2024, "
             f"peak in {MONTHS[peak_month - 1]}")
ax.annotate(f"{piv[2025].max():,.0f}", (peak_month, piv[2025].max()),
            textcoords="offset points", xytext=(0, 9), ha="center",
            color=ACCENT, fontweight="bold")
ax.legend(loc="upper left")
save(fig, "01_monthly_revenue.png")


# --------------------------------------------------------------------------
# 3. Category performance
# --------------------------------------------------------------------------
cat = (delivered_lines.groupby("category")
       .agg(net_revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            units=("quantity", "sum"))
       .assign(margin_pct=lambda d: d["gross_profit"] / d["net_revenue"] * 100,
               revenue_share_pct=lambda d: d["net_revenue"] / d["net_revenue"].sum() * 100))
shipped_lines = lines[lines["status"].isin(["Delivered", "Returned"])]
cat["return_rate_pct"] = (shipped_lines.assign(r=shipped_lines["status"] == "Returned")
                          .groupby("category")["r"].mean() * 100)
cat = cat.sort_values("net_revenue", ascending=False)
out((cat.assign(net_revenue=cat["net_revenue"] / 1e6,
                gross_profit=cat["gross_profit"] / 1e6)
        .rename(columns={"net_revenue": "net_revenue_uzs_m",
                         "gross_profit": "gross_profit_uzs_m"})
        .round(2)), "category_performance.csv", index=True)
print("\n=== Categories ===")
print(cat.assign(net_revenue=cat["net_revenue"] / 1e6).round(1)
      [["net_revenue", "revenue_share_pct", "margin_pct", "return_rate_pct"]].to_string())

c = cat.sort_values("net_revenue")
fig, ax = plt.subplots(figsize=(9, 4.6))
colors = [RED if m < 20 else ACCENT for m in c["margin_pct"]]
ax.barh(c.index, c["net_revenue"] / 1e6, color=colors)
for y, (rev, m, r) in enumerate(zip(c["net_revenue"] / 1e6, c["margin_pct"],
                                    c["return_rate_pct"])):
    ax.text(rev, y, f"  margin {m:.0f}%  |  returns {r:.1f}%", va="center",
            fontsize=9, color=DARK)
ax.set_xlim(0, (c["net_revenue"] / 1e6).max() * 1.45)
ax.set_xlabel("Net revenue, UZS million")
ax.grid(axis="y", visible=False)
ax.set_title("Revenue by category (red = gross margin below 20%)")
save(fig, "02_category_revenue_margin.png")


# --------------------------------------------------------------------------
# 4. Regions
# --------------------------------------------------------------------------
dv = delivered_orders
reg = (dv.groupby("region")
       .agg(net_revenue=("order_value", "sum"), orders=("order_id", "count"),
            buying_customers=("customer_id", "nunique"))
       .assign(revenue_per_customer=lambda d: d["net_revenue"] / d["buying_customers"],
               avg_order_value=lambda d: d["net_revenue"] / d["orders"]))
deliv_days = orders.groupby("region")["delivery_days"].mean()
cancel = orders.assign(c=orders["status"] == "Cancelled").groupby("region")["c"].mean() * 100
reg["avg_delivery_days"] = deliv_days
reg["cancel_rate_pct"] = cancel
reg["revenue_share_pct"] = reg["net_revenue"] / reg["net_revenue"].sum() * 100
reg = reg.sort_values("net_revenue", ascending=False)
out(reg.assign(net_revenue=reg["net_revenue"] / 1e6)
       .rename(columns={"net_revenue": "net_revenue_uzs_m"}).round(2),
    "region_performance.csv", index=True)
print("\n=== Regions ===")
print(reg.assign(net_revenue=reg["net_revenue"] / 1e6).round(1)
      [["net_revenue", "revenue_share_pct", "buying_customers",
        "revenue_per_customer", "avg_delivery_days", "cancel_rate_pct"]].to_string())

r = reg.sort_values("revenue_per_customer")
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(r.index, r["revenue_per_customer"] / 1e3,
        color=[ACCENT if v >= r["revenue_per_customer"].median() else GREY
               for v in r["revenue_per_customer"]])
for y, v in enumerate(r["revenue_per_customer"] / 1e3):
    ax.text(v, y, f"  {v:,.0f}", va="center", fontsize=9)
ax.set_xlim(0, (r["revenue_per_customer"] / 1e3).max() * 1.15)
ax.set_xlabel("Net revenue per buying customer, UZS thousand")
ax.grid(axis="y", visible=False)
ax.set_title("Customer value by region")
save(fig, "03_region_revenue_per_customer.png")


# --------------------------------------------------------------------------
# 5. Discounts vs margin
# --------------------------------------------------------------------------
disc = (delivered_lines.groupby("discount_pct")
        .agg(lines=("order_id", "count"), net_revenue=("net_revenue", "sum"),
             gross_profit=("gross_profit", "sum")))
disc["margin_pct"] = disc["gross_profit"] / disc["net_revenue"] * 100
disc["share_of_lines_pct"] = disc["lines"] / disc["lines"].sum() * 100
out(disc.assign(net_revenue=disc["net_revenue"] / 1e6,
                gross_profit=disc["gross_profit"] / 1e6)
        .rename(columns={"net_revenue": "net_revenue_uzs_m",
                         "gross_profit": "gross_profit_uzs_m"}).round(2),
    "discount_vs_margin.csv", index=True)
print("\n=== Discount bands ===")
print(disc.round(1)[["lines", "share_of_lines_pct", "margin_pct"]].to_string())

# which categories lose money at deep discounts?
deep = delivered_lines[delivered_lines["discount_pct"] >= 20]
deep_cat = (deep.groupby("category")[["net_revenue", "gross_profit"]].sum()
            .assign(margin_pct=lambda d: d["gross_profit"] / d["net_revenue"] * 100)
            .sort_values("margin_pct"))
out(deep_cat.assign(net_revenue=deep_cat["net_revenue"] / 1e6,
                    gross_profit=deep_cat["gross_profit"] / 1e6).round(2),
    "deep_discount_by_category.csv", index=True)
print("\nMargin at >=20% discount, by category:")
print(deep_cat["margin_pct"].round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.bar([f"{d}%" for d in disc.index], disc["margin_pct"],
       color=[RED if m < 0 else ACCENT for m in disc["margin_pct"]])
for x, m in enumerate(disc["margin_pct"]):
    ax.text(x, m + (0.8 if m >= 0 else -2.2), f"{m:.1f}%", ha="center", fontsize=9)
ax.axhline(0, color=DARK, lw=0.8)
ax.set_xlabel("Discount applied to the line")
ax.set_ylabel("Gross margin, %")
ax.grid(axis="x", visible=False)
ax.set_title("Deeper discounts eat the margin")
save(fig, "04_discount_vs_margin.png")


# --------------------------------------------------------------------------
# 6. Acquisition channels
# --------------------------------------------------------------------------
buyers = delivered_orders.groupby("customer_id").agg(
    orders=("order_id", "count"), revenue=("order_value", "sum")).reset_index()
buyers = buyers.merge(customers[["customer_id", "acquisition_channel"]], on="customer_id")
chan = (buyers.groupby("acquisition_channel")
        .agg(buying_customers=("customer_id", "count"),
             repeat_rate_pct=("orders", lambda s: (s >= 2).mean() * 100),
             revenue_per_customer=("revenue", "mean"))
        .join(customers.groupby("acquisition_channel").size().rename("signups")))
chan["conversion_pct"] = chan["buying_customers"] / chan["signups"] * 100
chan = chan.sort_values("repeat_rate_pct", ascending=False)
out(chan.round(2), "channel_performance.csv", index=True)
print("\n=== Channels ===")
print(chan.round(1).to_string())

ch = chan.sort_values("repeat_rate_pct")
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(ch.index, ch["repeat_rate_pct"],
        color=[ACCENT if v >= repeat_rate else GREY for v in ch["repeat_rate_pct"]])
for y, v in enumerate(ch["repeat_rate_pct"]):
    ax.text(v, y, f"  {v:.1f}%", va="center", fontsize=9)
ax.axvline(repeat_rate, color=DARK, ls="--", lw=1)
ax.text(repeat_rate, len(ch) - 0.45, f" overall {repeat_rate:.1f}%", fontsize=9, color=DARK)
ax.set_xlim(0, ch["repeat_rate_pct"].max() * 1.25)
ax.set_xlabel("Customers with 2+ delivered orders, %")
ax.grid(axis="y", visible=False)
ax.set_title("Repeat purchase rate by acquisition channel")
save(fig, "05_channel_repeat_rate.png")


# --------------------------------------------------------------------------
# 7. Payment methods
# --------------------------------------------------------------------------
pay = (orders.groupby("payment_method")
       .agg(orders=("order_id", "count"),
            cancel_rate_pct=("status", lambda s: (s == "Cancelled").mean() * 100),
            return_rate_pct=("status", lambda s: (s == "Returned").mean() * 100)))
pay["avg_order_value"] = dv.groupby("payment_method")["order_value"].mean()
pay["share_of_orders_pct"] = pay["orders"] / pay["orders"].sum() * 100
pay = pay.sort_values("orders", ascending=False)
out(pay.round(2), "payment_methods.csv", index=True)
print("\n=== Payment methods ===")
print(pay.round(1).to_string())


# --------------------------------------------------------------------------
# 8. RFM segmentation
# --------------------------------------------------------------------------
snapshot = delivered_orders["order_date"].max() + pd.Timedelta(days=1)
rfm = (delivered_orders.groupby("customer_id")
       .agg(last_order=("order_date", "max"), frequency=("order_id", "count"),
            monetary=("order_value", "sum")))
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm["R"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.cut(rfm["frequency"], [0, 1, 2, 3, 5, np.inf],
                  labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5,
                   labels=[1, 2, 3, 4, 5]).astype(int)


def segment(row):
    if row.R >= 4 and row.F >= 4:
        return "Champions"
    if row.R >= 3 and row.F >= 3:
        return "Loyal"
    if row.R <= 2 and row.F >= 3:
        return "At risk (valuable)"
    if row.R >= 4 and row.F == 1:
        return "New customers"
    if row.R <= 2:
        return "Lost / hibernating"
    return "Needs attention"


rfm["segment"] = rfm.apply(segment, axis=1)
seg = (rfm.groupby("segment")
       .agg(customers=("monetary", "size"), revenue=("monetary", "sum"),
            avg_orders=("frequency", "mean"), avg_recency_days=("recency", "mean"))
       .assign(customer_share_pct=lambda d: d["customers"] / d["customers"].sum() * 100,
               revenue_share_pct=lambda d: d["revenue"] / d["revenue"].sum() * 100)
       .sort_values("revenue_share_pct", ascending=False))
out(rfm.reset_index().drop(columns=["last_order"]), "rfm_customers.csv")
out(seg.assign(revenue=seg["revenue"] / 1e6)
       .rename(columns={"revenue": "revenue_uzs_m"}).round(2),
    "rfm_segments.csv", index=True)
print("\n=== RFM segments ===")
print(seg.round(1)[["customers", "customer_share_pct", "revenue_share_pct",
                    "avg_orders", "avg_recency_days"]].to_string())

s = seg.sort_values("revenue_share_pct")
yy = np.arange(len(s))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.barh(yy + 0.2, s["revenue_share_pct"], height=0.4, color=ACCENT, label="Share of revenue")
ax.barh(yy - 0.2, s["customer_share_pct"], height=0.4, color=GREY, label="Share of customers")
ax.set_yticks(yy)
ax.set_yticklabels(s.index)
for y, (a, b) in enumerate(zip(s["revenue_share_pct"], s["customer_share_pct"])):
    ax.text(a, y + 0.2, f" {a:.0f}%", va="center", fontsize=8.5)
    ax.text(b, y - 0.2, f" {b:.0f}%", va="center", fontsize=8.5)
ax.set_xlabel("%")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
ax.set_title("RFM segments: who brings the revenue")
save(fig, "06_rfm_segments.png")


# --------------------------------------------------------------------------
# 9. Cohort retention (customers by month of first delivered order)
# --------------------------------------------------------------------------
d = delivered_orders[["order_id", "customer_id", "order_date"]].copy()
d["order_m"] = d["order_date"].dt.to_period("M")
d["cohort"] = d.groupby("customer_id")["order_m"].transform("min")
d["period"] = ((d["order_m"].dt.year - d["cohort"].dt.year) * 12
               + (d["order_m"].dt.month - d["cohort"].dt.month))
counts = d.groupby(["cohort", "period"])["customer_id"].nunique().unstack()
retention = counts.div(counts[0], axis=0) * 100
cohort_size = counts[0]

ret24 = retention.loc[[p for p in retention.index if p.year == 2024], list(range(1, 13))]
out(ret24.round(1).rename(index=str), "cohort_retention_2024.csv", index=True)
avg_ret = ret24.mean()
print("\n=== Cohort retention (2024 cohorts, % of cohort active) ===")
print(avg_ret.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5.4))
im = ax.imshow(ret24.to_numpy(), cmap="Blues", aspect="auto",
               vmin=0, vmax=max(10, np.nanmax(ret24.to_numpy())))
ax.set_xticks(range(12))
ax.set_xticklabels(range(1, 13))
ax.set_yticks(range(len(ret24)))
ax.set_yticklabels([f"{p}  (n={int(cohort_size[p])})" for p in ret24.index])
for i in range(ret24.shape[0]):
    for j in range(ret24.shape[1]):
        v = ret24.iat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > im.norm.vmax * 0.6 else DARK)
ax.set_xlabel("Months since first purchase")
ax.grid(False)
ax.set_title("Cohort retention: % of each 2024 cohort that ordered in month N")
save(fig, "07_cohort_retention.png")

print("\nDone. Tables ->", TAB.relative_to(ROOT), "| Figures ->", FIG.relative_to(ROOT))
