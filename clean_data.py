"""
Clean the raw CSV files and build analysis-ready tables.

Input : data/raw/*.csv
Output: data/processed/customers.csv
        data/processed/orders.csv
        data/processed/order_lines.csv   (one row per product in an order)
        reports/tables/cleaning_log.csv  (what was fixed, and how many rows)

Usage:
    python src/clean_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"

REGION_MAP = {
    "tashkent city": "Tashkent City", "toshkent shahri": "Tashkent City",
    "samarkand": "Samarkand", "samarqand": "Samarkand",
    "fergana": "Fergana", "farg'ona": "Fergana",
    "andijan": "Andijan", "andijon": "Andijan",
    "namangan": "Namangan",
    "bukhara": "Bukhara", "buxoro": "Bukhara",
}

log = []


def note(step: str, rows: int):
    log.append({"step": step, "rows_affected": int(rows)})
    print(f"  {step:<55} {rows:>6,}")


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    key = df["region"].str.strip().str.lower()
    fixed = key.map(REGION_MAP)
    changed = (fixed.notna() & (fixed != df["region"])).sum()
    df["region"] = fixed.fillna(df["region"].str.strip())
    note("Standardised region spelling", changed)

    bad_age = df["age"].notna() & ~df["age"].between(14, 90)
    note("Impossible ages set to missing (<14 or >90)", bad_age.sum())
    df.loc[bad_age, "age"] = np.nan
    note("Missing ages left as NaN (not imputed)", df["age"].isna().sum())

    df["age_group"] = pd.cut(df["age"], bins=[17, 24, 34, 44, 54, 100],
                             labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    return df


def parse_dates(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dotted = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(dotted)


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    note("Duplicate order rows removed", before - len(df))

    dotted = df["order_date"].str.contains(r"\.", regex=True).sum()
    df["order_date"] = parse_dates(df["order_date"])
    note("Dates in dd.mm.yyyy converted to ISO", dotted)
    assert df["order_date"].notna().all(), "unparsed dates left"

    upper = (df["status"] != df["status"].str.title()).sum()
    df["status"] = df["status"].str.title()
    note("Status casing normalised", upper)

    note("Missing payment method -> 'Unknown'", df["payment_method"].isna().sum())
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["year"] = df["order_date"].dt.year
    return df


def clean_items(df: pd.DataFrame, valid_orders: pd.Series) -> pd.DataFrame:
    df = df.copy()
    bad_qty = df["quantity"] <= 0
    note("Rows with quantity <= 0 dropped", bad_qty.sum())
    df = df[~bad_qty]

    orphan = ~df["order_id"].isin(valid_orders)
    note("Items without a matching order dropped", orphan.sum())
    return df[~orphan]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    print("Cleaning raw data")
    customers = clean_customers(pd.read_csv(RAW / "customers.csv"))
    products = pd.read_csv(RAW / "products.csv")
    orders = clean_orders(pd.read_csv(RAW / "orders.csv"))
    items = clean_items(pd.read_csv(RAW / "order_items.csv"), orders["order_id"])

    # one row per product in an order, with money columns calculated
    lines = (items.merge(products, on="product_id", how="left")
                  .merge(orders[["order_id", "customer_id", "order_date",
                                 "year_month", "year", "status",
                                 "payment_method"]], on="order_id", how="left"))
    lines["gross_amount"] = lines["quantity"] * lines["unit_price"]
    lines["discount_amount"] = lines["gross_amount"] * lines["discount_pct"] / 100
    lines["net_revenue"] = lines["gross_amount"] - lines["discount_amount"]
    lines["cogs"] = lines["quantity"] * lines["unit_cost"]
    lines["gross_profit"] = lines["net_revenue"] - lines["cogs"]

    # an order whose only item had a bad quantity has nothing left to analyse
    empty = ~orders["order_id"].isin(lines["order_id"])
    note("Orders with no valid items dropped", empty.sum())
    orders = orders[~empty]

    customers.to_csv(OUT / "customers.csv", index=False)
    orders.to_csv(OUT / "orders.csv", index=False)
    lines.to_csv(OUT / "order_lines.csv", index=False)
    products.to_csv(OUT / "products.csv", index=False)
    pd.DataFrame(log).to_csv(TABLES / "cleaning_log.csv", index=False)

    print(f"\nclean customers : {len(customers):,}")
    print(f"clean orders    : {len(orders):,}")
    print(f"order lines     : {len(lines):,}")


if __name__ == "__main__":
    main()
