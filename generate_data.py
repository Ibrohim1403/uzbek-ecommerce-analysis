"""
Synthetic e-commerce data for an Uzbekistan online store (2024-2025).

Real transaction data is private, so this script simulates a realistic shop:
seasonality (Navruz, 11.11 sale, New Year), regional differences, repeat
buyers, discounts, returns and cancellations. The raw files are intentionally
messy (mixed date formats, spelling variants, duplicates, missing values)
so that the cleaning step has real work to do.

Usage:
    python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2024
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
N_CUSTOMERS = 6000
LIFETIME_DAYS = 400          # mean active life of a repeat customer
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
REGIONS = {
    "Tashkent City": 0.29, "Tashkent Region": 0.07, "Samarkand": 0.11,
    "Fergana": 0.11, "Andijan": 0.10, "Namangan": 0.09, "Bukhara": 0.07,
    "Kashkadarya": 0.06, "Khorezm": 0.05, "Surkhandarya": 0.05,
}
DELIVERY_MEAN_DAYS = {
    "Tashkent City": 1, "Tashkent Region": 2, "Samarkand": 3, "Fergana": 3,
    "Andijan": 3, "Namangan": 3, "Bukhara": 3, "Kashkadarya": 4,
    "Khorezm": 4, "Surkhandarya": 4,
}
CHANNELS = {  # share of new customers, repeat-purchase multiplier
    "Instagram": (0.32, 0.9), "Telegram": (0.22, 1.0), "Organic": (0.20, 1.1),
    "Google Ads": (0.14, 0.8), "Referral": (0.12, 1.3),
}

# category -> (popularity, cost as share of list price, [(product, price UZS)])
CATALOG = {
    "Electronics": (0.20, 0.82, [
        ("Wireless Earbuds", 380_000), ("Smartphone 128GB", 3_200_000),
        ("Smart Watch", 850_000), ("Power Bank 20000mAh", 260_000),
        ("Bluetooth Speaker", 420_000), ("Laptop 15.6in", 6_000_000),
        ("Tablet 10in", 2_300_000), ("Phone Case", 45_000),
        ("USB-C Charger", 150_000), ("Webcam HD", 330_000),
        ("Gaming Mouse", 280_000), ("LED Monitor 24in", 1_900_000)]),
    "Home & Kitchen": (0.15, 0.65, [
        ("Electric Kettle", 180_000), ("Blender", 320_000),
        ("Air Fryer", 950_000), ("Non-stick Pan Set", 420_000),
        ("Bed Sheet Set", 260_000), ("Vacuum Cleaner", 1_400_000),
        ("Dinner Set 24pcs", 380_000), ("Thermos 1L", 120_000),
        ("Table Lamp", 90_000), ("Storage Boxes", 60_000)]),
    "Fashion": (0.22, 0.50, [
        ("Men's Jacket", 520_000), ("Women's Dress", 380_000),
        ("Sneakers", 450_000), ("Jeans", 280_000), ("Hoodie", 220_000),
        ("Winter Coat", 890_000), ("T-shirt Pack", 120_000),
        ("Backpack", 190_000), ("Scarf", 70_000), ("Leather Belt", 110_000),
        ("Sports Socks Pack", 45_000), ("Handbag", 340_000)]),
    "Beauty": (0.15, 0.45, [
        ("Face Cream", 140_000), ("Perfume 50ml", 480_000),
        ("Shampoo", 65_000), ("Lipstick", 85_000), ("Hair Dryer", 260_000),
        ("Sunscreen SPF50", 110_000), ("Makeup Set", 300_000),
        ("Beard Trimmer", 190_000)]),
    "Sports": (0.08, 0.60, [
        ("Yoga Mat", 95_000), ("Dumbbells 10kg", 240_000),
        ("Football", 130_000), ("Running Shoes", 520_000),
        ("Fitness Tracker", 280_000), ("Resistance Bands", 60_000)]),
    "Books & Stationery": (0.08, 0.60, [
        ("Notebook A5", 18_000), ("Pen Set", 25_000),
        ("Business Book", 75_000), ("English Textbook", 90_000),
        ("Planner", 55_000), ("Art Supplies", 120_000)]),
    "Groceries": (0.12, 0.80, [
        ("Green Tea 200g", 35_000), ("Dried Fruit Mix", 80_000),
        ("Honey 1kg", 120_000), ("Walnuts 1kg", 110_000),
        ("Olive Oil 1L", 95_000), ("Coffee Beans 500g", 140_000)]),
}

MONTH_FACTOR = {1: 0.90, 2: 0.85, 3: 1.15, 4: 0.95, 5: 0.95, 6: 1.00,
                7: 0.90, 8: 1.05, 9: 1.00, 10: 1.00, 11: 1.35, 12: 1.30}


def seasonal_factor(d: pd.Timestamp) -> float:
    f = MONTH_FACTOR[d.month]
    if d.month == 11 and 10 <= d.day <= 12:      # 11.11 sale
        f *= 1.8
    if d.month == 3 and 18 <= d.day <= 21:       # Navruz
        f *= 1.25
    if d.dayofweek >= 5:                         # weekends
        f *= 1.08
    return f


# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------
def build_products() -> pd.DataFrame:
    rows = []
    for category, (_, cost_ratio, items) in CATALOG.items():
        for name, price in items:
            rows.append({
                "product_name": name,
                "category": category,
                "list_price": price,
                "unit_cost": int(round(price * cost_ratio, -3)),
            })
    df = pd.DataFrame(rows)
    df.insert(0, "product_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
    return df


# --------------------------------------------------------------------------
# Customers
# --------------------------------------------------------------------------
def build_customers(dates, fac):
    n_days = len(dates)
    growth = np.linspace(0.7, 1.3, n_days - 3)
    w = growth * fac[: n_days - 3]
    signup_idx = rng.choice(n_days - 3, size=N_CUSTOMERS, p=w / w.sum())

    channel_names = list(CHANNELS)
    channel_p = [CHANNELS[c][0] for c in channel_names]
    region_names = list(REGIONS)

    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        "signup_date": dates[signup_idx],
        "region": rng.choice(region_names, N_CUSTOMERS, p=list(REGIONS.values())),
        "gender": rng.choice(["F", "M"], N_CUSTOMERS, p=[0.54, 0.46]),
        "age": np.clip(rng.normal(31, 9, N_CUSTOMERS), 18, 65).round().astype(int),
        "acquisition_channel": rng.choice(channel_names, N_CUSTOMERS, p=channel_p),
    })
    return customers, signup_idx


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------
def build_orders(customers, signup_idx, dates, fac):
    n_days = len(dates)
    mult = customers["acquisition_channel"].map(
        {k: v[1] for k, v in CHANNELS.items()}).to_numpy()
    p_repeat = np.clip(0.55 * mult, 0, 0.85)
    is_repeater = rng.random(N_CUSTOMERS) < p_repeat
    rate_per_year = rng.gamma(2.0, 1.6, N_CUSTOMERS) * mult

    pairs = []
    for i in range(N_CUSTOMERS):
        first = min(signup_idx[i] + int(rng.integers(0, 4)), n_days - 1)
        pairs.append((i, first))
        if not is_repeater[i]:
            continue
        life = rng.exponential(LIFETIME_DAYS)
        gap_mean = 365 / max(rate_per_year[i], 0.3)
        t = first
        while True:
            t += max(1, int(rng.exponential(gap_mean)))
            if t >= n_days or t - first > life:
                break
            if rng.random() < min(1.0, fac[t] / 1.5):   # seasonal thinning
                pairs.append((i, t))

    orders = pd.DataFrame(pairs, columns=["cust_idx", "day_idx"])
    orders = orders.sort_values(["day_idx", "cust_idx"]).reset_index(drop=True)
    orders["order_id"] = [f"ORD-{i:06d}" for i in range(1, len(orders) + 1)]
    orders["customer_id"] = customers["customer_id"].to_numpy()[orders["cust_idx"]]
    orders["order_date"] = dates[orders["day_idx"].to_numpy()]
    return orders


def build_items(orders, customers, products):
    cats = list(CATALOG)
    cat_p = np.array([CATALOG[c][0] for c in cats])
    cat_p = cat_p / cat_p.sum()
    fav = rng.choice(cats, N_CUSTOMERS, p=cat_p)          # favourite category

    by_cat = {}
    for c in cats:
        sub = products[products["category"] == c]
        w = 1 / sub["list_price"].to_numpy() ** 0.4        # cheaper sells more
        by_cat[c] = (sub, w / w.sum())

    def discount_probs(month):
        if month == 11:
            return [0.15, 0.10, 0.20, 0.25, 0.20, 0.10]
        if month == 12:
            return [0.30, 0.15, 0.20, 0.15, 0.12, 0.08]
        return [0.45, 0.20, 0.15, 0.10, 0.07, 0.03]

    discounts = [0, 5, 10, 15, 20, 30]
    items = []
    for row in orders.itertuples(index=False):
        n_items = rng.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05])
        chosen = set()
        for _ in range(n_items):
            cat = fav[row.cust_idx] if rng.random() < 0.5 else rng.choice(cats, p=cat_p)
            sub, w = by_cat[cat]
            pid = sub["product_id"].to_numpy()[rng.choice(len(sub), p=w)]
            if pid in chosen:
                continue
            chosen.add(pid)
            prod = sub[sub["product_id"] == pid].iloc[0]
            if prod["category"] in ("Groceries", "Books & Stationery"):
                qty = int(rng.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05]))
            elif prod["list_price"] > 500_000:
                qty = 1
            else:
                qty = int(rng.choice([1, 2], p=[0.85, 0.15]))
            price = int(round(prod["list_price"] * rng.uniform(0.95, 1.08), -3))
            items.append({
                "order_id": row.order_id,
                "product_id": pid,
                "quantity": qty,
                "unit_price": max(price, 1000),
                "discount_pct": int(rng.choice(discounts,
                                               p=discount_probs(row.order_date.month))),
            })
    return pd.DataFrame(items)


def finish_orders(orders, items, customers, products):
    """Add status, payment method, shipping and delivery time."""
    gross = (items.assign(v=items["quantity"] * items["unit_price"]
                          * (1 - items["discount_pct"] / 100))
             .groupby("order_id")["v"].sum())
    has_fashion = set(items.merge(products[["product_id", "category"]], on="product_id")
                      .query("category == 'Fashion'")["order_id"])
    region_of = customers.set_index("customer_id")["region"]

    methods = ["Payme", "Click", "Card (Uzcard/Humo)", "Cash on delivery", "Installment"]
    pay, status, ship, days = [], [], [], []
    for row in orders.itertuples(index=False):
        total = gross.get(row.order_id, 0)
        if total > 1_500_000:
            p = [0.20, 0.17, 0.13, 0.15, 0.35]
        else:
            p = [0.28, 0.25, 0.21, 0.22, 0.04]
        m = rng.choice(methods, p=np.array(p) / sum(p))
        pay.append(m)

        if m == "Cash on delivery":
            sp = [0.84, 0.06, 0.10]
        else:
            sp = [0.89, 0.05, 0.06]
        if row.order_id in has_fashion:               # size issues -> more returns
            sp = [sp[0] - 0.05, sp[1] + 0.05, sp[2]]
        s = rng.choice(["Delivered", "Returned", "Cancelled"], p=sp)
        status.append(s)

        region = region_of[row.customer_id]
        free_from = 300_000 if region == "Tashkent City" else 500_000
        base_fee = 15_000 if region == "Tashkent City" else 25_000
        ship.append(0 if total >= free_from else base_fee)

        if s == "Cancelled":
            days.append(np.nan)
        else:
            d = rng.normal(DELIVERY_MEAN_DAYS[region], 0.8)
            days.append(max(1, int(round(d))))

    out = orders[["order_id", "customer_id", "order_date"]].copy()
    out["status"] = status
    out["payment_method"] = pay
    out["shipping_fee"] = ship
    out["delivery_days"] = days
    return out


# --------------------------------------------------------------------------
# Make the raw files messy on purpose
# --------------------------------------------------------------------------
REGION_VARIANTS = {
    "Tashkent City": ["Toshkent shahri", "tashkent city", "Tashkent city "],
    "Samarkand": ["Samarqand", "samarkand", " Samarkand"],
    "Fergana": ["Farg'ona", "fergana", "Fergana "],
    "Andijan": ["Andijon", "andijan"],
    "Namangan": ["namangan", "Namangan "],
    "Bukhara": ["Buxoro", "bukhara"],
}


def mess_up(customers, orders, items):
    c = customers.copy()
    o = orders.copy()
    it = items.copy()

    # spelling variants in region (~5%)
    c["region"] = c["region"].astype("object")
    for idx in c.sample(frac=0.05, random_state=1).index:
        reg = c.at[idx, "region"]
        if reg in REGION_VARIANTS:
            c.at[idx, "region"] = rng.choice(REGION_VARIANTS[reg])

    # missing / impossible ages
    c["age"] = c["age"].astype("float")
    c.loc[c.sample(frac=0.03, random_state=2).index, "age"] = np.nan
    c.loc[c.sample(frac=0.004, random_state=3).index, "age"] = 0
    c.loc[c.sample(frac=0.002, random_state=4).index, "age"] = 140

    # dates: mostly ISO, a few dd.mm.yyyy
    o["order_date"] = o["order_date"].dt.strftime("%Y-%m-%d")
    alt = o.sample(frac=0.04, random_state=5).index
    o.loc[alt, "order_date"] = pd.to_datetime(
        o.loc[alt, "order_date"]).dt.strftime("%d.%m.%Y")

    # status casing, missing payment method
    case_idx = o.sample(frac=0.02, random_state=6).index
    o.loc[case_idx, "status"] = o.loc[case_idx, "status"].str.upper()
    o.loc[o.sample(frac=0.012, random_state=7).index, "payment_method"] = np.nan

    # duplicated order rows
    dup = o.sample(frac=0.006, random_state=8)
    o = pd.concat([o, dup]).sample(frac=1, random_state=9).reset_index(drop=True)

    # a handful of bad quantities
    bad = it.sample(frac=0.001, random_state=10).index
    it.loc[bad, "quantity"] = rng.choice([0, -1], size=len(bad))

    return c, o, it


def main():
    dates = pd.date_range(START, END)
    fac = np.array([seasonal_factor(d) for d in dates])

    products = build_products()
    customers, signup_idx = build_customers(dates, fac)
    orders_core = build_orders(customers, signup_idx, dates, fac)
    items = build_items(orders_core, customers, products)
    orders = finish_orders(orders_core, items, customers, products)

    customers_raw, orders_raw, items_raw = mess_up(customers, orders, items)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    customers_raw.to_csv(RAW_DIR / "customers.csv", index=False)
    products.to_csv(RAW_DIR / "products.csv", index=False)
    orders_raw.to_csv(RAW_DIR / "orders.csv", index=False)
    items_raw.to_csv(RAW_DIR / "order_items.csv", index=False)

    print(f"customers   : {len(customers_raw):>7,}")
    print(f"products    : {len(products):>7,}")
    print(f"orders      : {len(orders_raw):>7,}  (incl. duplicates)")
    print(f"order items : {len(items_raw):>7,}")
    print(f"saved to    : {RAW_DIR}")


if __name__ == "__main__":
    main()
