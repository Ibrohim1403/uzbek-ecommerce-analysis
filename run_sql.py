"""
Load the cleaned CSVs into a SQLite database and run sql/queries.sql.

Each query's result is saved to reports/tables/sql_<query_name>.csv.

Usage:
    python src/run_sql.py
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "ecommerce.db"
QUERIES = ROOT / "sql" / "queries.sql"
TABLES = ROOT / "reports" / "tables"


def build_database() -> sqlite3.Connection:
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    for name in ["customers", "orders", "order_lines", "products"]:
        pd.read_csv(PROC / f"{name}.csv").to_sql(name, con, index=False)
    con.execute("CREATE INDEX idx_lines_order ON order_lines(order_id)")
    con.execute("CREATE INDEX idx_lines_customer ON order_lines(customer_id)")
    con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    return con


def split_queries(text: str) -> dict:
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    con = build_database()
    queries = split_queries(QUERIES.read_text(encoding="utf-8"))

    for name, sql in queries.items():
        result = pd.read_sql_query(sql, con)
        result.to_csv(TABLES / f"sql_{name}.csv", index=False)
        print(f"\n--- {name} ({len(result)} rows) ---")
        print(result.head(12).to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
