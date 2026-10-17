-- Business questions answered in SQL (SQLite dialect).
-- Tables: customers, orders, order_lines, products  (built from data/processed/)
-- Revenue = net_revenue of lines on orders with status = 'Delivered'.
-- Each block starts with "-- name: <query_name>"; src/run_sql.py runs them all.


-- name: monthly_revenue_mom
-- Monthly revenue, month-over-month growth and running total.
WITH monthly AS (
    SELECT substr(order_date, 1, 7)           AS month,
           ROUND(SUM(net_revenue) / 1e6, 1)   AS revenue_m
    FROM order_lines
    WHERE status = 'Delivered'
    GROUP BY 1
)
SELECT month,
       revenue_m,
       ROUND(100.0 * (revenue_m - LAG(revenue_m) OVER (ORDER BY month))
             / LAG(revenue_m) OVER (ORDER BY month), 1)  AS mom_pct,
       ROUND(SUM(revenue_m) OVER (ORDER BY month), 1)    AS cumulative_revenue_m
FROM monthly
ORDER BY month;


-- name: top3_products_per_category
-- Best three products by revenue inside every category.
WITH product_rev AS (
    SELECT category,
           product_name,
           ROUND(SUM(net_revenue) / 1e6, 1) AS revenue_m,
           SUM(quantity)                    AS units
    FROM order_lines
    WHERE status = 'Delivered'
    GROUP BY category, product_name
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue_m DESC) AS rn
    FROM product_rev
)
SELECT category, rn AS rank, product_name, revenue_m, units
FROM ranked
WHERE rn <= 3
ORDER BY category, rn;


-- name: customer_revenue_deciles
-- How concentrated is revenue? Customers split into 10 equal groups by spend.
WITH cust AS (
    SELECT customer_id, SUM(net_revenue) AS rev
    FROM order_lines
    WHERE status = 'Delivered'
    GROUP BY customer_id
),
bucketed AS (
    SELECT rev, NTILE(10) OVER (ORDER BY rev DESC) AS decile
    FROM cust
)
SELECT decile,
       COUNT(*)                                                AS customers,
       ROUND(SUM(rev) / 1e6, 1)                                AS revenue_m,
       ROUND(100.0 * SUM(rev) / (SELECT SUM(rev) FROM cust), 1) AS revenue_share_pct
FROM bucketed
GROUP BY decile
ORDER BY decile;


-- name: days_between_orders
-- Typical gap between two consecutive delivered orders of the same customer.
WITH ordered AS (
    SELECT customer_id,
           order_date,
           LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_date
    FROM orders
    WHERE status = 'Delivered'
),
gaps AS (
    SELECT julianday(order_date) - julianday(prev_date) AS gap_days
    FROM ordered
    WHERE prev_date IS NOT NULL
)
SELECT COUNT(*)                 AS repeat_orders,
       ROUND(AVG(gap_days), 1)  AS avg_days_between_orders,
       MIN(gap_days)            AS min_days,
       MAX(gap_days)            AS max_days
FROM gaps;


-- name: repeat_rate_by_first_category
-- Does the category of the first purchase predict who comes back?
WITH ranked AS (
    SELECT customer_id,
           category,
           ROW_NUMBER() OVER (PARTITION BY customer_id
                              ORDER BY order_date, net_revenue DESC) AS rn
    FROM order_lines
    WHERE status = 'Delivered'
),
first_cat AS (
    SELECT customer_id, category FROM ranked WHERE rn = 1
),
n_orders AS (
    SELECT customer_id, COUNT(*) AS n
    FROM orders
    WHERE status = 'Delivered'
    GROUP BY customer_id
)
SELECT f.category                                              AS first_purchase_category,
       COUNT(*)                                                AS customers,
       ROUND(100.0 * SUM(CASE WHEN o.n >= 2 THEN 1 ELSE 0 END)
             / COUNT(*), 1)                                    AS repeat_rate_pct
FROM first_cat f
JOIN n_orders o USING (customer_id)
GROUP BY f.category
ORDER BY repeat_rate_pct DESC;


-- name: region_ranking
-- Regions ranked by revenue and by revenue per buying customer.
WITH reg AS (
    SELECT c.region,
           SUM(l.net_revenue)             AS revenue,
           COUNT(DISTINCT l.customer_id)  AS buyers
    FROM order_lines l
    JOIN customers c USING (customer_id)
    WHERE l.status = 'Delivered'
    GROUP BY c.region
)
SELECT region,
       ROUND(revenue / 1e6, 1)                         AS revenue_m,
       buyers,
       ROUND(revenue / buyers / 1e3, 0)                AS revenue_per_buyer_k,
       RANK() OVER (ORDER BY revenue DESC)             AS rank_by_revenue,
       RANK() OVER (ORDER BY revenue / buyers DESC)    AS rank_by_value_per_buyer
FROM reg
ORDER BY rank_by_revenue;


-- name: cancellation_by_payment
-- Which payment methods lead to cancelled or returned orders?
SELECT payment_method,
       COUNT(*)                                                           AS orders,
       ROUND(100.0 * SUM(status = 'Cancelled') / COUNT(*), 1)             AS cancel_rate_pct,
       ROUND(100.0 * SUM(status = 'Returned')  / COUNT(*), 1)             AS return_rate_pct
FROM orders
GROUP BY payment_method
ORDER BY cancel_rate_pct DESC;


-- name: deep_discount_losses
-- Lines sold at 30% off that earned less than they cost.
SELECT category,
       COUNT(*)                          AS lines_at_30pct,
       ROUND(SUM(gross_profit) / 1e6, 1) AS gross_profit_m,
       ROUND(SUM(CASE WHEN gross_profit < 0 THEN gross_profit ELSE 0 END) / 1e6, 1)
                                         AS loss_on_negative_lines_m
FROM order_lines
WHERE status = 'Delivered' AND discount_pct = 30
GROUP BY category
ORDER BY gross_profit_m;
