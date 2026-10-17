# Online Store Sales Analysis — Uzbekistan E-commerce (2024–2025)

An end-to-end analysis of two years of orders from a mid-sized online store selling to customers across Uzbekistan: data cleaning, SQL, KPI reporting, customer segmentation (RFM) and cohort retention. The goal was to find out **where the money is made, where it leaks, and which customers are worth keeping**.

**Tools:** Python (pandas, NumPy, Matplotlib) · SQL (SQLite, window functions, CTEs)

> **About the data.** Real store transactions are private, so the dataset is **simulated**. `src/generate_data.py` builds it with realistic behaviour (Navruz and 11.11 sales peaks, regional delivery times, repeat buyers, discounts, returns) and then deliberately corrupts it (duplicates, mixed date formats, spelling variants, missing values) so the cleaning step is real work. Figures describe this simulated shop, not a real company.

---

## Business questions

1. How fast is revenue growing and when are the peaks?
2. Which categories bring revenue, and which bring profit?
3. Do discounts pay off?
4. Which regions and acquisition channels are most valuable?
5. How many customers come back, and who are the best ones?
6. Where do orders get lost (cancellations, returns)?

## Headline numbers

| | 2024 | 2025 | Total |
|---|---:|---:|---:|
| Net revenue, UZS million | 1,615 | 2,861 | 4,476 |
| Gross margin | 25.2% | 25.6% | 25.5% |
| Orders placed | 3,714 | 6,550 | 10,264 |
| Average order value, UZS | 505,049 | 503,615 | 504,132 |
| Return rate (of shipped orders) | 7.4% | 6.5% | 6.8% |
| Cancellation rate | 7.0% | 7.2% | 7.2% |

*Net revenue = quantity × price after discount, counted only for delivered orders.*

---

## Key findings

### 1. Revenue grew 77% year over year, with clear seasonal peaks
November is the strongest month in both years (UZS 361M in 2025, +49% on October), driven by the 11.11 sale. March also jumps (+81% month-on-month in 2025) around Navruz, and February is the weakest month. Stock and ad budget should be planned around these dates.

![Monthly revenue](reports/figures/01_monthly_revenue.png)

### 2. Electronics is half of revenue but has the thinnest margin
Electronics brings 48% of net revenue at a 12% gross margin, while Beauty (52%) and Fashion (47%) earn far more per sum sold. Fashion also has the highest return rate (10.0% against 5.6–6.7% elsewhere).

![Category revenue and margin](reports/figures/02_category_revenue_margin.png)

### 3. Deep discounts destroy margin, especially on Electronics and Groceries
Lines sold without a discount earn a 31.5% margin; at 30% off the margin falls to –0.8%. At discounts of 20% and above, Electronics (–6.3%) and Groceries (–3.3%) are sold at a loss, while Beauty and Fashion stay profitable. The 30%-off lines in those two categories lost about UZS 15M in gross profit.

![Discount vs margin](reports/figures/04_discount_vs_margin.png)

### 4. Referral customers come back twice as often as paid-ad customers
Only 26.8% of buyers ordered more than once. Referral customers repeat at 38.8% and spend about UZS 1.07M each, against 20.2% and UZS 0.71M for Google Ads. Instagram is the largest channel (1,707 buyers) but repeats at only 22.6%.

![Repeat rate by channel](reports/figures/05_channel_repeat_rate.png)

### 5. A small group of customers carries the business
The top 10% of customers generate 48% of revenue and the top 20% generate 66%. In the RFM segmentation, *Champions* are 5.3% of customers but 20.6% of revenue. At the other end, 37% of customers are lost or hibernating. A further 141 "valuable at risk" customers (4 orders on average, silent for about 14 months) are worth a win-back campaign.

![RFM segments](reports/figures/06_rfm_segments.png)

### 6. Retention is low and flat
Across the 2024 cohorts, only 6–7% of a cohort buys in any given month during the first half-year, falling to about 3–4% by month 12. The average gap between two orders of the same customer is 79 days, so a reminder at about 60 days is a natural trigger.

![Cohort retention](reports/figures/07_cohort_retention.png)

### 7. Cash on delivery has the highest cancellation rate
10.0% of cash-on-delivery orders are cancelled, against 6.0–6.7% for Click, card, Payme and installment orders. Installment plans carry the largest baskets (average UZS 1.39M, about three times the rest).

### 8. Regions: Tashkent is the volume, but customer value is similar everywhere
Tashkent City provides 28.9% of revenue, but revenue per buyer is within a narrow band across regions (UZS 765k–885k). Growth outside the capital therefore depends on acquiring more customers, not on higher spend. Delivery takes 1.3 days in Tashkent City and 3–4 days in other regions; Namangan has the highest cancellation rate (8.2%).

![Region value](reports/figures/03_region_revenue_per_customer.png)

---

## Recommendations

| Finding | Suggested action |
|---|---|
| Discounts of 20–30% lose money on Electronics and Groceries | Cap discounts at 15% for these categories; keep deep discounts for Beauty and Fashion |
| Referral customers repeat most | Start a referral bonus; shift part of the Google Ads budget to it |
| Cash on delivery cancels most | Offer a small prepayment incentive or confirm COD orders by phone/Telegram |
| Fashion returns are high | Add size guides and fit notes on product pages |
| 141 valuable customers are inactive | Send a win-back offer to the "At risk" RFM segment |
| Peaks in November and March | Build stock and campaign calendar around 11.11 and Navruz |

---

## Data and cleaning

Four tables: `customers` (6,000), `products` (60), `orders` (10,335 raw rows) and `order_items` (16,575).

| Issue found in raw data | Rows | Fix |
|---|---:|---|
| Region spelled several ways (`Toshkent shahri`, `samarkand `, …) | 240 | Mapped to one standard name |
| Impossible ages (0, 140) | 36 | Set to missing |
| Missing ages | 215 | Left as missing, not imputed |
| Duplicate order rows | 62 | Removed |
| Dates as `dd.mm.yyyy` instead of ISO | 411 | Converted |
| Status in inconsistent case (`DELIVERED`) | 205 | Normalised |
| Missing payment method | 123 | Labelled `Unknown` |
| Quantity of 0 or less | 17 | Rows dropped |
| Orders left with no valid item | 9 | Dropped |

The full log is in `reports/tables/cleaning_log.csv`.

## SQL

`sql/queries.sql` answers eight business questions using CTEs and window functions (`LAG`, `ROW_NUMBER`, `NTILE`, `RANK`): month-over-month growth and running totals, top products per category, customer revenue deciles, days between orders, repeat rate by first-purchase category, region ranking, cancellations by payment method and losses on deep discounts. The results match the pandas calculations, which served as a cross-check.

## Project structure

```
├── data/
│   ├── raw/                 # generated, intentionally messy CSV files
│   └── processed/           # cleaned tables used for analysis
├── sql/
│   └── queries.sql          # business questions in SQL
├── src/
│   ├── generate_data.py     # builds the simulated dataset
│   ├── clean_data.py        # cleaning and feature calculation
│   ├── analysis.py          # KPIs, segmentation, cohorts, charts
│   └── run_sql.py           # loads SQLite and runs queries.sql
├── reports/
│   ├── figures/             # PNG charts used above
│   └── tables/              # CSV outputs behind every number
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/uzbek-ecommerce-analysis.git
cd uzbek-ecommerce-analysis
pip install -r requirements.txt

python src/generate_data.py   # 1. create raw data
python src/clean_data.py      # 2. clean it
python src/analysis.py        # 3. tables and charts
python src/run_sql.py         # 4. SQL results
```

The random seed is fixed, so the numbers above can be reproduced exactly.

## Limitations

- The data is simulated; growth, seasonality and customer behaviour reflect the assumptions built into the generator, not a real market.
- Cohort retention is measured on delivered orders only, and the 2025 cohorts are too young for a full 12-month view, so only the 2024 cohorts are shown.
- Gross profit uses a fixed unit cost per product and ignores shipping, advertising and payment fees.
- The recommendations are directions to test (for example with an A/B test), not proven effects.

## Author

**<Your Name>** — Data Analyst · [LinkedIn](https://www.linkedin.com/in/<your-profile>) · <your-email>
