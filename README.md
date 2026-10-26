# Instamart Customer Intelligence & Support Copilot

## Overview
All generated data is synthetic and fictional. This repository implements the data pipeline, customer segmentation and lapse model, privacy-aware ticket triage, policy retrieval, and a bounded refund agent. LLM behavior is supplied by the local `mock_llm.py`; the project uses no API key, paid service, or network call.

## Reproduce
Run these numbered commands from the repository root. Python 3.11+ is required. On Windows PowerShell use the activation command shown; on macOS/Linux use the alternative.

1. `python -m venv .venv`
2. Windows: `.\.venv\Scripts\Activate.ps1`  |  macOS/Linux: `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python -m pip freeze` (the committed requirements file records the isolated environment's frozen packages)
6. `python generate_data.py`
7. `python run_project.py`

The generator prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The runner audits, cleans, validates, runs all seven SQL queries, trains and saves the models, evaluates both triage prompts and both RAG chunking settings, and creates the seven refund scenarios and required outputs. The teaching-version `generate_data.py` intentionally does not emit `agent_scenarios.csv`; `run_project.py` derives it deterministically from the clean database so the supplied generator remains unmodified. The generator and `mock_llm.py` are preserved as supplied in this workspace.

## Part A
### Raw audit
Raw CSVs were loaded with `keep_default_na=False`; therefore blank strings are counted as blanks and parsed pandas NA counts are zero. Every listed field below has 0 parsed missing values (0.00%).

| Table | Shape | Dtypes |
|---|---:|---|
| customers | 1,200 × 5 | `customer_id`, `city`, `signup_date`, `age`, `gender`: str |
| products | 40 × 4 | `product_id`, `product_name`, `category`: str; `price_inr`: int64 |
| orders | 13,084 × 9 | `order_id`, `customer_id`, `product_id`, `order_ts`, `status`, `order_amount`, `delivery_minutes`: str; `quantity`, `discount_pct`: int64 |
| tickets | 164 × 5 | all five fields: str; empty text remains `""` |

| Table | Per-column blank counts (and percentage of rows) |
|---|---|
| customers | `customer_id` 0 (0.00%); `city` 0 (0.00%); `signup_date` 0 (0.00%); `age` 61 (5.08%); `gender` 30 (2.50%) |
| products | `product_id` 0 (0.00%); `product_name` 0 (0.00%); `category` 0 (0.00%); `price_inr` 0 (0.00%) |
| orders | `order_id` 0 (0.00%); `customer_id` 0 (0.00%); `product_id` 0 (0.00%); `quantity` 0 (0.00%); `order_ts` 0 (0.00%); `status` 0 (0.00%); `discount_pct` 0 (0.00%); `order_amount` 0 (0.00%); `delivery_minutes` 2,541 (19.42%) |
| tickets | `ticket_id` 0 (0.00%); `customer_id` 0 (0.00%); `order_id` 8 (4.88%); `text` 4 (2.44%); `true_category` 0 (0.00%) |

| Raw audit check | Result |
|---|---:|
| Orders rows / exact duplicate rows / pending rows / distinct order IDs | 13,084 / 119 / 656 / 12,309 |
| Non-numeric amounts / negative delivery / blank delivery | 120 / 106 / 2,541 |
| Customer blank age / age outside 10–100 / city spellings | 61 / 6 / 24 |
| Distinct nonblank gender spellings / blank gender | 10 / 30 |

### Cleaning and validation
Customer city values were stripped and title-cased, reducing 24 spellings to six: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender variants map to female 610, male 560, unknown 30. Six ages outside 10–100 and 61 blank ages were treated as missing; city median was filled using `groupby(...).transform('median')`, leaving 0 missing ages with range 18–60. `signup_date` is parsed to datetime then stored as ISO date text.

Orders are parsed to datetime first, then 119 exact duplicate rows removed. Keeping the latest timestamp for each order ID removes the 656 earlier pending records and leaves 12,309 final rows (11,110 completed and 1,199 canceled). Of the 120 raw nonnumeric amounts, 112 remain after deduplication; those are coerced to missing and imputed from catalogue price × quantity × (1 − discount/100), rounded to two decimals. The 106 negative delivery values become missing; among completed final orders, 774 missing delivery values are filled by completed-order product-category medians. Canceled delivery stays missing for all 1,199 canceled orders. Completed revenue is INR 2,998,668.80. Clean CSVs and SQLite tables (`customers`, `products`, `orders`) are written; `order_ts` is stored as `YYYY-MM-DD HH:MM:SS`. `validate()` asserts more than eight exact checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL and pandas results
All seven queries in [`sql/queries.sql`](sql/queries.sql) are run through `pandas.read_sql_query` against `data/instamart.db`.

| Query | Result |
|---|---|
| Q1 WHERE/HAVING | 30 customers |
| Q2 three-table city/category revenue | 30 cells; total INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 products above average catalogue price | 11 products |
| Q4 customers without completed orders | 25 customers |
| Q5 two-CTE above-average spend | 550 customers; cohort average INR 2,552.06 |
| Q6 city ranking | Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10; 8 rows have `RANK() != ROW_NUMBER()` |
| Q7 daily running revenue | 183 days; final cumulative INR 2,998,668.80 |

Tied spend values share a `RANK()` position, while `ROW_NUMBER()` assigns every row a unique position, so ties make the values differ.

The pandas city/category pivot with margins totals INR 2,998,668.80, matching SQL. Per-city completed order amount: Bengaluru sum INR 523,643.40, mean INR 276.47, count 1,894; Chennai INR 463,629.15, INR 272.72, 1,700; Delhi INR 502,829.80, INR 243.62, 2,064; Hyderabad INR 528,589.90, INR 284.95, 1,855; Mumbai INR 479,058.50, INR 297.00, 1,613; Pune INR 500,918.05, INR 252.48, 1,984.

Completed order amount mean is INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80 and Q3 INR 297.00. The 1.5×IQR fences are INR -283.50 and INR 645.30; there are 0 lower and 897 upper outliers. This is completed-order spending, with a strongly right-skewed distribution: most orders are modest and a smaller group of high-value baskets lifts the mean above the median. The upper-tail outliers are largely plausible high-value purchases rather than proof of bad records; the raw pipeline separately repairs malformed amounts and impossible delivery values. The largest absolute correlation among amount, discount, quantity and delivery time is order amount versus discount at -0.095 (amount versus quantity is +0.087). This association is weak, and correlation of this size does not establish causation.

## Part B
RFM is calculated for completed orders as of 2025-10-01: 1,175 customers; recency median 19 days, range 0–180; frequency mean 9.46, max 33; monetary mean INR 2,552.06, max INR 14,766.55.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is chosen: it has the strongest silhouette and a clear elbow before diminishing returns. K=8 necessarily has lower inertia because adding clusters can only reduce within-cluster squared distances; inertia alone rewards ever more clusters and does not measure whether the groups are stable or useful.

| Segment ID | Name | Customers | Mean recency | Mean frequency | Mean monetary | Action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional Value | 436 | 28.86 days | 3.77 | INR 1,057.62 | Use relevant low-friction replenishment offers. |
| 1 | Premium High-Value (Premium) | 138 | 24.12 days | 6.77 | INR 7,129.70 | Protect high basket value with tailored service and retention offers. |
| 2 | Loyal Frequent | 419 | 8.77 days | 19.40 | INR 3,410.02 | Encourage category discovery without over-discounting. |
| 3 | Cooling Off | 182 | 109.04 days | 2.20 | INR 685.99 | Prioritize a win-back message and monitor response. |

Cluster numbers are arbitrary labels, not an ordered scale. PCA explained variance is 0.634 and 0.218 (total 0.852).

The lapse cohort contains 1,156 customers; 443 lapsed (38.32%). Features use only final-status orders before 2025-09-01 plus the clean customer table; labels use completed orders during September. The features are recency, frequency, monetary, average discount, cancellation rate, average delivery, favourite category, city, age, gender and tenure. Task 3 segment labels were not features. The 924/232 stratified train/test split uses random_state=42. All feature columns are asserted complete. Numeric scaling and k−1 nominal one-hot encoding are inside the training Pipeline; no encoder, scaler or resampler saw test rows. The fitted training scaler's frequency mean is 8.115 versus full-cohort mean 8.029 because the scaler is fitted only on the sampled training subset.

| Model | Search result | CV F1 | Test confusion matrix (TN, FP; FN, TP) | Test precision | Test recall | Test F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | `C=100` | 0.651 | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | `max_depth=8`, `min_samples_leaf=10` | 0.687 | [[110, 33], [32, 57]] | 0.633 | 0.640 | 0.637 |

Grid searches were LogisticRegression(max_iter=1000), C in {0.01, 0.1, 1, 10, 100}, and RandomForestClassifier(n_estimators=100, random_state=42), depth in {3, 5, 8} and leaf size in {5, 10, 20}, each with five-fold F1 scoring. The baseline always predicts not lapsed: accuracy 0.616 and lapsed-class F1 0.000. The forest is persisted as the selected model because it has the higher cross-validated F1 and provides nonlinear feature importance; its untouched test F1 is 0.637, above the 0.55 target. Missing a lapsing customer forfeits a chance to intervene, while targeting a loyal customer wastes campaign capacity and may train customers to expect discounts; the selected model balances both costs rather than maximizing recall alone.

On the training set only, the logistic class-balance experiment duplicates lapsed rows with replacement until the classes are 570 each. On the unchanged test set, original logistic precision/recall/F1 are 0.644/0.652/0.648; balanced values are 0.589/0.854/0.697. Recall moves most, meaning more at-risk customers are surfaced, at the cost of more loyal customers receiving retention outreach.

Random-forest top importances: monetary 0.280, frequency 0.278, recency 0.109, average delivery 0.060, cancellation rate 0.052. Logistic coefficient signs: frequency negative (higher repeat activity predicts lower lapse odds); recency positive (longer inactivity predicts higher lapse odds); cancellation rate negative in this fitted model conditional on other features (an association, not a causal claim). The selected model is saved and reloaded; predictions match exactly. The Premium test-set retention list has 10 customers, of whom 1 actually lapsed.

## Part C
The 164-ticket run skips 4 empty tickets and flags 4 tickets over the 400-token maximum (734 estimated tokens each); neither is sent to the model. Ordered redaction substitutions across the full file: 55 emails, 91 phones and 132 order IDs, 278 total; a second application detects 0 remaining matches. The model user content is exactly the redacted ticket text. A code assertion checks this before each eligible model call.

Both system prompts require JSON with one of billing, delivery, technical or fraud. Few-shot has labelled examples including fraud. Retries use at most three attempts with exponential jitter; each complete batch has 19 transient errors and all recover. Retry logs report ticket, retry number and wait.

| Prompt | Classified | Accuracy | Tokens total | Tokens / classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 126/147 = 85.7% | 7,372 | 50.15 |
| Few-shot | 147 | 147/147 = 100% | 12,520 | 85.17 |

The few-shot prompt costs 35.02 additional tokens per classified ticket and recovers all 21 zero-shot errors (11 fraud and 10 technical tickets predicted billing). That extra cost is worthwhile when routing accuracy matters more than prompt budget, especially for fraud and technical cases; the zero-shot prompt remains cheaper for low-risk, high-volume screening when downstream review can absorb misroutes.

Zero-shot confusion table (rows are true labels; columns are classified predictions):

| True \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

Across the 164 final few-shot rows, statuses are classified_json 135, classified_keyword 12, flagged_invalid_output 9, flagged_long 4 and skipped_empty 4. JSON category values are allow-listed; only malformed text with exactly one allowed category word extracted by `re.findall(r'[a-z]+', ...)` is keyword-accepted. Valid but disallowed categories and other malformed answers go to human review, as do empty/long/budget cases. The 6,000-token zero-shot budget demo stops before a model call for 29 tickets. `outputs/triage_results.csv` contains 164 rows: human review for guarded/invalid cases, fraud to security at HIGH priority, Premium customers to the premium desk at HIGH priority, and other validated classifications to the support queue.

## Part D
The five policy files yield 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2), each 40 words with 10-word overlap. `TfidfVectorizer` represents term frequency (how often a term appears in a chunk) weighted by inverse document frequency (how rare it is across chunks). TF-IDF is lexical, so equivalent phrasings with different words can retrieve poorly. For “refund policy late delivery” the top three retrieved chunks are all from `refund_policy.txt`.

| Evaluation | Expected document in top 3 | Answer contains expected fact | Unanswerable fallback |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

Without overlap, “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” retrieve the right source, but their answer facts fall across a chunk boundary and are not present in one selected context sentence. The assistant sends a system instruction and the exact `CONTEXT:` / `[source]` / `QUESTION:` user format, and returns its response with the retrieved source names.

Five tools are defined in `TOOLS_SPEC` and dispatched through `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. Refund rules only approve completed orders delivered after 45 minutes; those above INR 1,000 are logged for human approval rather than issued. Each ledger action includes a UTC timestamp and tool name.

| Scenario | Status | Refund INR | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 897 |
| S2 | REFUND | 135.00 | 5 | 897 |
| S3 | ESCALATE | 1,999.00 | 5 | 884 |
| S4 | DENIED | 0.00 | 4 | 605 |
| S5 | DENIED | 0.00 | 3 | 337 |
| S6 | NOT_FOUND | 0.00 | 2 | 72 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,000 |

Stop-safely demonstrations: S1 with token limit 200 returns `FALLBACK_TOKEN_LIMIT` (354 tokens observed, two prior tools completed); a zero-second time limit returns `FALLBACK_TIMEOUT` before any call. S7 reaches all six model calls at the default 6,000-token limit and then returns `FALLBACK_MAX_ITERATIONS`.

The S1 crash is injected after the second tool observation has been checkpointed. Resume completes with REFUND INR 128.25; the combined tool execution order is `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, with no repeated execution. Checkpoints live in `state/`; agent runs and ledger entries are JSONL files under `outputs/`.

ReAct uses 5 model calls / 897 tokens for S1, 3 / 337 for S5, and 2 / 72 for S6 because the model receives each observation before choosing the next action. A plan-and-execute design could use one planning call plus a replan only when an observation invalidates the plan, saving calls on ordinary cases but risking more stale planned actions. ReAct naturally reacts to NOT_FOUND and canceled-order observations before issuing a refund; a plan must explicitly guard every step and replan correctly. For this small refund workflow I would keep ReAct: the extra calls buy straightforward observation-by-observation checks, and the strict iteration, token, timeout, and human-fallback limits contain the cost.

## Limitations and Next Steps
The Premium flag uses whole-window RFM, including September, so the Premium retention list is illustrative and not a leakage-free evaluation; the lapse model’s own features still exclude all data from 2025-09-01 onward. The supplied mock is deterministic rule-based behavior, not a real language model. The database and refund ledger are local teaching artifacts, not production-grade controls or payment processing. Next steps for a production system include access control, durable idempotent transactions, monitoring, schema contracts, temporal validation, drift review, and human approval operations.
