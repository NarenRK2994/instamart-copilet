"""Run all reproducible stages and write the required project artifacts."""
from pathlib import Path
import csv
import json
import sqlite3

import pandas as pd

import mock_llm
from src.data_pipeline import audit, clean_data, validate
from src.customer_models import rfm_and_segments, build_lapse_model
from src.triage import run_pipeline, FEW, ZERO
from src.policy_rag import PolicyRAG, evaluate
from src.refund_agent import run_agent, resume_agent

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
STATE = ROOT / "state"


def build_agent_scenarios():
    with sqlite3.connect(ROOT / "data" / "instamart.db") as con:
        orders = pd.read_sql_query("SELECT * FROM orders ORDER BY order_id", con)
    eligible = orders[orders.status.isin(["completed", "cancelled"])].copy()
    late_small = eligible[(eligible.status == "completed") & (eligible.delivery_minutes > 45)
                          & (eligible.order_amount <= 1000)].head(2)
    late_big = eligible[(eligible.status == "completed") & (eligible.delivery_minutes > 45)
                        & (eligible.order_amount > 1000)].iloc[0]
    on_time = eligible[(eligible.status == "completed") & (eligible.delivery_minutes <= 45)].iloc[0]
    cancelled = eligible[eligible.status == "cancelled"].iloc[0]
    scenarios = []
    for row in late_small.itertuples(index=False):
        scenarios.append((row.customer_id, f"My order {row.order_id} reached me very late. I would like a refund.",
                          "REFUND", float(row.order_amount)))
    scenarios.extend([
        (late_big.customer_id, f"Order {late_big.order_id} arrived far too late, please refund me.",
         "ESCALATE", float(late_big.order_amount)),
        (on_time.customer_id, f"Please refund order {on_time.order_id}, I did not like the delivery.", "DENIED", 0.0),
        (cancelled.customer_id, f"I want my money back for order {cancelled.order_id}.", "DENIED", 0.0),
        ("C0001", "Refund for order ORD-999999 please.", "NOT_FOUND", 0.0),
        (late_small.iloc[0].customer_id, f"[LOOP] Refund order {late_small.iloc[0].order_id}.", "FALLBACK", 0.0),
    ])
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "agent_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario_id", "customer_id", "message", "expected_outcome", "expected_refund_inr"])
        for number, (customer_id, message, outcome, amount) in enumerate(scenarios, 1):
            writer.writerow([f"S{number}", customer_id, message, outcome, f"{amount:.2f}"])


def part_a_analysis():
    customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")
    orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
    products = pd.read_csv(ROOT / "data" / "clean" / "products.csv")
    completed = orders[orders.status.eq("completed")].copy()
    joined = completed.merge(customers[["customer_id", "city"]], on="customer_id")
    joined = joined.merge(products[["product_id", "category"]], on="product_id")
    pivot = pd.pivot_table(joined, values="order_amount", index="city", columns="category",
                           aggfunc="sum", fill_value=0, margins=True)
    city_summary = joined.groupby("city").agg(
        sum=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count"))
    amount = completed.order_amount
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    queries = []
    for section in (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8").split(";"):
        query = "\n".join(line for line in section.splitlines() if not line.strip().startswith("--")).strip()
        if query:
            queries.append(query)
    query_results = []
    with sqlite3.connect(ROOT / "data" / "instamart.db") as con:
        for query in queries:
            query_results.append(pd.read_sql_query(query, con))
        average_spend = pd.read_sql_query(
            "SELECT AVG(total_spend) AS average_spend FROM "
            "(SELECT customer_id, SUM(order_amount) AS total_spend FROM orders "
            "WHERE status='completed' GROUP BY customer_id)", con).iloc[0, 0]
    return {
        "sql_query_rows": [len(frame) for frame in query_results],
        "sql_q5_average_spend": float(average_spend),
        "pivot_grand_total": float(pivot.loc["All", "All"]),
        "city_summary": city_summary,
        "amount_stats": {"mean": float(amount.mean()), "median": float(amount.median()),
                         "skew": float(amount.skew()), "q1": float(q1), "q3": float(q3),
                         "lower_fence": float(q1 - 1.5 * iqr), "upper_fence": float(q3 + 1.5 * iqr),
                         "below_outliers": int((amount < q1 - 1.5 * iqr).sum()),
                         "above_outliers": int((amount > q3 + 1.5 * iqr).sum())},
        "correlation": completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr(),
    }


def run_agent_suite():
    OUT.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    (OUT / "agent_runs.jsonl").write_text("", encoding="utf-8")
    (OUT / "refund_ledger.jsonl").write_text("", encoding="utf-8")
    for checkpoint in STATE.glob("*.json"):
        checkpoint.unlink()
    scenarios = pd.read_csv(RAW / "agent_scenarios.csv", keep_default_na=False)
    results = []
    for row in scenarios.itertuples(index=False):
        mock_llm.reset()
        result = run_agent(row.message, row.scenario_id)
        expected = row.expected_outcome
        assert result["status"] == expected or (expected == "FALLBACK" and result["status"].startswith("FALLBACK"))
        if expected in {"REFUND", "ESCALATE"}:
            observed_amount = float(result["steps"][-1]["observation"]["refund_inr"])
            assert abs(observed_amount - float(row.expected_refund_inr)) <= 0.01
        record = {key: result[key] for key in ("status", "final_answer", "llm_calls", "tokens_used")}
        record["scenario_id"] = row.scenario_id
        record["steps"] = result["steps"]
        with (OUT / "agent_runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        amount = next((step["observation"].get("refund_inr", 0) for step in result["steps"]
                       if step["tool"] in {"issue_refund", "escalate_to_human"}), 0)
        results.append({"scenario_id": row.scenario_id, "status": result["status"], "refund_inr": amount,
                        "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"]})

    s1 = scenarios.iloc[0]
    mock_llm.reset()
    token_demo = run_agent(s1.message, "S1-token-demo", token_limit=200)
    mock_llm.reset()
    timeout_demo = run_agent(s1.message, "S1-timeout-demo", time_limit_seconds=0)

    mock_llm.reset()
    run_id = "S1-crash-resume"
    try:
        run_agent(s1.message, "S1", run_id=run_id, crash_after=2)
    except RuntimeError as error:
        assert "Simulated crash" in str(error)
    mock_llm.reset()
    resumed = resume_agent(run_id)
    assert resumed["status"] == "REFUND"
    assert [step["tool"] for step in resumed["steps"]] == [
        "get_order", "search_policy", "calculate_refund", "issue_refund"]
    return results, token_demo, timeout_demo, resumed


def main():
    OUT.mkdir(exist_ok=True)
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    audit(); clean_data(); validate()
    analysis = part_a_analysis()
    print("SQL query rows:", analysis["sql_query_rows"])
    print("Q5 average customer spend:", analysis["sql_q5_average_spend"])
    print("Pandas pivot grand total:", analysis["pivot_grand_total"])
    print("City spend summary:\n", analysis["city_summary"].round(2).to_string())
    print("Completed order amount statistics:", analysis["amount_stats"])
    print("Completed order correlations:\n", analysis["correlation"].round(4).to_string())
    print("\n=== PART B: SEGMENTS ===")
    rfm, diagnostics, profile, pca = rfm_and_segments()
    print("K-means diagnostics:\n", diagnostics.to_string(index=False))
    print("Segment profile:\n", profile.to_string(index=False))
    print("PCA explained variance:", pca, "total:", pca.sum())
    lapse = build_lapse_model()
    print("Lapse summary:", json.dumps({k: v for k, v in lapse.items() if k != "retention_list"}, default=str))
    print("\n=== PART C: TRIAGE ===")
    zero, zero_summary = run_pipeline(prompt=ZERO, token_budget=100000)
    few, few_summary = run_pipeline(prompt=FEW, token_budget=100000)
    budget, budget_summary = run_pipeline(prompt=ZERO, token_budget=6000)
    print("Zero-shot:", zero_summary)
    print("Few-shot:", few_summary)
    print("Budget demo exhausted:", int(budget.status.eq("budget_exhausted").sum()))
    print("\n=== PART D: POLICY RAG AND REFUNDS ===")
    rag = PolicyRAG()
    print("Chunks:", len(rag.chunks))
    print("Overlapping evaluation:\n", evaluate(rag).to_string(index=False))
    no_overlap = PolicyRAG(overlap=0)
    print("Zero-overlap evaluation:\n", evaluate(no_overlap).to_string(index=False))
    build_agent_scenarios()
    scenarios, token_demo, timeout_demo, resumed = run_agent_suite()
    print("Agent scenarios:\n", pd.DataFrame(scenarios).to_string(index=False))
    print("Token demo:", token_demo["status"], "timeout demo:", timeout_demo["status"])
    print("Crash/resume:", resumed["status"], resumed["steps"])
    print("Project pipeline finished. Review outputs/, data/instamart.db, and state/.")


if __name__ == "__main__":
    main()
