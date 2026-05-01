# Performance Evaluation Scripts

This directory contains standalone scripts that **measure the predictive
performance** of PackPal’s AI components using real or simulated data.
They complement the unit tests in `tests/` by producing **quantitative
metrics** (RMSE, accuracy, Precision@1, etc.) that are reported in the
project documentation.

## Purpose in the Project

The evaluation scripts are the primary source of empirical evidence for
**Section (Evaluation)** of the project report. Each script implements
one of the evaluation methodologies recommended by the course:

- **Offline Ranking Metrics** (Precision@K, Accuracy, F1)
- **Historical Simulation** (backtesting against withheld data)
- **Within‑subject Experiment** (for the KG layering logic, simulated via
  keyword matching)

For human‑subject experiments (SHAP/LIME preference studies), we rely on
user surveys; those results are summarised directly in the report.

## How to Run

From the **project root** (where `streamlit_app.py` lives), execute each
script individually:

    python evaluation/eval_weather.py
    python evaluation/eval_recommender.py
    python evaluation/eval_slots.py
    python evaluation/eval_slots_llm.py
    python evaluation/eval_kg.py
    python evaluation/eval_optimizer.py

Each script prints its results to the console. No additional arguments or
API keys are required—they use synthetic data or pre‑defined test sets.

## Script Descriptions

| Script | Module Evaluated | Evaluation Method | What It Does | Output Example | Report Reference |
|--------|------------------|-------------------|--------------|----------------|-----------------|
| `eval_weather.py` | `historical_forecast.py` | Historical Simulation | Generates synthetic temperature series, withholds one year, compares Theil‑Sen vs OLS RMSE | `Theil‑Sen RMSE: 0.192, OLS RMSE: 0.435, Reduction: 55.9%` | Section 6, Weather rows |
| `eval_recommender.py` | `recommender.py` | Offline Ranking | Trains KNN on 400 synthetic samples, tests on 100, computes mean accuracy and F1 | `Mean accuracy: 0.912, Mean F1: 0.886` | Section 6, Recommender row |
| `eval_slots.py` | `slot_detection.py` (fallback) | Offline Ranking (Precision@1) | Runs fallback extraction on 3 known utterances, counts correct slot values | `Precision@1: 0.94 (9/10)` | Section 6, Slot Detection row |
| `eval_slots_llm.py` | `slot_detection.py` (LLM + SBERT) | Offline Ranking (Precision@1) | Calls the full Groq + SBERT slot‑detection pipeline on known utterances, counts correct slot values | `Precision@1: 1.00 (10/10)` | Section 6, Slot Detection row |
| `eval_kg.py` | `kg_rules.py` (fallback) | Within‑subject (keyword match) | Checks 5 temperature bands for expected keywords in layering advice | `Accuracy: 5/5` | Section 6, KG/Layering rows |
| `eval_optimizer.py` | `packing_optimizer.py` (knapsack) | Historical Simulation (constraint) | Generates 50 random trip configurations, verifies weight limit compliance | `Success rate: 98.0% (49/50)` | Section 6, Optimizer row |

### `eval_weather.py`

- **What it does:** Creates a synthetic 10‑year temperature series with a known
  warming trend and Gaussian noise. Holds out 2024 and predicts it using both
  Theil‑Sen and OLS, then computes the RMSE for each.
- **Why it’s useful:** Demonstrates the Theil‑Sen estimator’s robustness compared
  to OLS, providing empirical evidence for the choice of default weather prediction
  method.
- **Output:** RMSE values and the percentage of error variance reduction.

### `eval_recommender.py`

- **What it does:** Generates a small synthetic dataset (500 samples) with ground‑truth
  labels from the expert rule function. Trains a KNN model on 400 samples and
  evaluates accuracy and F1 on the remaining 100.
- **Why it’s useful:** Gives a realistic estimate of the recommender’s performance
  on synthetic data, which can be compared against the rule‑based baseline.
- **Output:** Mean accuracy and mean F1 score across all 38 items.

### `eval_slots.py`

- **What it does:** Defines 3 natural‑language utterances with known expected slots
  and runs the fallback regex extractor on each. Counts how many slot values are
  correctly extracted.
- **Why it’s useful:** Quantifies the reliability of the slot‑detection fallback,
  which is critical for system uptime when the LLM is unavailable.
- **Output:** Precision@1 as a decimal (e.g., 0.94).

### `eval_slots_llm.py`

- **What it does:** Calls the full `extract_slots()` function – the same
  LangChain + Groq + SBERT pipeline used in the live Streamlit app – on a
  set of known user utterances. Compares every extracted slot against the
  expected value.
- **Why it’s useful:** Provides a real‑world Precision@1 measurement for
  the primary AI‑powered slot‑detection path (not the fallback). This is
  the number that reflects the actual user experience in the GUI.
- **Output:** Precision@1 as a decimal (e.g., 1.00 for 10/10).

### `eval_kg.py`

- **What it does:** Tests the layering advice fallback for 5 distinct temperature
  bands, checking whether the expected keywords appear in the returned advice string.
- **Why it’s useful:** Validates that the KG‑derived (or fallback) layering rules
  are sensible across the full temperature spectrum.
- **Output:** Accuracy as a fraction (e.g., 5/5).

### `eval_optimizer.py`

- **What it does:** Generates 50 randomised trip configurations, selects 10‑20
  random items, and runs the knapsack solver. Verifies that the selected items
  never exceed the weight limit.
- **Why it’s useful:** Provides a constraint‑satisfaction success rate that can be
  reported as a reliability metric for the optimisation pipeline.
- **Output:** Success rate as a percentage (e.g., 98.0%).

## How to Integrate Results into the Project Report

1. Run each script and record the printed numbers.
2. In the project report’s **Section 6 (Evaluation)**, locate the
   corresponding rows in the results table.
3. Replace the placeholder or simulated numbers with the actual outputs.
4. In **Appendix F (Automated Test Suite)**, include a brief note that the
   results shown are the output of these evaluation scripts.

## Dependencies

- numpy, pandas, scikit‑learn (already in `requirements.txt`)
- No API keys or network access required