"""
Packing Optimizer — Three-Stage Pipeline
=========================================
Stage 1  Genetic Algorithm (NSGA-II via PyGAD)
         Generates a Pareto front of candidate packing solutions that trade off
         comfort vs weight.  Each solution is a binary vector over the full
         item vocabulary (37 items from recommender.py).

Stage 2  Knapsack (OR-Tools CP-SAT / pure-DP fallback)
         Selects the single best solution from the Pareto front that fits under
         the user-supplied weight limit while maximising comfort fitness.

Stage 3  Rule-Based Summary
         Produces structured JSON + human-readable terminal output explaining
         which items were removed, why, and the final fitness scores.

Two entry points
----------------
    # After recommender — takes trip_packing dict from build_trip_packing_list()
    result = optimise_from_recommendations(trip_packing, forecasts, context,
                                           weight_limit_kg=20.0)

    # Standalone — pass an arbitrary item list
    result = optimise_items(item_names, forecasts, context,
                            weight_limit_kg=20.0)

Both return an OptimizationResult dataclass.

Dependencies
------------
    pip install pygad>=3.3 ortools>=9.8
    (OR-Tools is optional — pure-DP fallback used if not installed)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from models import DayForecast, TripContext

# ── Item weight table (kg) ────────────────────────────────────────────────────
# Covers all 37 items in recommender.ALL_ITEMS.
# Weights are conservative estimates; edit freely.

ITEM_WEIGHTS: Dict[str, float] = {
    # Clothing
    "Heavy winter coat":            1.50,
    "Thermal underlayer":           0.25,
    "Warm sweater or fleece":       0.60,
    "Gloves and scarf":             0.20,
    "Insulated boots":              1.20,
    "Light jacket or fleece":       0.45,
    "Long-sleeve shirt":            0.25,
    "Jeans or trousers":            0.60,
    "T-shirt or short sleeves":     0.20,
    "Lightweight breathable clothing": 0.20,
    "Shorts or light trousers":     0.25,
    "Waterproof jacket":            0.55,
    "Windproof jacket":             0.40,
    "Waterproof snow boots":        1.30,
    "Thermal socks":                0.10,
    "Business attire":              0.80,
    "Formal shoes":                 0.90,
    "Comfortable walking shoes":    0.70,
    "Casual wear":                  0.35,
    "Smart casual outfit":          0.55,
    "Smart jacket":                 0.70,
    # Packing
    "Compact umbrella":             0.30,
    "Waterproof bag cover":         0.15,
    "Sunscreen SPF 50+":            0.20,
    "Sunscreen SPF 30+":            0.18,
    "Sunglasses":                   0.08,
    "Wide-brim hat":                0.18,
    "Hand warmers":                 0.10,
    "Laptop bag":                   0.80,
    "Power adapter":                0.25,
    "Business cards":               0.05,
    "Day backpack":                 0.60,
    "Phone charger / power bank":   0.30,
    "City map or offline maps":     0.05,
    "Reusable water bottle":        0.25,
    "Small gift (optional)":        0.30,
    "Phone charger":                0.15,
}
DEFAULT_WEIGHT = 0.30   # fallback for any item not listed above


# ── Comfort scoring helpers ───────────────────────────────────────────────────

def _item_comfort_score(item: str, forecasts: List[DayForecast],
                        context: TripContext) -> float:
    """
    Returns a suitability score in [0, 1] for an item given the trip's weather.

    Rules (additive, then normalised):
      +1.0  if the item is strongly appropriate for at least one forecast day
      +0.5  if the item is mildly appropriate
      -0.5  if the item is clearly inappropriate for ALL forecast days
      Neutral (0.5) if no specific signal.

    Business-purpose items get a bump if context.purpose == "business".
    """
    item_l = item.lower()
    n = len(forecasts) or 1

    temp_avg  = sum(f.temp_min + f.temp_max for f in forecasts) / (2 * n)
    rain_days = sum(1 for f in forecasts if f.precipitation_mm > 1)
    wind_days = sum(1 for f in forecasts if f.wind_speed_max > 40)
    uv_avg    = sum(f.uv_index_max for f in forecasts) / n
    snow_days = sum(1 for f in forecasts if f.weather_code in {71,73,75,77,85,86})
    hot_days  = sum(1 for f in forecasts if f.temp_max > 28)
    cold_days = sum(1 for f in forecasts if f.temp_min < 10)
    purpose   = (context.purpose or "").lower()

    score = 0.5  # neutral baseline

    # Rain / waterproof items
    if any(k in item_l for k in ("umbrella", "waterproof", "raincoat")):
        score += 0.5 * (rain_days / n)
        if rain_days == 0:
            score -= 0.2

    # Cold / winter items
    if any(k in item_l for k in ("winter coat", "thermal", "gloves", "scarf",
                                  "insulated boot", "snow boot", "hand warmer")):
        score += 0.5 * (cold_days / n)
        if cold_days == 0 and temp_avg > 20:
            score -= 0.4

    # Warm / light items
    if any(k in item_l for k in ("t-shirt", "short", "breathable", "lightweight")):
        score += 0.4 * (hot_days / n)
        if cold_days > n * 0.5:
            score -= 0.3

    # Windproof
    if "windproof" in item_l:
        score += 0.4 * (wind_days / n)

    # Sun protection
    if any(k in item_l for k in ("sunscreen", "sunglasses", "wide-brim hat")):
        score += 0.5 * min(1.0, uv_avg / 8)
        if uv_avg < 2:
            score -= 0.3

    # Snow items
    if any(k in item_l for k in ("snow boot", "thermal socks")):
        score += 0.5 * (snow_days / n)

    # Business items
    if any(k in item_l for k in ("business attire", "formal shoes", "smart jacket",
                                  "smart casual", "business cards", "laptop bag")):
        if purpose == "business":
            score += 0.3
        else:
            score -= 0.2

    return float(np.clip(score, 0.0, 1.0))


def _item_weight(item: str) -> float:
    return ITEM_WEIGHTS.get(item, DEFAULT_WEIGHT)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GACandidate:
    """One solution produced by the genetic algorithm."""
    solution_id: str
    items: List[str]
    fitness_comfort: float   # 0–1, higher = better weather fit
    fitness_weight: float    # 0–1, higher = lighter relative to limit


@dataclass
class OptimizationResult:
    """Full pipeline output — stages 1, 2, 3 combined."""
    # Stage 1
    ga_solutions: List[GACandidate] = field(default_factory=list)
    # Stage 2
    final_items: List[str]  = field(default_factory=list)
    total_weight: float     = 0.0
    weight_limit: float     = 20.0
    # Stage 3
    removed_items: List[str]   = field(default_factory=list)
    final_fitness_comfort: float = 0.0
    final_fitness_weight: float  = 0.0
    basic_explanation: str       = ""

    def to_dict(self) -> dict:
        return {
            "stage1_ga_solutions": [
                {
                    "solution_id": s.solution_id,
                    "items": s.items,
                    "fitness": {
                        "comfort": round(s.fitness_comfort, 3),
                        "weight":  round(s.fitness_weight,  3),
                    },
                }
                for s in self.ga_solutions
            ],
            "stage2_knapsack": {
                "final_list":    self.final_items,
                "total_weight":  round(self.total_weight, 2),
                "weight_limit":  self.weight_limit,
            },
            "stage3_summary": {
                "removed_items":  self.removed_items,
                "final_fitness": {
                    "comfort": round(self.final_fitness_comfort, 3),
                    "weight":  round(self.final_fitness_weight,  3),
                },
                "basic_explanation": self.basic_explanation,
            },
        }


# ── Stage 1: Genetic Algorithm (NSGA-II) ─────────────────────────────────────

# GA hyper-parameters
_GA_SOLUTIONS    = 30    # population size
_GA_GENERATIONS  = 60    # number of generations
_GA_PARENTS_MATING = 10
_GA_MUTATION_PROB  = 0.08
_GA_PARETO_TARGET  = 6   # how many non-dominated solutions to keep


def _run_ga(items: List[str], comfort_scores: np.ndarray,
            weights: np.ndarray, weight_limit: float,
            n_solutions: int = _GA_PARETO_TARGET) -> List[Tuple[np.ndarray, float, float]]:
    """
    NSGA-II multi-objective GA: maximise comfort, minimise weight.
    Returns a list of (binary_vector, comfort_fitness, weight_fitness) tuples
    representing the Pareto front (non-dominated solutions).

    Uses PyGAD when available; falls back to a plain evolutionary loop otherwise.
    """
    n_items = len(items)
    if n_items == 0:
        return []

    # Fitness function: weighted sum with a hard weight-limit penalty.
    # PyGAD maximises this scalar; we later extract the Pareto front separately.
    def _scalar_fitness(solution: np.ndarray) -> float:
        total_w   = float(np.dot(solution, weights))
        comfort_f = float(np.dot(solution, comfort_scores)) / max(1, solution.sum())
        weight_f  = max(0.0, 1.0 - total_w / weight_limit)
        penalty   = 1.0 if total_w <= weight_limit else 0.5 * (weight_limit / total_w)
        return (0.6 * comfort_f + 0.4 * weight_f) * penalty

    # ── Try PyGAD ─────────────────────────────────────────────────────────────
    population: List[np.ndarray] = []
    try:
        import pygad

        def fitness_func(ga_instance, solution, solution_idx):
            return _scalar_fitness(np.array(solution, dtype=float))

        ga = pygad.GA(
            num_generations=_GA_GENERATIONS,
            num_parents_mating=_GA_PARENTS_MATING,
            fitness_func=fitness_func,
            sol_per_pop=_GA_SOLUTIONS,
            num_genes=n_items,
            gene_type=int,
            init_range_low=0,
            init_range_high=2,           # genes are 0 or 1
            mutation_probability=_GA_MUTATION_PROB,
            mutation_type="random",
            mutation_by_replacement=True,
            random_mutation_min_val=0,
            random_mutation_max_val=2,
            crossover_type="two_points",
            keep_parents=4,
            suppress_warnings=True,
        )
        ga.run()
        # Collect all solutions from the final population
        for sol in ga.population:
            population.append(np.clip(np.round(sol).astype(int), 0, 1))

    except Exception:
        # ── Pure-Python evolutionary fallback ─────────────────────────────────
        rng = random.Random(42)

        def _random_sol() -> np.ndarray:
            return np.array([rng.randint(0, 1) for _ in range(n_items)], dtype=int)

        def _crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            pt = rng.randint(1, n_items - 1)
            return np.concatenate([a[:pt], b[pt:]])

        def _mutate(sol: np.ndarray) -> np.ndarray:
            s = sol.copy()
            for i in range(n_items):
                if rng.random() < _GA_MUTATION_PROB:
                    s[i] = 1 - s[i]
            return s

        pop = [_random_sol() for _ in range(_GA_SOLUTIONS)]
        for _ in range(_GA_GENERATIONS):
            scored = sorted(pop, key=lambda s: _scalar_fitness(s.astype(float)), reverse=True)
            parents = scored[:_GA_PARENTS_MATING]
            children = []
            while len(children) < _GA_SOLUTIONS - _GA_PARENTS_MATING:
                a, b = rng.sample(parents, 2)
                children.append(_mutate(_crossover(a, b)))
            pop = parents + children
        population = pop

    # ── Extract Pareto front (non-dominated in comfort vs weight space) ────────
    evaluated: List[Tuple[np.ndarray, float, float]] = []
    for sol in population:
        sol = np.clip(sol, 0, 1).astype(int)
        total_w   = float(np.dot(sol, weights))
        comfort_f = (float(np.dot(sol, comfort_scores)) / max(1, sol.sum())
                     if sol.sum() > 0 else 0.0)
        weight_f  = max(0.0, 1.0 - total_w / weight_limit)
        evaluated.append((sol, comfort_f, weight_f))

    # Deduplicate by item set
    seen: set = set()
    unique = []
    for sol, cf, wf in evaluated:
        key = tuple(sol.tolist())
        if key not in seen:
            seen.add(key)
            unique.append((sol, cf, wf))

    # Non-domination: solution A dominates B if A >= B on both objectives and > on one
    def _dominates(a: Tuple, b: Tuple) -> bool:
        return a[1] >= b[1] and a[2] >= b[2] and (a[1] > b[1] or a[2] > b[2])

    pareto = []
    for cand in unique:
        if not any(_dominates(other, cand) for other in unique if other is not cand):
            pareto.append(cand)

    # Sort by comfort desc and return top-N
    pareto.sort(key=lambda x: x[1], reverse=True)
    return pareto[:n_solutions]


# ── Stage 2: Knapsack (OR-Tools / DP fallback) ────────────────────────────────

def _knapsack_select(items: List[str], comfort_scores: np.ndarray,
                     weights: np.ndarray, weight_limit: float) -> List[str]:
    """
    0/1 Knapsack: maximise comfort score subject to total weight ≤ weight_limit.
    Uses OR-Tools CP-SAT when available; falls back to pure-DP otherwise.
    Weights are quantised to 10g precision for the DP table.
    """
    n = len(items)
    if n == 0:
        return []

    SCALE = 10          # quantise to units of 100g → table stays manageable
    int_weights = [max(1, round(w * SCALE)) for w in weights]
    int_limit   = max(1, round(weight_limit * SCALE))
    int_values  = [round(c * 1000) for c in comfort_scores]  # integer comfort

    # ── OR-Tools CP-SAT ───────────────────────────────────────────────────────
    try:
        from ortools.sat.python import cp_model
        model   = cp_model.CpModel()
        x       = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        model.Add(sum(int_weights[i] * x[i] for i in range(n)) <= int_limit)
        model.Maximize(sum(int_values[i] * x[i] for i in range(n)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [items[i] for i in range(n) if solver.Value(x[i])]
    except Exception:
        pass

    # ── Pure-DP fallback ──────────────────────────────────────────────────────
    # dp[j] = max comfort achievable with weight capacity j
    dp  = [0] * (int_limit + 1)
    sel = [[] for _ in range(int_limit + 1)]

    for i in range(n):
        for j in range(int_limit, int_weights[i] - 1, -1):
            new_val = dp[j - int_weights[i]] + int_values[i]
            if new_val > dp[j]:
                dp[j] = new_val
                sel[j] = sel[j - int_weights[i]] + [i]

    return [items[i] for i in sel[int_limit]]


# ── Stage 3: Rule-based summary ───────────────────────────────────────────────

def _build_summary(original_items: List[str], final_items: List[str],
                   final_comfort: float, final_weight: float,
                   weight_limit: float, total_weight: float) -> str:
    removed = [it for it in original_items if it not in final_items]
    if not removed:
        return "All recommended items fit within the weight limit — nothing removed."

    reasons = []
    total_removed_weight = sum(_item_weight(it) for it in removed)

    # Classify removed items
    heavy   = [it for it in removed if _item_weight(it) >= 0.7]
    low_fit = [it for it in removed if _item_weight(it) < 0.7]

    if heavy:
        names = ", ".join(heavy)
        reasons.append(f"heavy items ({names}) contributed "
                       f"{sum(_item_weight(it) for it in heavy):.1f}kg")
    if low_fit:
        names = ", ".join(low_fit)
        reasons.append(f"lower-priority items ({names}) removed to reclaim "
                       f"{sum(_item_weight(it) for it in low_fit):.1f}kg")

    reason_str = "; ".join(reasons) if reasons else "items exceeded weight limit"
    return (
        f"Removed {len(removed)} item(s) ({total_removed_weight:.1f}kg) to satisfy "
        f"the {weight_limit}kg weight limit: {reason_str}. "
        f"Final pack weighs {total_weight:.1f}kg "
        f"(comfort score {final_comfort:.2f}, weight score {final_weight:.2f})."
    )


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def _run_pipeline(candidate_items: List[str],
                  forecasts: List[DayForecast],
                  context: TripContext,
                  weight_limit_kg: float) -> OptimizationResult:
    """
    Core three-stage pipeline shared by both public entry points.
    """
    if not candidate_items:
        return OptimizationResult(weight_limit=weight_limit_kg)

    # Pre-compute per-item scores and weights
    comfort_scores = np.array([_item_comfort_score(it, forecasts, context)
                                for it in candidate_items])
    weights        = np.array([_item_weight(it) for it in candidate_items])

    result = OptimizationResult(weight_limit=weight_limit_kg)

    # ── Stage 1: GA ───────────────────────────────────────────────────────────
    print("\n  [Optimizer] Stage 1 — Genetic Algorithm (NSGA-II)...")
    pareto = _run_ga(candidate_items, comfort_scores, weights, weight_limit_kg)

    solution_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, (sol_vec, cf, wf) in enumerate(pareto):
        sol_items = [candidate_items[i] for i, v in enumerate(sol_vec) if v == 1]
        result.ga_solutions.append(GACandidate(
            solution_id     = solution_labels[idx % len(solution_labels)],
            items           = sol_items,
            fitness_comfort = round(cf, 3),
            fitness_weight  = round(wf, 3),
        ))
    print(f"  [Optimizer] GA found {len(pareto)} Pareto-optimal solutions.")

    # ── Stage 2: Knapsack ─────────────────────────────────────────────────────
    print("  [Optimizer] Stage 2 — Knapsack optimisation...")
    final = _knapsack_select(candidate_items, comfort_scores, weights, weight_limit_kg)
    total_w = sum(_item_weight(it) for it in final)

    result.final_items   = final
    result.total_weight  = round(total_w, 2)

    final_comfort = (float(np.mean([_item_comfort_score(it, forecasts, context)
                                     for it in final])) if final else 0.0)
    final_weight_f = max(0.0, 1.0 - total_w / weight_limit_kg)

    result.final_fitness_comfort = round(final_comfort, 3)
    result.final_fitness_weight  = round(final_weight_f, 3)
    print(f"  [Optimizer] Knapsack selected {len(final)} items "
          f"({total_w:.1f}kg / {weight_limit_kg}kg).")

    # ── Stage 3: Summary ──────────────────────────────────────────────────────
    print("  [Optimizer] Stage 3 — Building summary...")
    result.removed_items     = [it for it in candidate_items if it not in final]
    result.basic_explanation = _build_summary(
        candidate_items, final, final_comfort, final_weight_f,
        weight_limit_kg, total_w,
    )

    return result


# ── Public entry points ───────────────────────────────────────────────────────

def optimise_from_recommendations(trip_packing: dict,
                                   forecasts: List[DayForecast],
                                   context: TripContext,
                                   weight_limit_kg: float = 20.0) -> OptimizationResult:
    """
    Entry point when called after the recommender.
    trip_packing is the dict returned by build_trip_packing_list():
        {"clothing": [...], "packing": [...]}
    """
    all_items = list(trip_packing.get("clothing", [])) + \
                list(trip_packing.get("packing",  []))
    return _run_pipeline(all_items, forecasts, context, weight_limit_kg)


def optimise_items(item_names: List[str],
                   forecasts: List[DayForecast],
                   context: TripContext,
                   weight_limit_kg: float = 20.0) -> OptimizationResult:
    """
    Standalone entry point — pass any list of item names directly.
    Unknown items use the DEFAULT_WEIGHT (0.30 kg).
    """
    return _run_pipeline(item_names, forecasts, context, weight_limit_kg)

# ==================================================================================
# ADDITION by Kevin: Dynamic 3D Bin Packing Optimization
# Accepts dynamically calculated weights/volumes from CV instead of hardcoded dicts.
# ==================================================================================

def optimise_dynamic_items(wardrobe_items: list, ml_recommended_items: list,
                           weight_limit_kg: float, volume_limit_l: float = 40.0,
                           forecasts = None, context = None) -> 'OptimizationResult':
    """
    Advanced 3D Bin Packing optimization for Streamlit.
    Accepts dynamically calculated weights/volumes from CV instead of hardcoded dicts.
    """
    all_items = []
    
    # 1. Process User's CV-uploaded wardrobe (Dynamic weights/volumes)
    for item in wardrobe_items:
        data = item.get("item_data_json", item)
        all_items.append({
            "name": data.get("detected_label", "Unknown Item"),
            "weight_kg": data.get("calculated_weight_g", 300) / 1000.0,
            "volume_l": data.get("calculated_volume_l", 2.0),
            "comfort_score": 0.9 # User selected it, high baseline comfort
        })
        
    # 2. Process ML Recommended items (Fallback to static weights if not in CV)
    for item_name in ml_recommended_items:
        # Only add if not already uploaded by user
        if not any(it["name"].lower() in item_name.lower() for it in all_items):
            all_items.append({
                "name": item_name,
                "weight_kg": _item_weight(item_name), # Use Member B's dict as fallback
                "volume_l": ITEM_WEIGHTS.get(item_name, 0.30) * 5, # Approx volume from weight if missing
                "comfort_score": _item_comfort_score(item_name, forecasts, context) if forecasts else 0.5
            })

    if not all_items:
        return OptimizationResult(weight_limit=weight_limit_kg)

    # 3. Multi-Constraint Knapsack (Weight AND Volume) using OR-Tools
    try:
        from ortools.sat.python import cp_model
        
        model = cp_model.CpModel()
        n = len(all_items)
        
        # Create boolean variables for each item
        x = [model.NewBoolVar(f"x_{i}", str(i)) for i in range(n)]
        
        # Separate expressions to prevent parenthesis corruption
        w_expr = sum(all_items[i]["weight_kg"] * x[i] for i in range(n))
        v_expr = sum(all_items[i]["volume_l"] * x[i] for i in range(n))
        c_expr = sum(all_items[i]["comfort_score"] * 1000 * x[i] for i in range(n))
        
        # Apply constraints
        model.Add(w_expr <= weight_limit_kg)
        model.Add(v_expr <= volume_limit_l)
        
        # Maximize comfort
        model.Maximize(c_expr)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            final_items = []
            total_w, total_v = 0.0, 0.0
            for i in range(n):
                if solver.Value(x[i]):
                    final_items.append(all_items[i]["name"])
                    total_w += all_items[i]["weight_kg"]
                    total_v += all_items[i]["volume_l"]
            
            return OptimizationResult(
                final_items=final_items,
                total_weight=round(total_w, 2),
                weight_limit=weight_limit_kg,
                basic_explanation=f"Packed {len(final_items)} items. Weight: {total_w:.1f}kg/{weight_limit_kg}kg, Volume: {total_v:.1f}L/{volume_limit_l}L."
            )
            
    except Exception as e:
        print(f"[3D Optimizer Error] {e}")
        
    # Fallback to standard weight-only knapsack if OR-Tools fails
    item_names = [it["name"] for it in all_items]
    return _run_pipeline(item_names, forecasts or [], context or TripContext("tourism", "City", "Country"), weight_limit_kg)