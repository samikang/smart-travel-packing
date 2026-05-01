"""
Packing Optimizer — Unified Optimization Pipeline with 3D Bin Packing
======================================================================
Architecture:
  Phase 1  Initial Assessment — Volume & weight calculation for all candidate items
  Phase 2  Spatial Feasibility — Try to fit everything in 3D bin packing
  Phase 3  GA Optimization — Genetic Algorithm finds best item combinations
           (always runs, even if everything fits — finds Pareto-optimal trade-offs)
  Phase 4  Redundancy Analysis — Detect and report duplicate/similar items
  Phase 5  Final 3D Packing — Verified spatial layout with exact positions
  Phase 6  Build Result — Compile all metrics, insights, and recommendations

Three optimization modes:
  "light"      - Remove only obvious redundancies (90% volume target)
  "balanced"   - Balance comfort vs space (80% volume target) [DEFAULT]
  "aggressive" - Keep only essentials (65% volume target)

Dependencies:
    pip install pygad>=3.3 ortools>=9.8 numpy
    (PyGAD optional — greedy fallback included)
    (OR-Tools optional — DP fallback included)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np

from models import DayForecast, TripContext
from recommender import calculate_needed_quantity
import re


# ── Item weight table (kg) ────────────────────────────────────────────────────
# Covers all 37 items in recommender.ALL_ITEMS.
# Weights are conservative estimates; edit freely.

ITEM_WEIGHTS: Dict[str, float] = {
    "Heavy winter coat": 1.50, "Thermal underlayer": 0.25,
    "Warm sweater or fleece": 0.60, "Gloves and scarf": 0.20,
    "Insulated boots": 1.20, "Light jacket or fleece": 0.45,
    "Long-sleeve shirt": 0.25, "Jeans or trousers": 0.60,
    "T-shirt or short sleeves": 0.20, "Lightweight breathable clothing": 0.20,
    "Shorts or light trousers": 0.25, "Waterproof jacket": 0.55,
    "Windproof jacket": 0.40, "Waterproof snow boots": 1.30,
    "Thermal socks": 0.10, "Business attire": 0.80,
    "Formal shoes": 0.90, "Comfortable walking shoes": 0.70,
    "Casual wear": 0.35, "Smart casual outfit": 0.55,
    "Smart jacket": 0.70, "Compact umbrella": 0.30,
    "Waterproof bag cover": 0.15, "Sunscreen SPF 50+": 0.20,
    "Sunscreen SPF 30+": 0.18, "Sunglasses": 0.08,
    "Wide-brim hat": 0.18, "Hand warmers": 0.10,
    "Laptop bag": 0.80, "Power adapter": 0.25,
    "Business cards": 0.05, "Day backpack": 0.60,
    "Phone charger / power bank": 0.30, "City map or offline maps": 0.05,
    "Reusable water bottle": 0.25, "Small gift (optional)": 0.30,
    "Phone charger": 0.15, "Sleepwear / pyjamas": 0.35,
}
DEFAULT_WEIGHT = 0.30

ITEM_DIMS_CM: Dict[str, Tuple[float, float, float]] = {
    "Heavy winter coat": (60, 45, 8), "Thermal underlayer": (35, 25, 3),
    "Warm sweater or fleece": (40, 35, 5), "Gloves and scarf": (30, 15, 3),
    "Insulated boots": (32, 20, 15), "Light jacket or fleece": (45, 35, 5),
    "Long-sleeve shirt": (38, 28, 2), "Jeans or trousers": (40, 30, 4),
    "T-shirt or short sleeves": (35, 25, 2), "Lightweight breathable clothing": (38, 28, 2),
    "Shorts or light trousers": (35, 25, 2), "Waterproof jacket": (50, 40, 6),
    "Windproof jacket": (48, 38, 5), "Waterproof snow boots": (34, 22, 16),
    "Thermal socks": (20, 10, 3), "Business attire": (55, 40, 5),
    "Formal shoes": (32, 18, 12), "Comfortable walking shoes": (30, 18, 12),
    "Casual wear": (38, 28, 3), "Smart casual outfit": (45, 35, 4),
    "Smart jacket": (55, 40, 5), "Compact umbrella": (30, 5, 5),
    "Waterproof bag cover": (25, 20, 2), "Sunscreen SPF 50+": (15, 5, 5),
    "Sunscreen SPF 30+": (15, 5, 5), "Sunglasses": (16, 7, 4),
    "Wide-brim hat": (40, 38, 8), "Hand warmers": (10, 7, 2),
    "Laptop bag": (40, 30, 10), "Power adapter": (10, 8, 6),
    "Business cards": (10, 6, 1), "Day backpack": (45, 30, 15),
    "Phone charger / power bank": (15, 10, 3), "City map or offline maps": (20, 12, 1),
    "Reusable water bottle": (25, 8, 8), "Small gift (optional)": (20, 15, 10),
    "Phone charger": (12, 8, 3), "Sleepwear / pyjamas": (30, 22, 3),
}
DEFAULT_DIMS = (20, 15, 5)

def _expand_to_quantities(
    wardrobe_items: List[dict], suitability_scores: Dict[str, float],
    trip_days: int, forecasts: List[DayForecast], purpose: str,
) -> List[dict]:
    """Duplicate wardrobe items into individual packing candidates.

    Each detected item is replicated according to the quantity needed for
    the trip (see ``_calculate_needed_quantity``).  Each copy keeps the
    original weight, volume and dimensions so the GA / knapsack can
    decide to drop individual copies.

    Args:
        wardrobe_items: Detected garments from ``image_recognition.py``.
        suitability_scores: Mapping from item name to 0‑1 comfort score.
        trip_days: Trip duration in days.
        forecasts: Weather predictions (used by the quantity helper).
        purpose: Trip purpose.

    Returns:
        A list of dicts, each representing one physical copy with keys
        ``name``, ``weight_kg``, ``volume_l``, ``dims_cm``, ``suitability``,
        and ``instance`` (1‑based copy number).
    """
    expanded = []
    item_counts = {}
    
    for item in wardrobe_items:
        name = item.get("detected_label") or item.get("name", "Unknown")
        score = suitability_scores.get(name, 0.5)
        if score < 0.3: continue
        
        weight_g = item.get("calculated_weight_g") or item.get("weight_g", DEFAULT_WEIGHT * 1000)
        volume_l = item.get("calculated_volume_l") or item.get("volume_l", 2.0)
        dims_cm = item.get("dims_cm", None)
        
        needed = calculate_needed_quantity(name, trip_days, purpose)
        current_count = item_counts.get(name, 0)
        
        for i in range(min(needed, 10)):
            expanded.append({
                "name": name, "weight_kg": weight_g / 1000.0,
                "volume_l": volume_l, "dims_cm": dims_cm,
                "suitability": score, "instance": current_count + i + 1,
            })
        item_counts[name] = current_count + needed
    
    return expanded


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GACandidate:
    """A single solution produced by the Genetic Algorithm.

    Attributes:
        solution_id: Short label (A, B, C, …) for display.
        items: Names of the items included in this candidate.
        fitness_comfort: Normalised comfort score (0‑1, higher is better).
        fitness_weight: Normalised weight score (0‑1, higher means lighter).
    """
    solution_id: str
    items: List[str]
    fitness_comfort: float
    fitness_weight: float


@dataclass
class PackedItem:
    """An item placed inside the 3D suitcase layout.

    Attributes:
        name: Item name.
        x: Position from left edge (cm).
        y: Position from front edge (cm).
        z: Position from bottom (cm).
        l: Length along x‑axis (cm).
        w: Width along y‑axis (cm).
        h: Height along z‑axis (cm).
    """
    name: str; x: float; y: float; z: float; l: float; w: float; h: float


@dataclass
class BinPackResult:
    """Output of the 3D bin packing stage.

    Attributes:
        bin_dims: Suitcase inner dimensions (L, W, H) in cm.
        packed: List of items that were successfully placed.
        unpacked: Names of items that could not be placed.
        utilisation_pct: Percentage of bin volume used.
    """
    bin_dims: Tuple[float, float, float]
    packed: List[PackedItem] = field(default_factory=list)
    unpacked: List[str] = field(default_factory=list)
    utilisation_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert the bin‑packing result to a JSON‑serialisable dictionary."""
        return {
            "bin_dims_cm": {"L": self.bin_dims[0], "W": self.bin_dims[1], "H": self.bin_dims[2]},
            "utilisation_pct": round(self.utilisation_pct, 1),
            "packed_count": len(self.packed), "unpacked_count": len(self.unpacked),
            "packed": [{"name": p.name, "position_cm": {"x": p.x, "y": p.y, "z": p.z},
                         "dims_cm": {"l": p.l, "w": p.w, "h": p.h}} for p in self.packed],
            "unpacked": self.unpacked,
        }


@dataclass
class OptimizationModeConfig:
    """Hyper‑parameters that control optimisation aggressiveness.

    Attributes:
        mode: One of ``"light"``, ``"balanced"``, ``"aggressive"``.
        target_volume_utilization: Fraction of suitcase volume to aim for (0‑1).
        target_weight_utilization: Fraction of weight limit to aim for (0‑1).
        comfort_threshold: Minimum comfort score required to keep an item.
        redundancy_penalty: Whether to penalise similar items in the GA fitness.
    """
    mode: str
    target_volume_utilization: float
    target_weight_utilization: float
    comfort_threshold: float
    redundancy_penalty: bool


@dataclass
class OptimizationResult:
    """Complete result of the three‑stage optimization pipeline.

    Attributes:
        ga_solutions: Pareto‑front solutions from Stage 1 (GA).
        final_items: Knapsack‑selected packing list (Stage 2).
        total_weight: Total weight of the selected items (kg).
        weight_limit: User‑specified weight limit (kg).
        removed_items: Items that were excluded by the knapsack.
        final_fitness_comfort: Average comfort score of the final list.
        final_fitness_weight: Weight fitness score of the final list.
        basic_explanation: Human‑readable summary of the optimization.
        original_volume: Total volume of all candidate items before optimization.
        original_weight: Total weight of all candidate items before optimization.
        kept_but_questionable: Items that fit but could be reconsidered.
        removal_reasons: Mapping from removed item name → list of reasons.
        optimization_insights: Bullet‑point insights for the user.
        volume_utilization: Percentage of suitcase volume used by final items.
        weight_utilization: Percentage of weight limit used.
        comfort_score: Overall comfort score of the final selection.
        redundancy_score: 1.0 = no redundancy, lower = more redundant.
        bin_pack_layout: 3D positions of packed items (if ``run_bin_pack``).
        space_left_for_souvenirs: Empty volume available for purchases (litres).
        suggested_swaps: Item‑swap recommendations.
        multi_bag_suggestion: Carry‑on vs. check‑in split (if requested).
        quantity_summary: Count per item type in the final list.
        missing_items: Items needed but not owned.
    """
    ga_solutions: List[GACandidate] = field(default_factory=list)
    final_items: List[str] = field(default_factory=list)
    total_weight: float = 0.0; weight_limit: float = 20.0
    removed_items: List[str] = field(default_factory=list)
    final_fitness_comfort: float = 0.0; final_fitness_weight: float = 0.0
    basic_explanation: str = ""
    original_volume: float = 0.0; original_weight: float = 0.0
    kept_but_questionable: List[Dict] = field(default_factory=list)
    removal_reasons: Dict[str, List[str]] = field(default_factory=dict)
    optimization_insights: List[str] = field(default_factory=list)
    volume_utilization: float = 0.0; weight_utilization: float = 0.0
    comfort_score: float = 0.0; redundancy_score: float = 0.0
    bin_pack_layout: Optional[Dict] = None
    space_left_for_souvenirs: float = 0.0
    suggested_swaps: List[Dict] = field(default_factory=list)
    multi_bag_suggestion: Optional[Dict] = None
    quantity_summary: Dict[str, int] = field(default_factory=dict)
    missing_items: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Export the full optimization result as a JSON‑compatible dict."""
        d = {
            "stage1_ga_solutions": [{
                "solution_id": s.solution_id, "items": s.items,
                "fitness": {"comfort": round(s.fitness_comfort, 3), "weight": round(s.fitness_weight, 3)},
            } for s in self.ga_solutions],
            "stage2_knapsack": {"final_list": self.final_items, "total_weight": round(self.total_weight, 2), "weight_limit": self.weight_limit},
            "stage3_summary": {"removed_items": self.removed_items, "final_fitness": {"comfort": round(self.final_fitness_comfort, 3), "weight": round(self.final_fitness_weight, 3)}, "basic_explanation": self.basic_explanation},
            "stage4_optimization": {
                "original_volume_l": round(self.original_volume, 1), "original_weight_kg": round(self.original_weight, 1),
                "volume_utilization_pct": round(self.volume_utilization, 1), "weight_utilization_pct": round(self.weight_utilization, 1),
                "comfort_score": round(self.comfort_score, 3), "redundancy_score": round(self.redundancy_score, 3),
                "space_left_for_souvenirs_l": round(self.space_left_for_souvenirs, 1),
                "removal_reasons": self.removal_reasons, "kept_but_questionable": self.kept_but_questionable,
                "insights": self.optimization_insights, "suggested_swaps": self.suggested_swaps,
                "quantity_summary": self.quantity_summary, "missing_items": self.missing_items,
            },
        }
        if self.bin_pack_layout: d["stage4_bin_pack_3d"] = self.bin_pack_layout
        if self.multi_bag_suggestion: d["stage5_multi_bag"] = self.multi_bag_suggestion
        return d


# ═══════════════════════════════════════════════════════════════════════════
# 3D BIN PACKING
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_dims(name, cv_dims, teammate_dims):
    """Resolve an item's dimensions with priority: teammate > CV > static table > default.

    Args:
        name: Item name string.
        cv_dims: Dimensions from the CV pipeline (``(L, W, H)`` tuple or ``None``).
        teammate_dims: External override dictionary (item name → ``(L, W, H)``).

    Returns:
        A ``(L, W, H)`` tuple in cm.
    """
    if teammate_dims and name in teammate_dims: return teammate_dims[name]
    if cv_dims is not None: return cv_dims
    return ITEM_DIMS_CM.get(name, DEFAULT_DIMS)


def bin_pack_3d(
    items: List[dict], bin_l: float = 70.0, bin_w: float = 45.0,
    bin_h: float = 30.0, teammate_dims: Optional[Dict] = None,
) -> BinPackResult:
    """Pack items into a 3D suitcase using a layer‑by‑layer guillotine heuristic.

    Items are sorted largest‑first.  Each item is placed in the current row;
    when the row is full a new row is started; when the layer is full a new
    layer is stacked on top.  Two rotations (swap L↔W) are tried per item.

    Args:
        items: List of dicts with keys ``"name"`` and optional ``"dims_cm"``.
        bin_l: Suitcase inner length (cm).
        bin_w: Suitcase inner width (cm).
        bin_h: Suitcase inner height (cm).
        teammate_dims: Optional external dimension overrides.

    Returns:
        A ``BinPackResult`` with packed / unpacked items and utilisation percentage.
    """
    result = BinPackResult(bin_dims=(bin_l, bin_w, bin_h))
    
    resolved = []
    for it in items:
        name = it["name"]; cv_dims = it.get("dims_cm")
        l, w, h = _resolve_dims(name, cv_dims, teammate_dims)
        best_rot = (w, l, h) if w > l else (l, w, h)
        resolved.append({"name": name, "l": best_rot[0], "w": best_rot[1], "h": best_rot[2]})
    
    resolved.sort(key=lambda x: x["l"] * x["w"] * x["h"], reverse=True)
    
    cur_z = cur_layer_h = cur_x = cur_y = cur_row_h = 0.0
    
    for it in resolved:
        l, w, h = it["l"], it["w"], it["h"]
        if l > bin_l or w > bin_w or h > bin_h:
            result.unpacked.append(it["name"]); continue
        
        placed = False
        for _ in range(2):
            if cur_x + l <= bin_l and cur_y + w <= bin_w and cur_z + h <= bin_h:
                result.packed.append(PackedItem(name=it["name"], x=round(cur_x,1), y=round(cur_y,1), z=round(cur_z,1), l=l, w=w, h=h))
                cur_row_h = max(cur_row_h, w); cur_layer_h = max(cur_layer_h, h)
                cur_x += l; placed = True; break
            else:
                new_y = cur_y + cur_row_h
                if new_y + w <= bin_w:
                    cur_x = cur_y = cur_row_h = 0.0; cur_y = new_y
                else:
                    new_z = cur_z + cur_layer_h
                    if new_z + h <= bin_h:
                        cur_z = new_z; cur_layer_h = cur_x = cur_y = cur_row_h = 0.0
                    else:
                        result.unpacked.append(it["name"]); placed = True; break
        if not placed: result.unpacked.append(it["name"])
    
    packed_vol = sum(p.l * p.w * p.h for p in result.packed)
    bin_vol = bin_l * bin_w * bin_h
    result.utilisation_pct = round(packed_vol / bin_vol * 100, 1) if bin_vol > 0 else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# COMFORT SCORING
# ═══════════════════════════════════════════════════════════════════════════

def _item_comfort_score(item: str, forecasts: List[DayForecast], context: TripContext) -> float:
    """Calculate a 0–1 weather‑suitability score for a clothing item.

    Starts at a neutral 0.5 and adds / subtracts based on:
      - Rain → waterproof items gain; if no rain, they lose a little.
      - Cold → warm items gain; if warm, they lose.
      - Hot → light items gain; if cold, they lose.
      - Wind, UV, snow, and trip purpose also influence the score.

    Args:
        item: Recommender item name.
        forecasts: Daily weather predictions for the whole trip.
        context: Trip metadata (purpose, city, country).

    Returns:
        A float in [0.0, 1.0].
    """
    item_l = item.lower(); n = len(forecasts) or 1
    temp_avg = sum(f.temp_min + f.temp_max for f in forecasts) / (2 * n)
    rain_days = sum(1 for f in forecasts if f.precipitation_mm > 1)
    wind_days = sum(1 for f in forecasts if f.wind_speed_max > 40)
    uv_avg = sum(f.uv_index_max for f in forecasts) / n
    snow_days = sum(1 for f in forecasts if f.weather_code in {71,73,75,77,85,86})
    hot_days = sum(1 for f in forecasts if f.temp_max > 28)
    cold_days = sum(1 for f in forecasts if f.temp_min < 10)
    purpose = (context.purpose or "").lower()
    score = 0.5
    
    if any(k in item_l for k in ("umbrella", "waterproof", "raincoat")):
        score += 0.5 * (rain_days / n)
        if rain_days == 0: score -= 0.2
    if any(k in item_l for k in ("winter coat", "thermal", "gloves", "scarf", "insulated boot", "snow boot", "hand warmer")):
        score += 0.5 * (cold_days / n)
        if cold_days == 0 and temp_avg > 20: score -= 0.4
    if any(k in item_l for k in ("t-shirt", "short", "breathable", "lightweight")):
        score += 0.4 * (hot_days / n)
        if cold_days > n * 0.5: score -= 0.3
    if "windproof" in item_l: score += 0.4 * (wind_days / n)
    if any(k in item_l for k in ("sunscreen", "sunglasses", "wide-brim hat")):
        score += 0.5 * min(1.0, uv_avg / 8)
        if uv_avg < 2: score -= 0.3
    if any(k in item_l for k in ("snow boot", "thermal socks")): score += 0.5 * (snow_days / n)
    if any(k in item_l for k in ("business attire", "formal shoes", "smart jacket", "smart casual", "business cards", "laptop bag")):
        score += 0.3 if purpose == "business" else -0.2
    return float(np.clip(score, 0.0, 1.0))


def _item_weight(item: str) -> float:
    """Look up the static weight (kg) for an item.

    Args:
        item: Item name.

    Returns:
        Weight in kilograms; falls back to ``DEFAULT_WEIGHT`` if unknown.
    """
    return ITEM_WEIGHTS.get(item, DEFAULT_WEIGHT)


def _item_volume(item_name: str, teammate_dims: Dict = None) -> float:
    """Estimate the packed volume of an item in litres.

    Args:
        item_name: Item name.
        teammate_dims: Optional external dimension overrides.

    Returns:
        Volume in litres (cm³ / 1000), defaulting to 2.0 L if dimensions are unknown.
    """
    dims = _resolve_dims(item_name, None, teammate_dims)
    if dims: l, w, h = dims; return (l * w * h) / 1000
    return 2.0


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _find_similar_items(item, all_items):
    """Find items that belong to the same clothing category.

    Categories include warm layers, rain protection, footwear, sun
    protection, tops, and bottoms.

    Args:
        item: The reference item name.
        all_items: Full list of item names to search within.

    Returns:
        List of item names (including the reference item) that are similar.
    """
    categories = {
        'warm_layers': ['coat', 'jacket', 'fleece', 'sweater', 'thermal'],
        'rain_protection': ['waterproof', 'umbrella', 'rain'],
        'footwear': ['boots', 'shoes', 'formal', 'walking'],
        'sun_protection': ['sunscreen', 'sunglasses', 'hat', 'sun'],
        'tops': ['shirt', 't-shirt', 'long-sleeve'],
        'bottoms': ['jeans', 'trousers', 'shorts'],
    }
    item_lower = item.lower(); similar = [item]
    for cat, keywords in categories.items():
        if any(kw in item_lower for kw in keywords):
            similar.extend([i for i in all_items if i != item and any(kw in i.lower() for kw in keywords)])
    return list(set(similar))


def _analyze_redundancy(selected_items, removed_items, all_items):
    """Detect redundant items in the final selection.

    An item is considered redundant when more than two items from the same
    category (warm layers, rain gear, footwear, sun protection, formal wear)
    are present.

    Args:
        selected_items: Items kept by the knapsack.
        removed_items: Items dropped by the knapsack (for reference).
        all_items: Complete candidate list.

    Returns:
        A dict with keys ``findings`` (list of human‑readable strings),
        ``redundant_count`` (int), and ``score`` (1.0 = no redundancy).
    """
    redundancy_groups = {
        'warm layers': ['coat', 'jacket', 'fleece', 'sweater'],
        'rain gear': ['waterproof', 'umbrella'], 'footwear': ['boots', 'shoes'],
        'sun protection': ['sunscreen', 'sunglasses', 'hat'],
        'formal wear': ['business', 'formal', 'smart'],
    }
    findings = []; redundancy_count = 0
    for group_name, keywords in redundancy_groups.items():
        selected_in_group = [item for item in selected_items if any(kw in item.lower() for kw in keywords)]
        if len(selected_in_group) > 2:
            findings.append(f"Multiple {group_name}: {', '.join(selected_in_group[:3])}")
            redundancy_count += len(selected_in_group) - 2
    return {'findings': findings, 'redundant_count': redundancy_count,
            'score': 1.0 - min(1.0, redundancy_count / max(1, len(selected_items)))}


def _suggest_multi_bag_split(all_items, selected_items, removed_items, luggage_volume, teammate_dims):
    """Suggest a carry‑on vs. check‑in split.

    Heavy (>2 kg) or bulky (>15 L) items go to check‑in.  Remaining items
    fill the carry‑on up to 70 % of the luggage volume.  Any leftovers
    are assigned to check‑in.

    Args:
        all_items: Full candidate list.
        selected_items: Items chosen by the knapsack.
        removed_items: Items that were excluded (up to 3 may be recovered for check‑in).
        luggage_volume: Total suitcase volume (litres).
        teammate_dims: Optional dimension overrides.

    Returns:
        A dict with ``carry_on``, ``check_in``, and their respective volumes.
    """
    carry_on_volume = luggage_volume * 0.7; check_in_volume = luggage_volume * 1.5
    carry_on, check_in = [], []; current_vol = 0
    for item in selected_items:
        vol = _item_volume(item, teammate_dims); weight = _item_weight(item)
        if weight > 2.0 or vol > 15: check_in.append(item)
        elif current_vol + vol <= carry_on_volume: carry_on.append(item); current_vol += vol
    for item in selected_items:
        if item not in carry_on and item not in check_in: check_in.append(item)
    for item in removed_items[:3]:
        vol = _item_volume(item, teammate_dims)
        check_in_vol = sum(_item_volume(i, teammate_dims) for i in check_in)
        if check_in_vol + vol <= check_in_volume: check_in.append(item)
    return {
        'carry_on': carry_on, 'check_in': check_in,
        'carry_on_volume': round(sum(_item_volume(i, teammate_dims) for i in carry_on), 1),
        'check_in_volume': round(sum(_item_volume(i, teammate_dims) for i in check_in), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# KNAPSACK OPTIMIZATION — Exact solution under weight & volume constraints
# ═══════════════════════════════════════════════════════════════════════════

def _knapsack_select(
    items: List[str],
    comfort_scores: np.ndarray,
    weights: np.ndarray,
    weight_limit: float,
    volumes: np.ndarray = None,
    volume_limit: float = None,
    max_per_item: Dict[str, int] = None, 
    ) -> List[str]:
    """Select the optimal subset of items maximising comfort under weight & volume constraints.

    Uses OR‑Tools CP‑SAT solver; falls back to 2D dynamic programming if OR‑Tools
    is not installed. All input values are scaled to integers for the solver.

    Args:
        items: Candidate item names. Each element is a single physical copy
            (quantities are pre‑expanded).
        comfort_scores: Suitability values (0‑1) for each item.
        weights: Item weights in kilograms.
        weight_limit: Maximum allowed total weight (kg).
        volumes: Item volumes in litres (optional). When provided together with
            ``volume_limit``, the total packed volume must not exceed this limit.
        volume_limit: Maximum allowed total volume in litres (optional).
            Derived from luggage dimensions (L×W×H / 1000).
        max_per_item: Optional mapping from item name to maximum allowed copies.

    Returns:
        List of item names that form the optimal packing selection.
    """
    n = len(items)
    if n == 0:
        return []

    W_SCALE = 10
    V_SCALE = 1
    C_SCALE = 1000

    int_weights = [max(1, int(round(w * W_SCALE))) for w in weights]
    int_values  = [int(round(c * C_SCALE)) for c in comfort_scores]
    w_limit     = max(1, int(round(weight_limit * W_SCALE)))

    use_volume = volumes is not None and volume_limit is not None
    if use_volume:
        int_volumes = [max(1, int(round(v * V_SCALE))) for v in volumes]
        v_limit     = max(1, int(round(volume_limit * V_SCALE)))

    try:
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        model.tocpu()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        model.Add(sum(int_weights[i] * x[i] for i in range(n)) <= w_limit)
        if use_volume:
            model.Add(sum(int_volumes[i] * x[i] for i in range(n)) <= v_limit)

        if max_per_item:
            # Group copies by item name
            item_indices: Dict[str, List[int]] = {}
            for i, item in enumerate(items):
                item_indices.setdefault(item, []).append(i)
            for item_name, indices in item_indices.items():
                limit = max_per_item.get(item_name, 1)
                model.Add(sum(x[i] for i in indices) <= min(limit, len(indices)))

        model.Maximize(sum(int_values[i] * x[i] for i in range(n)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [items[i] for i in range(n) if solver.Value(x[i])]
    except Exception:
        pass

    # Fallback to DP (no per-item limit)
    if use_volume:
        dp = [[0] * (v_limit + 1) for _ in range(w_limit + 1)]
        keep = [[[] for _ in range(v_limit + 1)] for _ in range(w_limit + 1)]
        for i in range(n):
            wi, vi, val = int_weights[i], int_volumes[i], int_values[i]
            for w in range(w_limit, wi - 1, -1):
                for v in range(v_limit, vi - 1, -1):
                    new_val = dp[w - wi][v - vi] + val
                    if new_val > dp[w][v]:
                        dp[w][v] = new_val
                        keep[w][v] = keep[w - wi][v - vi] + [i]
        return [items[i] for i in keep[w_limit][v_limit]]

    dp = [0] * (w_limit + 1)
    keep = [[] for _ in range(w_limit + 1)]
    for i in range(n):
        wi, val = int_weights[i], int_values[i]
        for w in range(w_limit, wi - 1, -1):
            new_val = dp[w - wi] + val
            if new_val > dp[w]:
                dp[w] = new_val
                keep[w] = keep[w - wi] + [i]
    return [items[i] for i in keep[w_limit]]


# ═══════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def _run_unified_pipeline_with_candidates(
    candidates: List[dict],
    max_per_item: Dict[str, int],
    forecasts: List[DayForecast],
    context: TripContext,
    weight_limit_kg: float = 20.0,
    volume_limit_l: float = 40.0,
    run_bin_pack: bool = True,
    bin_dims: Tuple[float, float, float] = (70, 45, 30),
    teammate_dims: Optional[Dict] = None,
    optimization_mode: str = "balanced",
    reserve_space_percent: float = 15.0,
    multi_bag: bool = False,
) -> OptimizationResult:
    """Run the full GA (NSGA‑II) + Knapsack + 3D bin packing pipeline on a list of per‑photo candidates.

    Args:
        candidates: List of dicts, each with keys 'id', 'recommender_item', 'weight_kg',
            'volume_l', 'dims_cm', 'source'.
        max_per_item: Mapping from item name to maximum allowed copies.
        forecasts: List of DayForecast for the trip.
        context: TripContext (purpose, city, country).
        weight_limit_kg: Max allowed total weight (kg).
        volume_limit_l: Not directly used; luggage volume is derived from bin_dims.
        run_bin_pack: If True, run the 3D bin packing stage.
        bin_dims: (L, W, H) inner dimensions of the suitcase in cm.
        teammate_dims: Optional external dimensions overrides for items.
        optimization_mode: 'light', 'balanced', or 'aggressive'.
        reserve_space_percent: Percentage of luggage volume to reserve for souvenirs.
        multi_bag: If True, produce a carry‑on vs. check‑in split suggestion.

    Returns:
        An OptimizationResult containing the Pareto front, final selection, metrics,
        packing layout, and insights.
    """
    if not candidates:
        return OptimizationResult(weight_limit=weight_limit_kg)

    # ── Helper: per‑item max constraint check ─────────────────────────
    def _violates_max(item_indices):
        item_counts = {}
        for i in item_indices:
            name = item_names[i]
            item_counts[name] = item_counts.get(name, 0) + 1
        for name, cnt in item_counts.items():
            if cnt > max_per_item.get(name, 1):
                return True
        return False

    # ── Helper: 2‑objective non‑dominated sorting (maximisation) ──────
    def _pareto_front_indices(obj1, obj2):
        n = len(obj1)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if obj1[j] >= obj1[i] and obj2[j] >= obj2[i] and (obj1[j] > obj1[i] or obj2[j] > obj2[i]):
                    dominated[i] = True
                    break
        return [i for i in range(n) if not dominated[i]]

    mode_configs = {
        "light": OptimizationModeConfig("light", 0.90, 0.85, 0.3, False),
        "balanced": OptimizationModeConfig("balanced", 0.80, 0.75, 0.5, True),
        "aggressive": OptimizationModeConfig("aggressive", 0.65, 0.60, 0.7, True),
    }
    config = mode_configs.get(optimization_mode, mode_configs["balanced"])
    luggage_volume = (bin_dims[0] * bin_dims[1] * bin_dims[2]) / 1000

    n = len(candidates)
    candidate_ids = [c["id"] for c in candidates]
    item_names = [c["recommender_item"] for c in candidates]

    comfort_arr = np.array([_item_comfort_score(name, forecasts, context) for name in item_names])
    weight_arr = np.array([c["weight_kg"] for c in candidates])
    volume_arr = np.array([c["volume_l"] for c in candidates])

    initial_volume = sum(volume_arr)
    initial_weight = sum(weight_arr)
    print(f"  Candidates: {n} (volume: {initial_volume:.1f}L, weight: {initial_weight:.1f}kg)")

    # ── NSGA‑II / Multi‑objective GA ──────────────────────────────────
    print(f"\n🎯 OPTIMIZATION ({optimization_mode.upper()})")
    target_volume = luggage_volume * config.target_volume_utilization * (1 - reserve_space_percent / 100)
    ga_solutions_list = []

    try:
        import pygad

        INVALID = -1000.0

        def comfort_obj(ga, sol, idx):
            sel = [i for i, v in enumerate(sol) if v == 1]
            if not sel: return INVALID
            if sum(volume_arr[i] for i in sel) > luggage_volume: return INVALID
            if sum(weight_arr[i] for i in sel) > weight_limit_kg: return INVALID
            if _violates_max(sel): return INVALID
            return float(np.mean([comfort_arr[i] for i in sel]))

        def weight_obj(ga, sol, idx):
            sel = [i for i, v in enumerate(sol) if v == 1]
            if not sel: return INVALID
            if sum(volume_arr[i] for i in sel) > luggage_volume: return INVALID
            if sum(weight_arr[i] for i in sel) > weight_limit_kg: return INVALID
            if _violates_max(sel): return INVALID
            return float(1.0 - sum(weight_arr[i] for i in sel) / weight_limit_kg)

        def multi_fitness(ga, sol, idx):
            return (comfort_obj(ga, sol, idx), weight_obj(ga, sol, idx))

        ga = pygad.GA(
            num_generations=50,
            num_parents_mating=10,
            fitness_func=multi_fitness,
            sol_per_pop=40,
            num_genes=n,
            gene_type=int,
            init_range_low=0,
            init_range_high=2,
            mutation_percent_genes=15,
            mutation_num_genes=2,
            crossover_type="uniform",
            parent_selection_type="nsga2",
            keep_parents=2,
            suppress_warnings=True,
        )
        ga.run()

        final_fitness = np.array(ga.last_generation_fitness)
        final_fitness[final_fitness == INVALID] = -np.inf
        pareto_idx = _pareto_front_indices(final_fitness[:, 0], final_fitness[:, 1])

        # Deduplicate by unique item set to avoid showing the same solution many times
        seen_sets = set()
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ga_solutions_list = []
        # Sort Pareto solutions by comfort descending (or any order) for consistent display
        sorted_pareto = sorted(pareto_idx,
                               key=lambda i: (final_fitness[i, 0], final_fitness[i, 1]),
                               reverse=True)
        for pop_idx in sorted_pareto:
            sol = ga.population[pop_idx].astype(int)
            sol_items = sorted([candidate_ids[j] for j, v in enumerate(sol) if v == 1])
            sol_key = tuple(sol_items)
            if sol_key in seen_sets:
                continue
            seen_sets.add(sol_key)
            c_val = float(final_fitness[pop_idx, 0]) if final_fitness[pop_idx, 0] != -np.inf else 0.0
            w_val = float(final_fitness[pop_idx, 1]) if final_fitness[pop_idx, 1] != -np.inf else 0.0
            ga_solutions_list.append(GACandidate(
                solution_id=labels[len(ga_solutions_list) % len(labels)],
                items=sol_items,
                fitness_comfort=round(c_val, 3),
                fitness_weight=round(w_val, 3),
            ))
            if len(ga_solutions_list) >= 10:   # show at most 10 distinct plans
                break

        print(f"  NSGA‑II found {len(ga_solutions_list)} unique Pareto‑optimal solutions"
              f" (from {len(pareto_idx)} total Pareto individuals)")

    except ImportError:
        print("  ℹ️  PyGAD not installed — using greedy fallback")
        order = sorted(range(n), key=lambda i: -comfort_arr[i] / (weight_arr[i] + 0.01))
        selected = np.zeros(n, dtype=int)
        item_counts = {}
        total_vol = total_w = 0.0
        for i in order:
            name = item_names[i]; allowed = max_per_item.get(name, 1)
            if item_counts.get(name, 0) >= allowed: continue
            if total_vol + volume_arr[i] <= target_volume and total_w + weight_arr[i] <= weight_limit_kg:
                selected[i] = 1
                total_vol += volume_arr[i]; total_w += weight_arr[i]
                item_counts[name] = item_counts.get(name, 0) + 1
        sol_items = [candidate_ids[i] for i, v in enumerate(selected) if v == 1]
        if sol_items:
            c_f = float(np.mean([comfort_arr[i] for i in range(n) if selected[i]]))
            w_f = max(0.0, 1.0 - total_w / weight_limit_kg)
            ga_solutions_list = [GACandidate(solution_id="A", items=sol_items, fitness_comfort=round(c_f,3), fitness_weight=round(w_f,3))]

    # ── Knapsack stage (optimal) ───────────────────────────────────────
    print(f"\n🎒 KNAPSACK OPTIMIZATION")
    selected_ids = _knapsack_select(
        items=candidate_ids,
        comfort_scores=comfort_arr,
        weights=weight_arr,
        weight_limit=weight_limit_kg,
        volumes=volume_arr,
        volume_limit=luggage_volume,
        max_per_item=max_per_item,
    )
    selected_names = [item_names[candidate_ids.index(cid)] for cid in selected_ids]

    removed_ids = [cid for cid in candidate_ids if cid not in selected_ids]

    redundancy_analysis = _analyze_redundancy(selected_names, removed_ids, selected_names)

    final_pack = None
    if run_bin_pack and selected_ids:
        sel_indices = [candidate_ids.index(cid) for cid in selected_ids]
        items_for_bp = [{"name": candidate_ids[i], "dims_cm": candidates[i].get("dims_cm")} for i in sel_indices]
        final_pack = bin_pack_3d(items_for_bp, bin_l=bin_dims[0], bin_w=bin_dims[1], bin_h=bin_dims[2], teammate_dims=teammate_dims)
        print(f"  ✅ Final pack: {len(final_pack.packed)} items, {final_pack.utilisation_pct}% utilized")

    final_volume = sum(volume_arr[candidate_ids.index(cid)] for cid in selected_ids)
    final_weight = sum(weight_arr[candidate_ids.index(cid)] for cid in selected_ids)
    souvenir_space = luggage_volume * (1 - reserve_space_percent / 100) - final_volume

    quantity_summary = {}
    for name in selected_names:
        quantity_summary[name] = quantity_summary.get(name, 0) + 1

    unique_items = len(set(selected_names))
    total_copies = len(selected_names)

    insights = []
    if initial_volume > final_volume:
        insights.append(f"📦 Freed {initial_volume - final_volume:.1f}L of space")
    if souvenir_space > 0:
        insights.append(f"🛍️  {souvenir_space:.1f}L available for souvenirs")
    if redundancy_analysis['findings']:
        insights.append(f"🔄 Reduced redundancy: {', '.join(redundancy_analysis['findings'])}")
    insights.append(f"⚖️  {optimization_mode.capitalize()} optimization applied")

    removal_reasons = {cid: ["Optimization trade-off"] for cid in removed_ids}

    result = OptimizationResult(
        ga_solutions=ga_solutions_list,
        final_items=selected_ids,
        total_weight=round(final_weight, 2),
        weight_limit=weight_limit_kg,
        removed_items=removed_ids,
        final_fitness_comfort=round(np.mean([comfort_arr[candidate_ids.index(cid)] for cid in selected_ids]), 3) if selected_ids else 0,
        final_fitness_weight=round(max(0, 1 - final_weight / weight_limit_kg), 3),
        basic_explanation=(f"Optimized ({optimization_mode}): {unique_items} unique items ({total_copies} total), {final_weight:.1f}kg"),
        original_volume=round(initial_volume, 1), original_weight=round(initial_weight, 1),
        removal_reasons=removal_reasons, optimization_insights=insights,
        volume_utilization=round(final_volume / luggage_volume * 100, 1) if luggage_volume > 0 else 0,
        weight_utilization=round(final_weight / weight_limit_kg * 100, 1),
        comfort_score=round(np.mean([comfort_arr[candidate_ids.index(cid)] for cid in selected_ids]), 3) if selected_ids else 0,
        redundancy_score=round(redundancy_analysis['score'], 3),
        bin_pack_layout=final_pack.to_dict() if final_pack else None,
        space_left_for_souvenirs=round(souvenir_space, 1),
        multi_bag_suggestion=None,
        quantity_summary=quantity_summary,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

def optimise_from_recommendations(  
    trip_packing: dict, forecasts: List[DayForecast], context: TripContext,
    weight_limit_kg: float = 20.0, volume_limit_l: float = 40.0,
    run_bin_pack: bool = True, bin_dims: Tuple[float, float, float] = (70, 45, 30),
    teammate_dims: Optional[Dict] = None, optimization_mode: str = "balanced",
    reserve_space_percent: float = 15.0, multi_bag: bool = False,
) -> OptimizationResult:
    """Optimize the recommender's master packing list.

    Extracts all unique clothing and packing items from the trip packing
    dictionary and runs the full pipeline on them.

    Args:
        trip_packing: Dict with keys ``"clothing"`` and ``"packing"``, each a
            list of item name strings.
        forecasts: Daily weather predictions.
        context: Trip metadata.
        weight_limit_kg: Max total weight (kg).
        volume_limit_l: Nominal volume limit (litres).
        run_bin_pack: Whether to perform 3D bin packing.
        bin_dims: Suitcase inner dimensions (L, W, H) in cm.
        teammate_dims: Optional dimension overrides.
        optimization_mode: ``"light"``, ``"balanced"``, or ``"aggressive"``.
        reserve_space_percent: Suitcase space to reserve for souvenirs.
        multi_bag: If True, also generate carry‑on vs. check‑in suggestion.

    Returns:
        An ``OptimizationResult`` with the final packing selection and all metrics.
    """
    all_items = list(trip_packing.get("clothing", [])) + list(trip_packing.get("packing", []))
    return _run_unified_pipeline(
        candidate_items=all_items, forecasts=forecasts, context=context,
        weight_limit_kg=weight_limit_kg, volume_limit_l=volume_limit_l,
        run_bin_pack=run_bin_pack, bin_dims=bin_dims, teammate_dims=teammate_dims,
        optimization_mode=optimization_mode, reserve_space_percent=reserve_space_percent, multi_bag=multi_bag)

def optimise_items(
    item_names: List[str], forecasts: List[DayForecast], context: TripContext,
    weight_limit_kg: float = 20.0, volume_limit_l: float = 40.0,
    run_bin_pack: bool = True, bin_dims: Tuple[float, float, float] = (70, 45, 30),
    teammate_dims: Optional[Dict] = None, optimization_mode: str = "balanced",
    reserve_space_percent: float = 15.0, multi_bag: bool = False,
) -> OptimizationResult:
    """Run the optimization pipeline on an arbitrary list of item names.

    This entry point bypasses the recommender entirely and is useful for
    testing or when an item list is already known.

    Args:
        item_names: List of item name strings to consider for packing.
        forecasts: Daily weather predictions.
        context: Trip metadata.
        weight_limit_kg: Max total weight (kg).
        volume_limit_l: Nominal volume limit (litres).
        run_bin_pack: Whether to perform 3D bin packing.
        bin_dims: Suitcase inner dimensions (L, W, H) in cm.
        teammate_dims: Optional dimension overrides.
        optimization_mode: ``"light"``, ``"balanced"``, or ``"aggressive"``.
        reserve_space_percent: Suitcase space to reserve for souvenirs.
        multi_bag: If True, also generate carry‑on vs. check‑in suggestion.

    Returns:
        An ``OptimizationResult`` with the final packing selection and all metrics.
    """
    return _run_unified_pipeline(
        candidate_items=item_names, forecasts=forecasts, context=context,
        weight_limit_kg=weight_limit_kg, volume_limit_l=volume_limit_l,
        run_bin_pack=run_bin_pack, bin_dims=bin_dims, teammate_dims=teammate_dims,
        optimization_mode=optimization_mode, reserve_space_percent=reserve_space_percent, multi_bag=multi_bag)


def optimise_dynamic_items(
    wardrobe_items: list,
    ml_recommended_items: list,
    weight_limit_kg: float,
    volume_limit_l: float = 40.0,
    forecasts: List[DayForecast] = None,
    context: TripContext = None,
    suitability_scores: Dict[str, float] = None,
    run_bin_pack: bool = True,
    bin_dims: Tuple[float, float, float] = (70, 45, 30),
    teammate_dims: Optional[Dict] = None,
    optimization_mode: str = "balanced",
    reserve_space_percent: float = 15.0,
    multi_bag: bool = False,
    trip_days: int = 1,
) -> OptimizationResult:
    """Full pipeline entry point with per‑photo candidates.

    Converts raw wardrobe items (from image recognition) and missing items
    (recommended but not in photos) into a candidate list, then calls
    the unified pipeline.

    Args:
        wardrobe_items: List of per‑photo dicts (image_name, detected_label,
            weight_g, volume_l, etc.).
        ml_recommended_items: Additional item names recommended by the ML
            model (used for purchase items).
        weight_limit_kg: Max total weight (kg).
        volume_limit_l: Not directly used; luggage volume is derived from bin_dims.
        forecasts: Optional list of DayForecast.
        context: Optional TripContext.
        suitability_scores: Optional dict mapping item name → comfort score.
        run_bin_pack: Whether to perform 3D bin packing.
        bin_dims: Suitcase inner dimensions (L, W, H) in cm.
        teammate_dims: Optional external dimension overrides.
        optimization_mode: ``"light"``, ``"balanced"``, or ``"aggressive"``.
        reserve_space_percent: Suitcase space to reserve for souvenirs.
        multi_bag: If True, also generate carry‑on vs. check‑in suggestion.
        trip_days: Trip duration in days.

    Returns:
        An ``OptimizationResult`` with the final packing selection and all metrics.
    """
    _forecasts = forecasts or []
    _context = context or TripContext("tourism", "City", "Country")
    _scores = suitability_scores or {}

    # ---------- Build candidate list ----------
    candidates = []
    max_per_item = {}   # recommender_item -> max allowed copies

    # 1. All wardrobe photos become candidates
    for photo in wardrobe_items:
        label = photo.get("detected_label", "Unknown")
        if re.search(r"person|no detection|unknown|car|dog|cat", label, re.I):
            continue
        # Map to standard recommender item (use same matching logic)
        recommender_item = _map_to_recommender(label)   # new helper (see below)
        if not recommender_item:
            recommender_item = label   # fallback, use raw label

        candidates.append({
            "id":               photo.get("image_name", f"photo_{len(candidates)}"),
            "recommender_item": recommender_item,
            "weight_kg":        photo.get("weight_g", DEFAULT_WEIGHT * 1000) / 1000.0,
            "volume_l":         photo.get("volume_l", 2.0),
            "dims_cm":          photo.get("dims_cm", None),
            "source":           "wardrobe",
        })
        # Track the expected maximum for this item type (will be filled later)
        max_per_item.setdefault(recommender_item, 0)
        # We'll adjust after we know the needed quantities

    # 2. Determine needed quantities for all recommended items
    needed_quantities = {}
    for item_name in set([c["recommender_item"] for c in candidates] + ml_recommended_items):
        score = _scores.get(item_name, 0.5)
        if score >= 0.3:   # only consider items with enough suitability
            needed = calculate_needed_quantity(item_name, trip_days, _context.purpose)
            needed_quantities[item_name] = needed

    # Update max_per_item based on needed quantities
    for item_name, needed in needed_quantities.items():
        max_per_item[item_name] = max(max_per_item.get(item_name, 0), needed)

    # 3. Add purchase candidates for missing items
    for item_name, needed in needed_quantities.items():
        already_have = sum(1 for c in candidates if c["recommender_item"] == item_name)
        missing = max(0, needed - already_have)
        for i in range(missing):
            candidates.append({
                "id":               f"{item_name}_purchase_{i+1}",
                "recommender_item": item_name,
                "weight_kg":        ITEM_WEIGHTS.get(item_name, DEFAULT_WEIGHT),
                "volume_l":         _item_volume(item_name),      # static dimensions
                "dims_cm":          ITEM_DIMS_CM.get(item_name, DEFAULT_DIMS),
                "source":           "purchase",
            })

    # Ensure max_per_item includes purchase items (they are already counted, but just in case)
    for c in candidates:
        if c["recommender_item"] not in max_per_item:
            max_per_item[c["recommender_item"]] = 1   # at least one allowed

    if not candidates:
        return OptimizationResult(weight_limit=weight_limit_kg)

    return _run_unified_pipeline_with_candidates(
        candidates=candidates,
        max_per_item=max_per_item,
        forecasts=_forecasts,
        context=_context,
        weight_limit_kg=weight_limit_kg,
        volume_limit_l=volume_limit_l,
        run_bin_pack=run_bin_pack,
        bin_dims=bin_dims,
        teammate_dims=teammate_dims,
        optimization_mode=optimization_mode,
        reserve_space_percent=reserve_space_percent,
        multi_bag=multi_bag,
    )


def _map_to_recommender(detected_label: str) -> Optional[str]:
    """Map a YOLO/CLIP label to a standard recommender item name.

    Args:
        detected_label: The raw label from the computer vision backend.

    Returns:
        The name of the recommender item if a match is found, otherwise None.
    """
    # Simple keyword matching; adjust as needed
    mapping = {
        'short_sleeved_shirt': 'T-shirt or short sleeves',
        'long_sleeved_shirt': 'Long-sleeve shirt',
        'trousers': 'Jeans or trousers',
        'shorts': 'Shorts or light trousers',
        'jacket': 'Light jacket or fleece',
        'coat': 'Heavy winter coat',
        'dress': 'Casual wear',
        'skirt': 'Casual wear',
        'boot': 'Insulated boots',
        'shoe': 'Comfortable walking shoes',
        'umbrella': 'Compact umbrella',
        'backpack': 'Day backpack',
        'sunscreen': 'Sunscreen SPF 30+',
        'sunglasses': 'Sunglasses',
        'hat': 'Wide-brim hat',
        'gloves': 'Gloves and scarf',
        'scarf': 'Gloves and scarf',
        'pajama': 'Sleepwear / pyjamas',
        'sleepwear': 'Sleepwear / pyjamas',
        'sweater': 'Warm sweater or fleece',
        'sweatshirt': 'Warm sweater or fleece',
        'hoodie': 'Warm sweater or fleece',
        'fleece': 'Warm sweater or fleece',
        'blazer': 'Smart jacket',
        'formal': 'Business attire',
        'business': 'Business attire',
        'suit': 'Business attire',
        'waterproof': 'Waterproof jacket',
        'windproof': 'Windproof jacket',
        'thermal': 'Thermal underlayer',
    }
    detected_lower = detected_label.lower().replace('_', ' ')
    for key, item in mapping.items():
        if key in detected_lower:
            return item
    # If no match, try to find any keyword overlap with ALL_ITEMS
    from recommender import ALL_ITEMS
    for item in ALL_ITEMS:
        for word in item.lower().split():
            if word in detected_lower:
                return item
    return None

def _violates_max_per_item(selected_indices, item_names, max_per_item):
    """Return True if any item type exceeds its max allowed copies.

    Args:
        selected_indices: Indices of selected items.
        item_names: List of item names corresponding to all candidates.
        max_per_item: Mapping from item name to maximum allowed count.

    Returns:
        True if any item type's selected count exceeds its allowed maximum.
    """
    item_counts = {}
    for i in selected_indices:
        name = item_names[i]
        item_counts[name] = item_counts.get(name, 0) + 1
    for name, cnt in item_counts.items():
        allowed = max_per_item.get(name, cnt)
        if cnt > allowed:
            return True
    return False


def _non_dominated_sorting_2d(obj1, obj2):
    """Return indices of non‑dominated solutions for maximisation problems.

    Args:
        obj1: Array of first objective values.
        obj2: Array of second objective values.

    Returns:
        List of indices where the corresponding solution is not dominated
        by any other solution.
    """
    n = len(obj1)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            # i dominated by j if j >= i in all objectives and > in at least one
            if obj1[j] >= obj1[i] and obj2[j] >= obj2[i] and (obj1[j] > obj1[i] or obj2[j] > obj2[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]