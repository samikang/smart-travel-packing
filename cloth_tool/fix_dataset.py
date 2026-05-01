"""
Fix schema mismatches in dataset CSV and flag rows needing manual review.

Auto-fixes applied:
  cloth_type        : denim_skirt→skirt, denim_jacket→jacket, denim_pants→pants,
                      leather_jacket→jacket, leather_skirt→skirt, shorts→pants,
                      hoodie→jacket (if notes contain 'zip'), hoodie→sweater (otherwise)
  season_group      : all_season→spring_autumn
  material_group    : woven→cotton_like, fleece→wool_like
  folded_size_class : large→l, medium→m, small→s, xl→l
  pressed_size_class: low→compact, medium→standard, high→bulky

Flagged for manual review:
  cloth_type = denim_overall  (no clear schema mapping)
  cloth_type = hoodie         (auto-remapped, but confirm zip→jacket or plain→sweater)
  pressed_size_class          (all remapped values need visual confirmation)
  material_group = woven      (auto-remapped to cotton_like — confirm per item)
"""

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

VALID = {
    "cloth_type":        {"t_shirt", "shirt", "sweater", "jacket", "coat",
                          "down_jacket", "pants", "skirt", "dress", "vest"},
    "season_group":      {"summer", "spring_autumn", "winter"},
    "material_group":    {"cotton_like", "knit", "denim", "wool_like", "padded_down_like",
                          "leather_like", "synthetic_sportswear", "mixed_unknown"},
    "fold_state":        {"unfolded", "folded", "partially_folded", "compressed"},
    "weight_class":      {"light", "medium", "heavy"},
    "folded_size_class": {"xs", "s", "m", "l"},
    "pressed_size_class":{"pocket_size", "compact", "standard", "bulky"},
}

# ---------------------------------------------------------------------------
# Remap tables
# ---------------------------------------------------------------------------

CLOTH_TYPE_REMAP = {
    "denim_skirt":    ("skirt",   "auto"),
    "denim_jacket":   ("jacket",  "auto"),
    "denim_pants":    ("pants",   "auto"),
    "leather_jacket": ("jacket",  "auto"),
    "leather_skirt":  ("skirt",   "auto"),
    "shorts":         ("pants",   "auto"),
    # hoodie handled separately
}

SEASON_REMAP = {
    "all_season": "spring_autumn",
}

MATERIAL_REMAP = {
    "woven":  ("cotton_like", "confirm"),
    "fleece": ("wool_like",   "auto"),
}

FOLDED_SIZE_REMAP = {
    "large":  "l",
    "medium": "m",
    "small":  "s",
    "xl":     "l",
}

PRESSED_SIZE_REMAP = {
    "low":    "compact",
    "medium": "standard",
    "high":   "bulky",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fix(input_path: Path, output_path: Path, flagged_path: Path) -> None:
    rows_fixed = []
    rows_flagged = []

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = [f for f in (reader.fieldnames or []) if f != "fix_needed"]

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            issues = []
            notes = row.get("notes", "").lower()

            # --- cloth_type ---
            ct = row.get("cloth_type", "").strip()
            if ct in CLOTH_TYPE_REMAP:
                new_ct, mode = CLOTH_TYPE_REMAP[ct]
                row["cloth_type"] = new_ct
                if mode == "confirm":
                    issues.append(f"cloth_type '{ct}'→'{new_ct}' — confirm")
            elif ct == "hoodie":
                new_ct = "jacket" if "zip" in notes else "sweater"
                row["cloth_type"] = new_ct
                issues.append(f"cloth_type 'hoodie'→'{new_ct}' — confirm (zip→jacket, plain→sweater)")
            elif ct == "denim_overall":
                issues.append("cloth_type='denim_overall' has no schema mapping — set manually")
            elif ct not in VALID["cloth_type"]:
                issues.append(f"cloth_type='{ct}' unknown — set manually")

            # --- season_group ---
            sg = row.get("season_group", "").strip()
            if sg in SEASON_REMAP:
                row["season_group"] = SEASON_REMAP[sg]
            elif sg not in VALID["season_group"]:
                issues.append(f"season_group='{sg}' unknown — set manually")

            # --- material_group ---
            mg = row.get("material_group", "").strip()
            if mg in MATERIAL_REMAP:
                new_mg, mode = MATERIAL_REMAP[mg]
                row["material_group"] = new_mg
                if mode == "confirm":
                    issues.append(f"material_group '{mg}'→'{new_mg}' — confirm")
            elif mg not in VALID["material_group"]:
                issues.append(f"material_group='{mg}' unknown — set manually")

            # --- folded_size_class ---
            fsc = row.get("folded_size_class", "").strip()
            if fsc in FOLDED_SIZE_REMAP:
                row["folded_size_class"] = FOLDED_SIZE_REMAP[fsc]
            elif fsc not in VALID["folded_size_class"]:
                issues.append(f"folded_size_class='{fsc}' unknown")

            # --- pressed_size_class ---
            psc = row.get("pressed_size_class", "").strip()
            if psc in PRESSED_SIZE_REMAP:
                remapped = PRESSED_SIZE_REMAP[psc]
                row["pressed_size_class"] = remapped
                issues.append(f"pressed_size_class '{psc}'→'{remapped}' — confirm")
            elif psc not in VALID["pressed_size_class"]:
                issues.append(f"pressed_size_class='{psc}' unknown — set manually")

            # --- remaining fields ---
            for field in ("fold_state", "weight_class"):
                val = row.get(field, "").strip()
                if val not in VALID[field]:
                    issues.append(f"{field}='{val}' unknown")

            row["fix_needed"] = "; ".join(issues)
            rows_fixed.append(row)
            if issues:
                rows_flagged.append(row)

    out_fields = list(original_fields) + ["fix_needed"]

    def write_csv(path: Path, rows: list) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_path, rows_fixed)
    write_csv(flagged_path, rows_flagged)

    total   = len(rows_fixed)
    flagged = len(rows_flagged)
    print(f"Total rows   : {total}")
    print(f"Clean rows   : {total - flagged}")
    print(f"Flagged rows : {flagged}  → {flagged_path.name}")
    print(f"Output saved : {output_path.name}")

    from collections import Counter
    counter: Counter = Counter()
    for r in rows_flagged:
        for part in r["fix_needed"].split(";"):
            part = part.strip()
            if part:
                key = part.split("'")[0].strip().split(" ")[0]
                counter[key] += 1
    print("\nFlag summary (rows needing manual confirmation):")
    for k, v in counter.most_common():
        print(f"  {k}: {v} row(s)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    inp  = base / "dataset1_fixed.csv"
    out  = base / "dataset1_clean.csv"
    flag = base / "dataset1_review.csv"

    if not inp.exists():
        print(f"ERROR: {inp} not found", file=sys.stderr)
        sys.exit(1)

    fix(inp, out, flag)
