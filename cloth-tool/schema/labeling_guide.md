# Cloth Attribute Labeling Guide

**Schema version:** 1.0  
**Task:** Assign 5 attributes to each garment crop image.  
**Each image = exactly one garment.**

---

## Workflow Overview

1. Open a crop image.
2. Fill in all 5 attributes in order: `cloth_type` → `season_group` → `material_group` → `fold_state` → `weight_class`.
3. Save as one JSON record (see format at the bottom).
4. If you are genuinely unsure about an attribute, set `confidence: 0` and leave a `notes` comment. Do **not** guess.

---

## Attribute 1 — `cloth_type`

**What it is:** The garment category. Base this on shape and construction, not color or pattern.

| Label | Pick when you see… | Do NOT confuse with |
|---|---|---|
| `t_shirt` | Short sleeve, knit fabric, no buttons/collar | `shirt` (woven, buttoned), `sweater` (long sleeve) |
| `shirt` | Woven fabric, collar, button placket (long or short sleeve) | `t_shirt` (knit, no buttons) |
| `sweater` | Long sleeve knit top, no full zip | `vest` (no sleeves), `jacket` (structured shell) |
| `jacket` | Outer layer, roughly hip-length, structured/woven shell | `coat` (longer), `down_jacket` (quilted fill) |
| `coat` | Outer layer reaching **at or below the knee** | `jacket` (shorter) |
| `down_jacket` | Quilted or puffed outer layer, visible baffle stitching or puffy surface | `jacket` (no fill), `vest` (sleeveless) |
| `pants` | Lower body garment with two separate leg tubes | `skirt` (no leg separation) |
| `skirt` | Lower body garment, no leg separation | `dress` (has a top attached), `pants` |
| `dress` | One-piece: top and skirt are the same garment | `skirt` (no top), `shirt` (no skirt) |
| `vest` | Sleeveless — used as outer layer or mid-layer | `t_shirt` (has sleeves), `down_jacket` (has sleeves) |

**Edge cases:**
- A quilted sleeveless puffer → `vest` (cloth_type) + `padded_down_like` (material).
- A long cardigan (knee-length, knit) → `coat` over `sweater` if it is clearly outerwear length.
- A zip-up sweatshirt with a hood → label as `jacket` (structured) or `sweater` (soft knit) based on shell construction.

---

## Attribute 2 — `season_group`

**What it is:** The primary season this garment is designed for. Think about insulation and breathability, not color.

| Label | Indicators |
|---|---|
| `summer` | Very thin or sheer fabric, short sleeves or sleeveless, mesh or open weave, linen, light cotton |
| `spring_autumn` | Medium thickness, layering pieces, light sweaters, unlined jackets, long-sleeve shirts |
| `winter` | Heavy insulation (down fill, thick wool, fleece lining), coats, heavy sweaters, padded jackets |

**Decision rule:**
> Ask: *"Would I sweat in this on a 25 °C (77 °F) day?"* → `summer`.  
> *"Would I be cold in this below 5 °C (41 °F)?"* → `winter`.  
> Anything in between → `spring_autumn`.

**Edge cases:**
- A light denim jacket → `spring_autumn` (not summer, not insulated enough for winter).
- A thin long-sleeve shirt → `spring_autumn` (transitional), not `summer`.
- A fleece sweatshirt → `winter` if thick fleece, `spring_autumn` if thin mid-layer fleece.

---

## Attribute 3 — `material_group`

**What it is:** The dominant fabric type based on **visual texture and construction**, not the care label. When mixed, pick whichever material covers >50% of the surface.

| Label | Visual cues |
|---|---|
| `cotton_like` | Smooth flat weave, matte surface, no visible stretch, wrinkles easily (cotton, linen, chambray, oxford cloth) |
| `knit` | Visible knit loops or ribs, stretchy, soft surface (jersey, rib-knit, waffle-knit, sweatshirt fleece) |
| `denim` | Diagonal twill weave, indigo or colored, stiff and heavy (jeans denim, denim jacket fabric) |
| `wool_like` | Textured surface, fuzzy or hairy, warm-looking (wool, cashmere, boiled wool, polar fleece, sherpa) |
| `padded_down_like` | Quilted surface with visible stitched channels or puffy baffles (down jacket shell, synthetic fill jacket) |
| `leather_like` | Smooth or slightly grainy surface with sheen, no woven texture visible (genuine leather, PU leather, suede) |
| `synthetic_sportswear` | Slick or matte nylon/polyester, may have mesh panels, performance fabric (windbreaker shell, athletic wear) |
| `mixed_unknown` | Clearly two distinct fabrics with no dominant one, or image quality too low to judge |

**Edge cases:**
- A sweatshirt → `knit` (sweatshirt fabric is knit-constructed, even if it feels thick).
- A fleece-lined denim jacket → `denim` (outer shell dominates visually).
- A waterproof hardshell with mesh lining → `synthetic_sportswear`.
- A cotton/polyester blend t-shirt → `knit` if it looks like jersey knit, `cotton_like` if it looks woven and flat.

---

## Attribute 4 — `fold_state`

**What it is:** The physical state of the garment **in this specific photo**. This records how it was stored or photographed, not how it is normally stored.

| Label | What you see |
|---|---|
| `unfolded` | Laid completely flat, hung on hanger, or spread out — full garment shape visible |
| `folded` | Neatly folded into a compact rectangle or square; edges aligned; garment shape not visible |
| `partially_folded` | One sleeve tucked in, loosely folded in half, or rolled but not compressed — partial shape still visible |
| `compressed` | Vacuum-sealed bag, tightly rolled into a tiny bundle, or space-bag compressed — significantly reduced volume |

**Decision tree:**

```
Can you see the full silhouette of the garment?
├── Yes → unfolded
└── No
    ├── Is it a tidy rectangle/square?
    │   ├── Yes → folded
    │   └── No, still some shape → partially_folded
    └── Is the volume dramatically reduced (vacuum/tight roll)?
        └── Yes → compressed
```

---

## Attribute 5 — `weight_class`

**What it is:** A visual estimate of fabric weight and thickness. You are **not weighing** the item — judge from how it looks: drape, stiffness, thickness at edges, and garment type context.

| Label | GSM range | Visual cues |
|---|---|---|
| `light` | < ~200 gsm | Drapes softly, edges thin, see-through or semi-sheer possible, thin t-shirts, summer dresses, silk-like |
| `medium` | ~200–400 gsm | Standard everyday weight, normal t-shirts, jeans, casual shirts, light sweaters |
| `heavy` | > ~400 gsm | Stiff or thick edges, stands up on its own, heavy coats, thick denim, padded jackets, heavy wool |

**Shortcut by cloth_type:**

| cloth_type | Likely weight_class |
|---|---|
| `t_shirt` | light or medium |
| `shirt` | light or medium |
| `sweater` | medium or heavy |
| `jacket` | medium or heavy |
| `coat` | heavy |
| `down_jacket` | heavy |
| `pants` (denim) | medium or heavy |
| `pants` (other) | light or medium |
| `skirt` | light or medium |
| `dress` | light or medium |
| `vest` | light → heavy depending on fill |

> These are defaults — override based on what you actually see.

---

## Attribute 6 — `pressed_size_class`

**What it is:** The storage footprint of the garment when **normally folded** — how much space it takes up in a drawer or suitcase. This is NOT the same as weight, and NOT about how it is folded in the photo.

> Always imagine the garment folded in the **standard way for that type** (e.g. shirt folded flat, coat folded in thirds). Do NOT imagine it vacuum-compressed unless that is literally what you see.

| Label | What it looks like | Typical examples |
|---|---|---|
| `pocket_size` | Stuffs into its own chest pocket or a fist-sized pouch | Packable down jacket, ultralight windbreaker, thin synthetic tee |
| `compact` | Small flat stack; fits easily in a corner of a bag | Thin t-shirt, leggings, light summer dress, tank top |
| `standard` | Normal folded pile — the expected size for that garment type | Regular sweater, chinos, casual button shirt, light jeans |
| `bulky` | Large thick stack; hard to squeeze further without compression | Heavy wool coat, thick denim jacket, heavy puffer (uncompressed), chunky knit sweater |

**Why this is different from `weight_class`:**

| Garment | `weight_class` | `pressed_size_class` | Reason |
|---|---|---|---|
| Packable down jacket | `heavy` | `pocket_size` | Fill is heavy but compresses dramatically |
| Thick wool coat | `heavy` | `bulky` | Heavy AND stays large |
| Thin linen shirt | `light` | `compact` | Light and folds small |
| Heavy denim jacket | `heavy` | `bulky` | Dense fabric, does not compress |
| Lightweight synthetic puffer | `medium` | `compact` | Synthetic fill compresses more than cotton |

**Edge cases:**
- A down jacket that is NOT packable (no stuffsack, thick baffles) → `standard` or `bulky`, not `pocket_size`.
- A heavy wool sweater → `bulky` even though it is not outerwear.
- A dress with a lot of tulle/layers → `bulky` even if the fabric is `light`.

---

## Attribute 7 — `folded_size_class`

**What it is:** The **flat 2D footprint** (length × width) when the garment is normally folded — no pressing, no vacuum. Think: how much shelf or drawer area does it occupy when stacked?

> This is distinct from `pressed_size_class` (which is about thickness/3D bulk). A heavy coat and a thin blanket can both be `l` in footprint but differ completely in `pressed_size_class`.

| Label | Approx. size | Real-world reference | Typical examples |
|---|---|---|---|
| `xs` | ≤ 15 × 20 cm | Smaller than a paperback book | Packable jacket stuffed, very small tee |
| `s` | ~20 × 30 cm | Paperback book | Thin t-shirt, leggings, light summer dress |
| `m` | ~30 × 40 cm | A4 sheet of paper | Standard shirt, regular sweater, chinos, jeans |
| `l` | ~40 × 50 cm+ | Larger than A4 | Heavy coat, thick outerwear, wide-cut pants |

**How to judge from a photo:**
- If the garment is shown **folded**, estimate the footprint directly from the visible rectangle.
- If the garment is **unfolded**, mentally fold it to its standard form:
  - Shirt → fold in thirds lengthwise, then fold in half → ~m
  - Coat → fold in half lengthwise, then thirds → ~l
  - Leggings → fold in thirds → ~s

**Compared to `pressed_size_class`:**

| Garment | `folded_size_class` | `pressed_size_class` | Why different |
|---|---|---|---|
| Heavy wool coat | `l` | `bulky` | Large footprint AND thick stack |
| Regular t-shirt | `s` | `compact` | Small footprint AND thin stack |
| Down jacket (non-packable) | `m` | `standard` | Medium footprint, medium stack |
| Packable down jacket | `xs` | `pocket_size` | Tiny footprint when stuffed |
| Wide-leg trousers | `l` | `standard` | Large footprint but not especially thick |

---

## Confidence Score

Use the `confidence` field to flag uncertain labels.

| Value | Meaning |
|---|---|
| `1` | Clear, no doubt |
| `0` | Uncertain — write reason in `notes` |

Only label as `confidence: 0` when genuinely unsure. Do not use it as a hedge on every label.

---

## Output Format

Each labeled crop produces one JSON record:

```json
{
  "image_id": "img_0042",
  "crop_file": "crops/img_0042_crop_01.jpg",
  "cloth_type": "down_jacket",
  "season_group": "winter",
  "material_group": "padded_down_like",
  "fold_state": "folded",
  "weight_class": "heavy",
  "pressed_size_class": "standard",
  "folded_size_class": "m",
  "annotator": "your_name",
  "confidence": 1,
  "notes": ""
}
```

If flagging uncertainty:

```json
{
  "image_id": "img_0107",
  "crop_file": "crops/img_0107_crop_01.jpg",
  "cloth_type": "jacket",
  "season_group": "spring_autumn",
  "material_group": "mixed_unknown",
  "fold_state": "folded",
  "weight_class": "medium",
  "pressed_size_class": "standard",
  "folded_size_class": "m",
  "annotator": "your_name",
  "confidence": 0,
  "notes": "Outer shell looks synthetic but lining is visible fleece — unclear which dominates"
}
```

---

## Common Mistakes to Avoid

- **Do not label by brand or care tag** — label only what you see visually.
- **Do not use `mixed_unknown` as a default** — only use it when you genuinely cannot decide after 10 seconds.
- **`fold_state` is about this photo**, not how the item is normally stored.
- **`season_group` is about insulation**, not sleeve length alone (a long-sleeve linen shirt is still `spring_autumn`).
- **A hooded sweatshirt** — label as `sweater` if soft knit construction, `jacket` if it has a structured woven shell.
- **A down vest is `vest`**, not `down_jacket` — cloth_type captures shape, material_group captures fill.
