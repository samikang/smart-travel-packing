## Model comparison used in `image_recognition.py`

This repo has two separate “model families”:

- **Vision backends**: detect a garment label from an image (`--vision` mode).
- **Weight/volume estimators**: estimate packed weight (g) and volume (L) from the image + label.

---

### Vision backends (garment label detection)

| Option (in code) | What it is | Runs where | Needs internet? | Needs key? | Main output | Pros | Cons | Best usage |
|---|---|---|---|---|---|---|---|---|
| **YOLO (`vision="yolo"`)** | Local YOLO11 *classification* (`yolo11n-cls.pt`) then *detection* (`yolo11n.pt`) | Your machine | No | No | A single label (plus confidence when cls works) | Fast, offline, repeatable; no per-call cost | Requires local weights; limited label set; can be generic/wrong | Default for offline demos and batch runs |
| **Google Vision (`vision="google"`)** | Cloud label detection + object localization API | Google Cloud | Yes | **Yes** (`GOOGLE_VISION_API_KEY`) | Top clothing-ish labels with scores | Good general-purpose labels; no local weights to manage | Latency + quota/cost; privacy (uploads images); requires API key | When you want reasonable labels without local ML setup |
| **CLIP (`vision="clip"`)** | Local CLIP semantic match (`openai/clip-vit-base-patch32`) vs recommender vocab + generic vocab | Your machine | First run downloads weights | No | Best match with similarity score (or top-2 generic) | Aligns well to your own item vocabulary (`ALL_ITEMS`); flexible | Heavy (RAM/VRAM); slower; threshold tuning matters | When you want labels that map to your recommender items |
| **Both (`vision="both"`)** | Runs YOLO + Google + CLIP and prints a side-by-side table | Mixed | Yes (for Google) | Optional | Comparison table (per backend) | Best for debugging/evaluation; shows disagreement | Slowest; most dependencies; Google costs/privacy | Benchmarking and picking the backend to trust |

---

### Weight & volume estimators (packed size estimation)

These run inside `_estimate_garment_properties(...)` and choose the “best available” in this order:

**midas_sam → midas → sam → rule_based**

| Method (in code) | Uses | Pros | Cons | Best usage |
|---|---|---|---|---|
| **rule_based** | Keyword lookup table (baseline) | Always available; fast; deterministic | Not personalized to the actual photo | Baseline / fallback |
| **midas** | MiDaS depth (`Intel/dpt-hybrid-midas`) + YOLO bbox area + heuristic corrections | Adds “3D-ish” correction without segmentation | Depth is approximate; bbox area is crude | When SAM isn’t available but you want some geometric correction |
| **sam** | SAM mask area (`sam_vit_b_01ec64.pth`) + packing formula | Much better area estimate than bbox | Needs SAM checkpoint + dependency; no depth correction | When you can segment well but can’t/ won’t use MiDaS |
| **midas_sam (★)** | SAM mask area + MiDaS depth corrections | Best accuracy in this repo’s design (better area + correction) | Heaviest setup; slowest runtime | Best quality estimates for final results/analysis |

---

### Model file / download locations (quick reference)

- **Local files next to `image_recognition.py`**:
  - `yolo11n-cls.pt`, `yolo11n.pt`, `sam_vit_b_01ec64.pth`
- **HuggingFace auto-download/cache**:
  - CLIP: `openai/clip-vit-base-patch32`
  - MiDaS: `Intel/dpt-hybrid-midas`
