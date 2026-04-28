#!/usr/bin/env python3
"""
SAM Setup — download Segment Anything Model ViT-B checkpoint
=============================================================
Run once before the SAM or MiDaS+SAM estimation methods are used.

Usage:
    python download_sam.py

What it downloads
-----------------
    sam_vit_b_01ec64.pth  (~375 MB)
    Source: https://dl.fbaipublicfiles.com/segment_anything/

MiDaS (Intel/dpt-hybrid-midas) is auto-downloaded by HuggingFace
on first use — no script needed for that one.
"""

import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
CHECKPOINT_DST = Path(__file__).parent / "sam_vit_b_01ec64.pth"


def _check_segment_anything() -> bool:
    try:
        import segment_anything  # noqa: F401
        return True
    except ImportError:
        return False


def _show_progress(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb  = downloaded / 1024 / 1024
        print(f"\r    {pct:3d}%  {mb:.1f} MB downloaded", end="", flush=True)


def main() -> None:
    print("=" * 60)
    print("SAM (Segment Anything Model) Setup")
    print("=" * 60)

    # 1. Check segment-anything library
    if not _check_segment_anything():
        print("\n[!] segment-anything is not installed.")
        print("    Run: pip install segment-anything")
        print("    Then re-run: python download_sam.py")
        sys.exit(1)

    import segment_anything
    print(f"\n  segment-anything : installed")

    # 2. Download checkpoint
    if CHECKPOINT_DST.exists():
        size_mb = CHECKPOINT_DST.stat().st_size / 1024 / 1024
        print(f"  Checkpoint       : already exists ({size_mb:.1f} MB) — skipping download")
    else:
        print(f"\n  Downloading SAM ViT-B checkpoint (~375 MB)...")
        print(f"  URL : {CHECKPOINT_URL}")
        print(f"  Dest: {CHECKPOINT_DST}")
        try:
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_DST, _show_progress)
            print()  # newline after progress
            size_mb = CHECKPOINT_DST.stat().st_size / 1024 / 1024
            print(f"  Done ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print(f"  Download manually and place at: {CHECKPOINT_DST}")
            sys.exit(1)

    # 3. Quick smoke-test
    print("\n  Verifying checkpoint loads correctly...")
    try:
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_DST))
        sam.eval()
        print("  [OK] SAM ViT-B loaded successfully.")
    except Exception as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\nSetup complete. Both methods are now available:")
    print("  sam       — SAM precise mask + rule-based thickness")
    print("  midas_sam — MiDaS depth × SAM mask  (best accuracy)")
    print("\nExample:")
    print("  python main.py --city Tokyo --start 2026-12-10 --end 2026-12-15 \\")
    print("                 --purpose tourism --images img/ --vision yolo --json")


if __name__ == "__main__":
    main()
