"""
Cloudflare R2 Storage Client (S3 Compatible)
Handles uploading user wardrobe images.

Changes from original:
- upload_image() now returns dict {url, size_bytes} instead of plain str
- Added list_images_in_folder() to retrieve all images in a trip's R2 folder
- R2 folder structure: {city}_{country}_{start}_to_{end}/{uuid}.{ext}
- public=True by default (for testing as requested)
"""
import uuid
import boto3
from botocore.exceptions import ClientError
from config.settings import (
    R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME,
    DEPS_AVAILABLE, LOCAL_UPLOAD_DIR, R2_PUBLIC_BASE_URL
)

_s3_client = None

def _get_s3_client():
    global _s3_client
    if _s3_client is None and DEPS_AVAILABLE["boto3"] and R2_ENDPOINT_URL:
        _s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto'  # R2 specific
        )
    return _s3_client


def _build_folder_name(trip_context: dict) -> str:
    """
    Builds R2 folder name from trip context slots.
    Format: {city}_{country}_{start}_to_{end}
    e.g. beijing_china_2025-12-12_to_2025-12-15
    Falls back to 'unknown_trip' if slots missing.
    """
    city    = trip_context.get("destination", "unknown").lower().replace(" ", "_").replace(",", "")
    country = trip_context.get("country", "unknown").lower().replace(" ", "_")
    start   = trip_context.get("start_date", "unknown")
    end     = trip_context.get("end_date",   "unknown")
    return f"{city}_{country}_{start}_to_{end}"


def upload_image(file_bytes: bytes, original_filename: str,
                 trip_context: dict = None) -> dict:
    """
    Uploads image to R2. Returns dict with url and size_bytes.
    size_bytes is the actual compressed size stored in R2.

    Public access is controlled at the bucket level in the Cloudflare R2
    dashboard (R2 → bucket → Settings → Public Access → Allow).

    Args:
        file_bytes:        Compressed image bytes to upload.
        original_filename: Original filename for extension.
        trip_context:      Slot dict used to build folder name.

    Returns:
        {"url": str, "size_bytes": int}
    """
    client = _get_s3_client()
    if not client:
        return _fallback_upload(file_bytes, original_filename)

    ext         = original_filename.rsplit(".", 1)[-1].lower()
    folder      = _build_folder_name(trip_context or {})
    object_name = f"{folder}/{uuid.uuid4()}.{ext}"
    size_bytes  = len(file_bytes)

    try:
        # Note: Cloudflare R2 does not support per-object ACL (unlike AWS S3).
        # Public access must be enabled at the bucket level in the R2 dashboard:
        # R2 → your bucket → Settings → Public Access → Allow
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=object_name,
            Body=file_bytes,
            ContentType=f"image/{ext}",
        )
        url = f"{R2_PUBLIC_BASE_URL}/{object_name}"
        return {"url": url, "size_bytes": size_bytes}

    except ClientError as e:
        print(f"[R2 Upload Error] {e}")
        return _fallback_upload(file_bytes, original_filename)


def list_images_in_folder(trip_context: dict) -> list:
    """
    Lists all image objects in the trip's R2 folder.
    Returns list of dicts: [{"url": str, "size_bytes": int, "key": str}, ...]
    Falls back to empty list if R2 unreachable.

    Used by Section 2 to display thumbnails directly from R2
    so they persist across reruns and page refreshes.
    """
    client = _get_s3_client()
    if not client:
        return []

    folder = _build_folder_name(trip_context)
    try:
        response = client.list_objects_v2(
            Bucket=R2_BUCKET_NAME,
            Prefix=f"{folder}/",
        )
        items = []
        for obj in response.get("Contents", []):
            key  = obj["Key"]
            url  = f"{R2_PUBLIC_BASE_URL}/{key}"
            items.append({
                "url":        url,
                "size_bytes": obj.get("Size", 0),
                "key":        key,
            })
        return items
    except ClientError as e:
        print(f"[R2 List Error] {e}")
        return []


def get_image_bytes(object_name: str) -> bytes | None:
    """Downloads image bytes from R2."""
    client = _get_s3_client()
    if not client:
        return _fallback_get(object_name)

    try:
        response = client.get_object(Bucket=R2_BUCKET_NAME, Key=object_name)
        return response['Body'].read()
    except ClientError:
        return None


# ── Fallback: Local Disk Storage ──────────────────────────────────────────────
from pathlib import Path

def _fallback_upload(file_bytes: bytes, original_filename: str) -> dict:
    print("[FALLBACK MODE] Cloudflare R2 unreachable. Saving image locally.")
    ext        = original_filename.rsplit(".", 1)[-1].lower()
    local_path = LOCAL_UPLOAD_DIR / f"{uuid.uuid4()}.{ext}"
    local_path.write_bytes(file_bytes)
    return {"url": str(local_path), "size_bytes": len(file_bytes)}

def _fallback_get(object_name: str) -> bytes | None:
    for f in LOCAL_UPLOAD_DIR.iterdir():
        if object_name in f.name:
            return f.read_bytes()
    return None