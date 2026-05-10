#!/usr/bin/env python3
"""Convert receipt JPGs to PDFs and extract amount + description via Claude vision."""

import base64
import io
import re
import sys
from pathlib import Path

import anthropic
from PIL import Image


INPUT_DIR = Path("input_jpg")
OUTPUT_DIR = Path("output_pdf")
JPG_GLOBS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")


def collect_jpgs(folder: Path) -> list[Path]:
    paths = sorted({p for ext in JPG_GLOBS for p in folder.glob(ext)})
    if not paths:
        print(f"No JPG files found in {folder}/", file=sys.stderr)
    return paths


def to_pdf(jpg: Path, out_dir: Path, filename: str) -> Path:
    out = out_dir / filename
    img = Image.open(jpg)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out, "PDF", resolution=100.0)
    return out


_DATE_PATTERNS = [
    # YYYY-MM-DD or YYYY_MM_DD
    (r"(\d{4})[-_](\d{2})[-_](\d{2})", "{0}-{1}-{2}"),
    # YYYYMMDD  (8 consecutive digits)
    (r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", "{0}-{1}-{2}"),
    # DD-MM-YYYY or DD_MM_YYYY
    (r"(\d{2})[-_](\d{2})[-_](\d{4})", "{2}-{1}-{0}"),
]


def date_from_filename(name: str) -> str | None:
    """Try to extract a YYYY-MM-DD date from a filename."""
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, name)
        if m:
            return fmt.format(*m.groups())
    return None


def slugify(text: str) -> str:
    """Return a filename-safe lowercase slug (letters, digits, hyphens only)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:40].strip("-") or "unknown"


def build_filename(info: dict[str, str]) -> str:
    date = slugify(info["date"])
    desc = slugify(info["description"])
    amount = slugify(info["amount"])
    return f"{date}_{desc}_{amount}.pdf"


_MAX_IMAGE_BYTES = 5 * 1024 * 1024 * 3 // 4  # base64 limit is 5 MB, so raw max is 3.75 MB
_MAX_DIMENSION = 1920


def _load_image_bytes(jpg: Path) -> bytes:
    raw = jpg.read_bytes()
    if len(raw) <= _MAX_IMAGE_BYTES:
        return raw
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    scale = min(_MAX_DIMENSION / max(img.width, img.height), 1.0)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    for quality in (85, 75, 65, 55, 45):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= _MAX_IMAGE_BYTES:
            return data
    raise RuntimeError(f"Cannot compress {jpg.name} below 5 MB")


def analyze_receipt(jpg: Path, client: anthropic.Anthropic, known_date: str | None = None) -> dict[str, str]:
    image_data = base64.standard_b64encode(_load_image_bytes(jpg)).decode()
    media_type = "image/jpeg"

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a receipt. Reply with exactly three lines:\n"
                            + ("" if known_date else "Date: <date in YYYY-MM-DD format>\n")
                            + "Description: <short merchant name or category, max 4 words>\n"
                            "Amount: <total amount with currency symbol>\n"
                            "If you cannot determine a value, write 'Unknown'."
                        ),
                    },
                ],
            }
        ],
    )

    result = {"date": known_date or "unknown", "description": "unknown", "amount": "unknown"}
    for line in message.content[0].text.splitlines():
        lower = line.lower()
        if lower.startswith("date:") and not known_date:
            result["date"] = line.split(":", 1)[1].strip()
        elif lower.startswith("description:"):
            result["description"] = line.split(":", 1)[1].strip()
        elif lower.startswith("amount:"):
            result["amount"] = line.split(":", 1)[1].strip()
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jpgs = collect_jpgs(INPUT_DIR)
    if not jpgs:
        return 1

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    print(f"Processing {len(jpgs)} receipt(s)...\n")

    for jpg in jpgs:
        print(f"  {jpg.name}")
        known_date = date_from_filename(jpg.name)
        info = analyze_receipt(jpg, client, known_date)
        pdf = to_pdf(jpg, OUTPUT_DIR, build_filename(info))
        print(f"    Date        : {info['date']}")
        print(f"    Description : {info['description']}")
        print(f"    Amount      : {info['amount']}")
        print(f"    Saved PDF   : {pdf}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
