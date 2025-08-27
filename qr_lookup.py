# qr_lookup.py
import base64
from typing import Optional, Dict, Any

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth

LIST_API_BASE = "http://10.74.18.162:8075/db/copyproof/sticker"
API_USER = "testerzero"
API_PASS = "lessthanone"
TIMEOUT   = 30

_auth = HTTPBasicAuth(API_USER, API_PASS)


def _strip_data_url(b64_str: str) -> str:
    """Buang prefix data URL bila ada (mis. 'data:image/png;base64,...')."""
    b64_str = (b64_str or "").strip()
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,", 1)[1].strip()
    return b64_str


def _decode_base64_to_cv(b64_str: str) -> np.ndarray:
    """Decode base64 (PNG/JPG) ke gambar OpenCV (BGR)."""
    raw = base64.b64decode(b64_str, validate=False)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Base64 ter-decode tetapi bukan gambar PNG/JPG yang valid.")
    return img


def _read_qr_text(img_bgr: np.ndarray) -> str:
    """Baca QR memakai OpenCV. Kembalikan string ('' jika tak terbaca)."""
    detector = cv2.QRCodeDetector()
    text, pts, _ = detector.detectAndDecode(img_bgr)
    return text or ""


def _extract_sticker_id_from_json(data: Any) -> Optional[str]:
    """
    Robust parser untuk berbagai bentuk respons:
      - {"sticker_id": "..."}
      - {"@name": "..."}
      - {"items": [{ "sticker_id": "..." }, ...]}
      - [{"sticker_id": "..."}, ...]
    """
    if isinstance(data, dict):
        if isinstance(data.get("sticker_id"), str):
            return data["sticker_id"]
        if isinstance(data.get("@name"), str):
            return data["@name"]
        for key in ("items", "data", "results", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and isinstance(row.get("sticker_id"), str):
                        return row["sticker_id"]
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and isinstance(row.get("sticker_id"), str):
                return row["sticker_id"]
    return None


def qr_to_kode_unik(qr_data: str) -> str:
    """
    Ubah qr_data jadi kode_unik dengan mengganti prefix:
      STICKERTEST001-1755845004-00001  ->  AHM-PM-1755845004-00001
    """
    if not qr_data or "-" not in qr_data:
        raise ValueError("qr_data tidak valid")
    _, rest = qr_data.split("-", 1)
    return "AHM-PM-" + rest


def get_qr_and_sticker_id_from_base64(
    raw_base64: str,
    *,
    list_api_base: str = LIST_API_BASE,
    auth: HTTPBasicAuth = _auth,
    timeout: int = TIMEOUT,
) -> Dict[str, Any]:
    """
    1) Decode base64 → OpenCV image
    2) Baca QR payload dari gambar
    3) GET ke {list_api_base}/@list_sticker?sticker_qr_code={qr}
    4) Ambil sticker_id dari JSON

    Return:
      { ok, qr_payload, sticker_id, error?, raw_sample? }
    """
    # decode -> image
    b64 = _strip_data_url(raw_base64)
    img = _decode_base64_to_cv(b64)

    # read QR
    qr_text = _read_qr_text(img)
    if not qr_text:
        return {"ok": False, "qr_payload": None, "sticker_id": None, "error": "QR tidak terdeteksi / tidak terbaca."}

    # call list API
    url = f"{list_api_base}/@list_sticker"
    try:
        resp = requests.get(url, params={"sticker_qr_code": qr_text}, auth=auth, timeout=timeout)
    except requests.RequestException as e:
        return {"ok": False, "qr_payload": qr_text, "sticker_id": None, "error": f"Gagal hubungi list API: {e}"}

    if resp.status_code != 200:
        sample = None
        try:
            sample = resp.text[:500]
        except Exception:
            pass
        return {
            "ok": False,
            "qr_payload": qr_text,
            "sticker_id": None,
            "error": f"List API HTTP {resp.status_code}",
            "raw_sample": sample,
        }

    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "qr_payload": qr_text, "sticker_id": None, "error": "List API bukan JSON valid."}

    sticker_id = _extract_sticker_id_from_json(data)
    if not sticker_id:
        return {
            "ok": False,
            "qr_payload": qr_text,
            "sticker_id": None,
            "error": "Tidak menemukan field 'sticker_id' pada respons.",
        }

    return {"ok": True, "qr_payload": qr_text, "sticker_id": sticker_id}
