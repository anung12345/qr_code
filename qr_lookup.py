# qr_lookup.py
import base64
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from pyzbar.pyzbar import decode

LIST_API_BASE = "http://10.84.136.64:8075/db/copyproof/sticker"
API_USER = "testerzero"
API_PASS = "lessthanone"
TIMEOUT   = 30

_auth = HTTPBasicAuth(API_USER, API_PASS)


# ----------------------- helpers umum -----------------------
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


def _cap_max_side(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    """Batasi sisi terpanjang agar tak kelewat besar (hemat memori/waktu)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ----------------------- preprocessing & pembacaan QR -----------------------
def _enhance_for_qr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline ringan untuk meningkatkan kontras & ketajaman:
    - convert ke gray
    - CLAHE
    - unsharp masking
    - adaptive threshold (membantu QR kecil/kontras rendah)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE (kontras lokal)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # Unsharp masking
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)

    # Adaptive threshold (binarize tapi tetap simpan grayscale juga sebagai alternatif)
    th = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    # Kembalikan 3-channel BGR dari binary agar kompatibel
    th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return th_bgr


def _try_decode(det: cv2.QRCodeDetector, img: np.ndarray) -> Tuple[str, bool]:
    """Coba detectAndDecode lebih dulu, lalu detectAndDecodeMulti."""
    # single
    txt, pts, _ = det.detectAndDecode(img)
    if isinstance(txt, str) and txt.strip():
        return txt.strip(), True

    # multi
    try:
        texts, points, _ = det.detectAndDecodeMulti(img)
        if isinstance(texts, (list, tuple)):
            for t in texts:
                if isinstance(t, str) and t.strip():
                    return t.strip(), True
    except Exception:
        pass

    return "", False


def _read_qr_text(img_bgr: np.ndarray) -> str:
    """
    Baca QR dengan strategi multi-scale & preprocessing:
    - batasi ukuran gambar sangat besar
    - coba langsung (as-is)
    - coba dengan preprocessing (CLAHE + unsharp + adaptive threshold)
    - coba berbagai skala (down & up) pada kedua versi (as-is dan enhanced)
    - fallback: pyzbar (lebih toleran)
    """
    img_bgr = _cap_max_side(img_bgr, 2000)

    det = cv2.QRCodeDetector()

    # 1) coba langsung
    txt, ok = _try_decode(det, img_bgr)
    if ok:
        return txt

    # 2) enhanced
    enh = _enhance_for_qr(img_bgr)
    txt, ok = _try_decode(det, enh)
    if ok:
        return txt

    # 3) multi-scale (down & up) di kedua versi
    scales = [0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0]
    for base in (img_bgr, enh):
        h, w = base.shape[:2]
        for s in scales:
            nw, nh = int(w * s), int(h * s)
            # hindari terlalu kecil
            if nw < 120 or nh < 120:
                continue
            # hindari terlalu besar (limit 2400)
            if max(nw, nh) > 2400:
                continue
            resized = cv2.resize(
                base, (nw, nh),
                interpolation=cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
            )
            txt, ok = _try_decode(det, resized)
            if ok:
                return txt

    # 4) fallback sangat kecil: perbesar agresif dari grayscale polos
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for s in (3.0, 4.0, 5.0):
        big = cv2.resize(gray, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        txt, ok = _try_decode(det, big_bgr)
        if ok:
            return txt

    # 5) fallback terakhir: pyzbar
    decoded = decode(img_bgr)
    if decoded:
        return decoded[0].data.decode("utf-8")

    return ""


# ----------------------- parsing & API list -----------------------
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
    return rest


def get_qr_and_sticker_id_from_base64(
    raw_base64: str,
    *,
    list_api_base: str = LIST_API_BASE,
    auth: HTTPBasicAuth = _auth,
    timeout: int = TIMEOUT,
) -> Dict[str, Any]:
    """
    1) Decode base64 → OpenCV image
    2) Baca QR payload dari gambar (robust)
    3) GET ke {list_api_base}/@list_sticker?sticker_qr_code={qr}
    4) Ambil sticker_id dari JSON

    Return:
      { ok, qr_payload, sticker_id, error?, raw_sample? }
    """
    # decode -> image
    b64 = _strip_data_url(raw_base64)
    img = _decode_base64_to_cv(b64)

    # read QR (dengan strategi multi-scale)
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


# ----------------------- entry point tanpa argparse -----------------------
def main(
    raw_base64: str,
    *,
    list_api_base: str = LIST_API_BASE,
    user: str = API_USER,
    password: str = API_PASS,
    timeout: int = TIMEOUT,
) -> Dict[str, Any]:
    """
    Jalankan lookup hanya dengan parameter (tanpa CLI/argparse).
    Mengembalikan dict hasil yang sama dengan get_qr_and_sticker_id_from_base64.
    """
    auth = HTTPBasicAuth(user, password)
    result = get_qr_and_sticker_id_from_base64(
        raw_base64,
        list_api_base=list_api_base,
        auth=auth,
        timeout=timeout,
    )
    # print ringkas biar keliatan saat dipanggil langsung
    try:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(result)
    return result


# Opsional: contoh minimal kalau file ini dieksekusi langsung
if __name__ == "__main__":
    import sys
    def run_from_txt(txt_path: str):
        # baca string base64 dari file .txt
        with open(txt_path, "r", encoding="utf-8") as f:
            b64_str = f.read().strip()
        # jalankan pipeline lookup (decode -> baca QR -> hit API list)
        return main(b64_str)


    if len(sys.argv) < 2:
        print("Usage: python run_qr_lookup.py <path_ke_file_base64.txt>")
        sys.exit(1)
    run_from_txt(sys.argv[1])
