import os
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import qrcode
import time
from skimage.metrics import structural_similarity as ssim
from cv2 import GaussianBlur


# ---------------- util font ----------------
def _load_font(size_px: int):
    try:
        return ImageFont.truetype("arial.ttf", size_px)
    except Exception:
        return ImageFont.load_default()

def _font_fit(draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int,
              min_size: int = 4, max_size: int = 300):
    """Cari ukuran font terbesar yang muat di (max_w, max_h)."""
    lo, hi = min_size, max_size
    best = _load_font(min_size)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = _load_font(mid)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if (r - l) <= max_w and (b - t) <= max_h:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1
    return best

# ---------------- fungsi utama ----------------

def buat_label_keaslian(
    counter=1,
    output_name: str = "",
    dpi: int = 1200,
    *,
    fft_amp: float = 1.5,
    fft_fx: int = 10,
    fft_fy: int = 6,
    fp_ratio: float = 0.10,
    seed: int | None = None,
    prefix_kode: str = "",
    prefix_qr_payload: str = "STICKERTEST001-",
    fixed_name: str | None = None
):
    """
    Label 12mm x 15mm @1200dpi:
      - QR fix 9x9 mm di tengah
      - watermark teks 'AHM' halus
      - kode_unik di atas QR
      - fingerprint (noise acak) di tengah QR
      - FFT-watermark di background
    """
    if counter is None or str(counter).strip() == "":
        counter = 0
    kode_unik = str(counter).strip().zfill(5)

    ts = str(int(time.time()) % (10**10)).zfill(10)
    basename_core = f"{ts}-{kode_unik}"
    qr_payload   = f"{prefix_qr_payload}{basename_core}"
    kode_teks    = f"{prefix_kode}{basename_core}"

    if fixed_name:
        final_png_name = fixed_name if fixed_name.lower().endswith(".png") else (fixed_name + ".png")
    else:
        final_png_name = f"{basename_core}.png"

    os.makedirs("data", exist_ok=True)
    if seed is not None:
        np.random.seed(seed)

    mm_to_px = lambda mm: int(round((mm / 25.4) * dpi))
    width_px, height_px = mm_to_px(12.0), mm_to_px(15.0)
    qr_side = mm_to_px(9.0)

    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)

    # watermark AHM
    spacing = max(mm_to_px(2.0), 6)
    wm_font = _load_font(max(mm_to_px(0.9), 6))
    for y in range(0, height_px, spacing):
        for x in range(0, width_px, spacing):
            draw.text((x, y), "AHM", fill=(205, 205, 205), font=wm_font)

    # QR
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize(
        (qr_side, qr_side), resample=Image.NEAREST
    )
    qr_x = (width_px - qr_side) // 2
    qr_y = (height_px - qr_side) // 2 + mm_to_px(0.5)  # naik sedikit biar ada space teks bawah
    img.paste(img_qr, (qr_x, qr_y))

    # kode unik di atas QR
    font_kode = _font_fit(draw, kode_teks, max_w=qr_side, max_h=mm_to_px(1.0))
    l, t, r, b = draw.textbbox((0, 0), kode_teks, font=font_kode)
    text_w, text_h = r - l, b - t
    text_x = (width_px - text_w) // 2
    text_y = qr_y - text_h - mm_to_px(0.3)
    draw.text((text_x, text_y), kode_teks, fill="black", font=font_kode)

    # ========================================
    # Hidden text di header (atas kode unik) - COPYPROOF
    # ========================================
    hidden_text = "COPY"  # teks tersembunyi
    font_hidden = _load_font(mm_to_px(1.8))  # font untuk mask teks

    # Buat mask teks hidden
    mask_img = Image.new("L", (width_px, height_px), 0)
    draw_mask = ImageDraw.Draw(mask_img)
    l, t, r, b = draw_mask.textbbox((0, 0), hidden_text, font=font_hidden)
    tw, th = r - l, b - t
    hx = (width_px - tw) // 2
    hy = text_y - th - mm_to_px(0.5)  # posisinya di atas kode_unik
    draw_mask.text((hx, hy), hidden_text, fill=255, font=font_hidden)
    mask = np.array(mask_img) > 128

    # Buat pola sinusoidal high-frequency
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    pattern = (np.sin(2 * np.pi * (xx / 3)) * 6).astype(np.int8)  # amplitude rendah, freq tinggi

    # Terapkan hanya di area teks tersembunyi
    arr = np.array(img)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + (pattern * mask), 0, 255)

    # Replace gambar
    img = Image.fromarray(arr)

    # fingerprint
    noise_radius = max(2, int(round(qr_side * fp_ratio)))
    cx, cy = qr_x + qr_side // 2, qr_y + qr_side // 2
    noise = np.random.randint(0, 256, (2*noise_radius, 2*noise_radius), dtype=np.uint8)
    noise_img = Image.fromarray(np.stack([noise] * 3, axis=2))
    mask_fp = Image.new("L", (2*noise_radius, 2*noise_radius), 0)
    ImageDraw.Draw(mask_fp).ellipse((0, 0, 2*noise_radius, 2*noise_radius), fill=255)
    img.paste(noise_img, (cx-noise_radius, cy-noise_radius), mask_fp)

    # =========================================================
    # FFT-watermark v2 — anti-copy adaptive sinusoidal embedding
    # =========================================================
    arr = np.array(img)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]

    # mask area (hindari QR)
    wm_mask = np.ones_like(Y, np.float32)
    margin = 3
    wm_mask[qr_y - margin:qr_y + qr_side + margin, qr_x - margin:qr_x + qr_side + margin] = 0.0

    # koordinat
    yy, xx = np.mgrid[0:Y.shape[0], 0:Y.shape[1]]

    # pola sinusoidal diagonal ganda (lebih kompleks)
    fx = 30 + np.random.randint(-3,4)
    fy = 22 + np.random.randint(-3,4)

    yy, xx = np.mgrid[0:Y.shape[0], 0:Y.shape[1]]
    pattern = (
            np.sin(2 * np.pi * (fx * xx / Y.shape[1] + fy * yy / Y.shape[0])) +
            0.5 * np.sin(2 * np.pi * (fx * xx / Y.shape[1] - fy * yy / Y.shape[0]))
    ).astype(np.float32)

    # amplitudo adaptif berdasar luminance lokal
    amp_base = 0.25  # rata-rata amplitudo dasar
    amp_map = amp_base * (0.3 + 0.7 * (Y / 255.0) ** 2)

    # tambahkan sedikit noise acak untuk menghindari pattern printer
    rand_noise = np.random.normal(0, 0.25, Y.shape).astype(np.float32)

    # gabungkan semua
    Y_mod = Y + (amp_map * pattern + 0.25 * rand_noise) * wm_mask

    # clamp
    Y_mod = np.clip(Y_mod, 0, 255)
    ycrcb[:, :, 0] = Y_mod

    # ubah balik ke RGB
    img = Image.fromarray(cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB))

    # simpan
    output_path = os.path.join("data", final_png_name)
    base_noext = os.path.splitext(output_path)[0]
    fingerprint_path = base_noext + "_fingerprint.npy"
    meta_path = base_noext + "_meta.npy"

    # --- tambahkan meta fingerprint + FFT ---
    cx_rel = cx / width_px
    cy_rel = cy / height_px
    r_rel  = noise_radius / width_px

    np.save(fingerprint_path, noise)

    np.save(meta_path, {
        "dpi": dpi,
        "size_px": (width_px, height_px),
        "size_mm": (12.0, 15.0),
        "qr_box": [qr_x, qr_y, qr_x+qr_side, qr_y+qr_side],
        "kode_unik": kode_teks,
        "qr_payload": qr_payload,
        "cx_rel": cx_rel,
        "cy_rel": cy_rel,
        "r_rel": r_rel,
        "fft": {"fx": fx, "fy": fy, "amp": amp_base}
    })
    os.sync()

    img.save(output_path, dpi=(dpi, dpi))
    created_files = [output_path, fingerprint_path, meta_path]

    print(f"✅ Label disimpan: {output_path}")
    return {
        "output_path": output_path,
        "fingerprint_path": fingerprint_path,
        "meta_path": meta_path,
        "output_name": final_png_name,
        "kode_unik_text": kode_teks,
        "qr_payload": qr_payload,
        "width_px": width_px,
        "height_px": height_px,
        "dpi": dpi,
        "created_files": created_files,
    }




def _cari_path(kandidat):
    """Balik path valid. Coba persis, lalu prefix 'data/'."""
    if kandidat and os.path.exists(kandidat):
        return kandidat
    alt = os.path.join("data", kandidat) if kandidat else None
    if alt and os.path.exists(alt):
        return alt
    return None

def _local_energy(Mag, x, y, r=3):
    h, w = Mag.shape
    x0, x1 = max(0, x-r), min(w, x+r+1)
    y0, y1 = max(0, y-r), min(h, y+r+1)
    return float(Mag[y0:y1, x0:x1].mean())

def _fft_snr(img_bgr, meta_fft=None, fft_snr_radius=3, fft_fx=10, fft_fy=6, save_dbg=None):
    """
    Hitung SNR watermark periodik pada luminance (Y) dengan masking background.
    meta_fft (opsional): dict {'fx','fy','bar_h','qr_box':[x1,y1,x2,y2]}
    Return: (snr, details_dict)
    """
    # luminance
    Y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
    H, W = Y.shape

    # mask background: 1=ikut dihitung, 0=abaikan
    mask = np.ones((H, W), np.float32)

    # dari meta (jika ada)
    if meta_fft and isinstance(meta_fft, dict):
        fx = int(meta_fft.get("fx", fft_fx))
        fy = int(meta_fft.get("fy", fft_fy))
        bar_h = int(meta_fft.get("bar_h", 0))
        qr_box = meta_fft.get("qr_box", None)
    else:
        fx, fy, bar_h, qr_box = fft_fx, fft_fy, 0, None

    # exclude bars
    if bar_h > 0:
        mask[:bar_h, :] = 0.0
        mask[H-bar_h:, :] = 0.0

    # exclude QR box (+margin)
    if qr_box and len(qr_box) == 4:
        x1, y1, x2, y2 = [int(v) for v in qr_box]
        mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = 0.0

    # windowing + masking
    win_y = np.hanning(H).astype(np.float32)
    win_x = np.hanning(W).astype(np.float32)
    window = np.outer(win_y, win_x)
    Z = Y * mask * window

    # FFT magnitude (log)
    F = np.fft.fftshift(np.fft.fft2(Z))
    Mag = np.abs(F)

    # target peaks (simetris)
    cx, cy = W//2, H//2
    tx1, ty1 = cx + fx, cy + fy
    tx2, ty2 = cx - fx, cy - fy

    signal = max(_local_energy(Mag, tx1, ty1, fft_snr_radius),
                 _local_energy(Mag, tx2, ty2, fft_snr_radius))
    noise = float(np.median(Mag[(mask>0)])) if np.any(mask>0) else float(np.median(Mag))
    snr = (signal + 1e-6) / (noise + 1e-6)

    # debug simpan spectrum
    dbg_paths = {}
    if save_dbg:
        os.makedirs(os.path.dirname(save_dbg), exist_ok=True)
        # spektrum log agar terlihat
        Mag_log = np.log(Mag + 1.0)
        Mag_norm = cv2.normalize(Mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        spec_path = save_dbg.replace(".png", "_fft.png")
        mask_vis = (mask*255).astype(np.uint8)
        mask_path = save_dbg.replace(".png", "_mask.png")
        cv2.imwrite(spec_path, Mag_norm)
        cv2.imwrite(mask_path, mask_vis)
        dbg_paths = {"fft_spec": spec_path, "fft_mask": mask_path}

    details = {
        "fx": int(fx), "fy": int(fy),
        "signal": signal, "noise_med": noise,
        "peaks": {"p1": [int(tx1), int(ty1)], "p2": [int(tx2), int(ty2)]},
        "debug": dbg_paths
    }
    return float(snr), details


def _check_hidden_text(gray, meta, contrast_threshold=50):
    hx1, hy1, hx2, hy2 = meta["hidden_roi"]  # rel koordinat
    H, W = gray.shape
    x1, x2 = int(hx1*W), int(hx2*W)
    y1, y2 = int(hy1*H), int(hy2*H)
    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return False, 0.0

    lap = cv2.Laplacian(roi, cv2.CV_64F)
    contrast_val = lap.var()
    return bool(contrast_val > contrast_threshold), float(contrast_val)


def verifikasi_label_fleksibel(
    foto_path: str | None = None,
    output_name: str = "label_ali.png",
    threshold_mse: int = 12000,
    threshold_ssim: float = 0.5,
    simpan_debug: bool = True,
    fft_snr_threshold: float = 2.0
):

    def _make_json_safe(x):
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None:
            if isinstance(x, _np.generic):
                return x.item()
            if isinstance(x, _np.ndarray):
                return x.tolist()
        if isinstance(x, dict):
            return {str(k): _make_json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_make_json_safe(v) for v in x]
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8")
            except Exception:
                return str(x)
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    # ----- Ambil fingerprint & meta -----
    base = os.path.splitext(output_name)[0]
    fingerprint_path = _cari_path(f"{base}_fingerprint.npy")
    meta_path        = _cari_path(f"{base}_meta.npy")
    if not fingerprint_path or not meta_path:
        raise FileNotFoundError("Fingerprint/meta tidak ditemukan di folder data/.")

    fingerprint = np.load(fingerprint_path)
    meta = np.load(meta_path, allow_pickle=True).item()
    cx_rel, cy_rel, r_rel = float(meta["cx_rel"]), float(meta["cy_rel"]), float(meta["r_rel"])
    meta_fft = meta.get("fft", None)

    # ----- Tentukan sumber gambar -----
    if foto_path is None:
        sumber = _cari_path(output_name)
        if not sumber:
            raise FileNotFoundError(f"File generate tidak ditemukan: data/{output_name}")
    else:
        sumber = _cari_path(foto_path)
        if not sumber:
            raise FileNotFoundError(f"Foto tidak ditemukan: {foto_path}")

    img_bgr = cv2.imread(sumber)
    if img_bgr is None:
        raise ValueError("Gagal membaca gambar. Format tidak didukung.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ----- ROI fingerprint -----
    cx = int(round(cx_rel * W))
    cy = int(round(cy_rel * H))
    r  = int(round(r_rel  * W))
    r = max(4, min(r, cx, cy, W - cx - 1, H - cy - 1))
    y1, y2 = cy - r, cy + r
    x1, x2 = cx - r, cx + r
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Crop kosong—meta/posisi tidak cocok dengan gambar.")

    crop_resized = cv2.resize(
        crop, (fingerprint.shape[1], fingerprint.shape[0]), interpolation=cv2.INTER_AREA
    )

    # --- Gaussian smoothing untuk SSIM agar lebih stabil ---
    crop_smooth = GaussianBlur(crop_resized, (3, 3), 0)
    fingerprint_smooth = GaussianBlur(fingerprint, (3, 3), 0)

    # --- MSE + SSIM ---
    mse = float(np.mean((crop_resized.astype("float32") - fingerprint.astype("float32")) ** 2))
    ssim_val = ssim(fingerprint_smooth, crop_smooth, data_range=255)

    # --- Modifikasi fp_score: fokus MSE, beri floor minimal pada SSIM ---
    mse_norm = 1 - min(mse / (threshold_mse * 1.2), 1.0)
    ssim_floor = max(ssim_val, 0.01)
    fp_score = 0.95 * mse_norm + 0.05 * ssim_floor
    fp_ok = fp_score > 0.35

    # ----- QR (3 cara) -----
    qr = cv2.QRCodeDetector()
    data_raw, _, _ = qr.detectAndDecode(img_bgr)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data_thr, _, _ = qr.detectAndDecode(thr)
    qr_half = int(r * 2.5)
    Y1, Y2 = max(0, cy - qr_half), min(H, cy + qr_half)
    X1, X2 = max(0, cx - qr_half), min(W, cx + qr_half)
    crop_qr = img_bgr[Y1:Y2, X1:X2]
    data_crop, _, _ = qr.detectAndDecode(crop_qr)
    qr_data = data_raw or data_thr or data_crop or ""
    qr_ok = bool(qr_data)

    # ----- FFT-watermark -----
    gray_norm = cv2.equalizeHist(gray)
    img_bgr_eq = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)
    dbg_base = os.path.join("data", f"{base}_dbg") if simpan_debug else None
    fft_snr, fft_detail = _fft_snr(
        img_bgr_eq,
        meta_fft=meta_fft,
        fft_snr_radius=3,
        fft_fx=(meta_fft or {}).get("fx", 10),
        fft_fy=(meta_fft or {}).get("fy", 6),
        save_dbg=(dbg_base + "_spec.png") if simpan_debug else None
    )
    wm_ok = bool(fft_snr >= float(fft_snr_threshold))

    # ----- Hidden text -----
    hidden_ok, hidden_val = False, 0.0
    if "hidden_roi" in meta:
        hidden_ok, hidden_val = _check_hidden_text(gray, meta, contrast_threshold=50)

    # ----- Debug save -----
    debug_paths = {}
    if simpan_debug:
        os.makedirs("data", exist_ok=True)
        base_dbg = os.path.join("data", base)
        dbg_crop_path = f"{base_dbg}_dbg_crop.png"
        dbg_thr_path  = f"{base_dbg}_dbg_thr.png"
        dbg_qr_path   = f"{base_dbg}_dbg_qr_crop.png"
        cv2.imwrite(dbg_crop_path, crop_resized)
        cv2.imwrite(dbg_thr_path,   thr)
        if crop_qr.size:
            cv2.imwrite(dbg_qr_path, crop_qr)
        debug_paths.update({"crop_resized": dbg_crop_path, "thr": dbg_thr_path, "qr_crop": dbg_qr_path})
        if isinstance(fft_detail, dict):
            debug_paths.update(fft_detail.get("debug", {}))

    # ----- Ensemble decision -----
    reason_text = ""
    if fp_ok and wm_ok and not hidden_ok:
        status_code = "ASLI"
        status = "✅ ASLI"
        reason_text = "Fingerprint cocok, watermark terdeteksi, hidden text belum muncul."
    elif hidden_ok:
        status_code = "SALINAN"
        status = "❌ SALINAN"
        reason_text = "Hidden text muncul → kemungkinan hasil copy/scan."
    elif wm_ok or fp_ok or qr_ok:
        status_code = "RUSAK"
        parts = []
        if not wm_ok: parts.append("watermark tidak terdeteksi")
        if not fp_ok: parts.append("fingerprint tidak cocok")
        if not qr_ok: parts.append("QR sulit/tidak terbaca")
        reason_text = " / ".join(parts) if parts else "Sinyal tidak ideal."
        status = f"⚠️ RUSAK — {reason_text}"
    else:
        status_code = "SALINAN"
        status = "❌ SALINAN"
        reason_text = "Fingerprint tidak cocok, watermark tidak terdeteksi, QR tidak terbaca."

    # ----- Return -----
    result = {
        "source": str(sumber),
        "mse": float(round(mse, 2)),
        "threshold_mse": int(threshold_mse),
        "ssim": float(round(ssim_val, 3)),
        "threshold_ssim": float(threshold_ssim),
        "fp_score": float(round(fp_score, 3)),
        "fp_ok": bool(fp_ok),
        "fft_snr": float(round(fft_snr, 2)),
        "fft_threshold": float(fft_snr_threshold),
        "wm_ok": bool(wm_ok),
        "qr_data": str(qr_data or ""),
        "qr_ok": bool(qr_ok),
        # tambahan hidden text
        "hidden_ok": bool(hidden_ok),
        "hidden_val": float(round(hidden_val, 2)),
        "status": str(status),
        "status_code": str(status_code),
        "reason": str(reason_text),
        "paths": {
            "fingerprint": str(fingerprint_path),
            "meta": str(meta_path),
            "debug": debug_paths
        },
        "fft_detail": fft_detail
    }

    result_safe = _make_json_safe(result)
    return result_safe



def file_to_base64(path_file: str) -> str:
    with open(path_file, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def buat_label_keaslian_b64(kode_unik: str,
                            output_name: str = "label_ali.png",
                            dpi: int = 1200,
                            simpan_b64: bool = True):
    """
    Membuat label keaslian + hasilnya dikonversi ke Base64.
    - kode_unik: string yang akan ditaruh di atas QR
    - output_name: nama file PNG hasil
    - dpi: resolusi cetak
    - simpan_b64: jika True, simpan Base64 ke file .txt
    """
    os.makedirs("data", exist_ok=True)

    # 1) Buat label
    buat_label_keaslian(kode_unik=kode_unik, output_name=output_name, dpi=dpi)

    # 2) Lokasi file hasil
    output_path = os.path.join("data", output_name)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"File {output_path} tidak ditemukan setelah proses generate.")

    # 3) Konversi ke base64
    b64 = file_to_base64(output_path)

    # 4) Simpan base64 ke file txt (opsional)
    b64_path = os.path.splitext(output_path)[0] + "_base64.txt"
    if simpan_b64:
        with open(b64_path, "w", encoding="utf-8") as f:
            f.write(b64)

    return {
        "output_path": output_path,
        "base64_path": b64_path if simpan_b64 else None,
        "base64": b64
    }

def base64txt_to_file(txt_path: str, output_path: str):
    """
    Membaca file TXT yang berisi Base64, lalu menyimpannya sebagai file biner (mis. PNG).
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File Base64 TXT tidak ditemukan: {txt_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as f:
        b64_string = f.read().strip()

    with open(output_path, "wb") as out_f:
        out_f.write(base64.b64decode(b64_string))

    return output_path


if __name__ == "__main__":
    for i in range(9):
        n = i + 1
        hasil = verifikasi_label_fleksibel(f"copy{n}.jpg", "sticker_copyproof.png", 12000)
        print(hasil)

    for i in range(10):
        n = i + 1
        hasil = verifikasi_label_fleksibel(f"asli{n}.jpg", "sticker_copyproof1.png", 12000)
        print(hasil)
    # b = base64txt_to_file("photo_2025-08-28_13-56-32_base64.txt", "output.png")
    # print(b)
    # a = buat_label_keaslian()
    # print(a)
    # buat_label_keaslian_b64()
