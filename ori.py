import os
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import qrcode
import time


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
# def buat_label_keaslian(
#     counter=1,
#     output_name: str = "",     # legacy arg (diabaikan jika fixed_name != None)
#     dpi: int = 1200,
#     *,
#     fft_amp: float = 1.5,      # 0.8–2.5 disarankan
#     fft_fx: int = 10,
#     fft_fy: int = 6,
#     fp_ratio: float = 0.10,    # ukuran fingerprint relatif sisi QR (mis. 0.10 = 10%)
#     seed: int | None = None,
#     prefix_kode: str = "AHM-PM-",               # prefix untuk teks di atas QR
#     prefix_qr_payload: str = "STICKERTEST001-", # prefix isi QR
#     fixed_name: str | None = None               # paksa nama file (mis. "label_ali.png")
# ):
#     """
#     Label 6.5mm x 16mm @1200dpi:
#       - watermark teks 'AHM' halus
#       - bar hitam atas/bawah + teks
#       - kode_unik kecil & menempel di atas QR
#       - QR berisi prefix_qr_payload + <epoch>-<kodeZfill>, ex: STICKERTEST001-1724210000-00042
#       - fingerprint (noise acak) di tengah QR (fragile)
#       - FFT-watermark di background (robust, exclude area QR & bars)
#     Simpan di data/: PNG + *_fingerprint.npy + *_meta.npy
#     """
#     # format & sanitasi counter -> kode unik
#     if counter is None or str(counter).strip() == "":
#         counter = 0
#     kode_unik = str(counter).strip()
#     kode_unik_z = kode_unik.zfill(5)
#
#     # buat id waktu + nama berkas
#     ts = int(time.time())
#     ts = str(ts % (10**10)).zfill(10)
#     basename_core = f"{ts}-{kode_unik_z}"                   # contoh: 1724210000-00042
#     qr_payload   = f"{prefix_qr_payload}{basename_core}"    # contoh: STICKERTEST001-1724210000-00042
#     kode_teks    = f"{prefix_kode}{basename_core}"          # contoh: AHM-PM-1724210000-00042
#
#     # tentukan nama file final
#     if fixed_name:  # paksa nama tertentu
#         final_png_name = fixed_name if fixed_name.lower().endswith(".png") else (fixed_name + ".png")
#     else:
#         # abaikan output_name legacy; gunakan pola waktu-counter
#         final_png_name = f"{basename_core}.png"
#
#     os.makedirs("data", exist_ok=True)
#     if seed is not None:
#         np.random.seed(seed)
#
#     # ukuran kanvas
#     mm_to_px = lambda mm: int(round((mm / 25.4) * dpi))
#     width_px, height_px = mm_to_px(6.5), mm_to_px(16.0)
#
#     img = Image.new("RGB", (width_px, height_px), "white")
#     draw = ImageDraw.Draw(img)
#
#     # watermark teks 'AHM' (visual halus)
#     spacing = max(mm_to_px(2.0), 6)
#     wm_font = _load_font(max(mm_to_px(0.9), 6))
#     for y in range(0, height_px, spacing):
#         for x in range(0, width_px, spacing):
#             draw.text((x, y), "AHM", fill=(205, 205, 205), font=wm_font)
#
#     # bar hitam atas & bawah
#     bar_h = max(mm_to_px(1.1), int(height_px * 0.08))
#     draw.rectangle([0, 0, width_px, bar_h], fill="black")
#     draw.rectangle([0, height_px - bar_h, width_px, height_px], fill="black")
#
#     # teks bar
#     top_text = "JAMINAN KEASLIAN PRODUK"
#     bottom_text = "AHM"
#     font_top = _font_fit(draw, top_text, width_px - 2, bar_h - 2)
#     font_bot = _font_fit(draw, bottom_text, width_px - 2, int(bar_h * 0.6))
#     draw.text((width_px // 2, bar_h // 2), top_text, fill="white", anchor="mm", font=font_top)
#     draw.text((width_px // 2, height_px - bar_h // 2), bottom_text, fill="white", anchor="mm", font=font_bot)
#
#     # QR code
#     qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
#     qr.add_data(qr_payload)
#     qr.make(fit=True)
#     inner_h = height_px - 2 * bar_h
#     max_qr_side = int(min(width_px * 0.88, inner_h * 0.88))
#     img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize(
#         (max_qr_side, max_qr_side), resample=Image.NEAREST
#     )
#     qr_x = (width_px - max_qr_side) // 2
#     qr_y = bar_h + (inner_h - max_qr_side) // 2
#     img.paste(img_qr, (qr_x, qr_y))
#
#     # KODE UNIK: kecil & menempel di atas QR
#     max_w_teks = int(max_qr_side * 0.55)   # 55% lebar QR
#     max_h_teks = mm_to_px(0.35)            # tinggi maks ~0.35 mm
#     font_kode = _font_fit(draw, kode_teks, max_w=max_w_teks, max_h=max_h_teks)
#     l, t, r, b = draw.textbbox((0, 0), kode_teks, font=font_kode)
#     text_w, text_h = r - l, b - t
#     text_x = (width_px - text_w) // 2
#     text_y = qr_y - text_h + 1
#     text_y = max(text_y, bar_h + mm_to_px(0.15))  # safety dari bar atas
#     draw.text((text_x, text_y), kode_teks, fill="black", font=font_kode)
#
#     # fingerprint (noise acak) di tengah QR
#     noise_radius = max(2, int(round(max_qr_side * fp_ratio)))
#     cx, cy = qr_x + max_qr_side // 2, qr_y + max_qr_side // 2
#     noise = np.random.randint(0, 256, (2 * noise_radius, 2 * noise_radius), dtype=np.uint8)
#     noise_img = Image.fromarray(np.stack([noise] * 3, axis=2))
#     mask_fp = Image.new("L", (2 * noise_radius, 2 * noise_radius), 0)
#     ImageDraw.Draw(mask_fp).ellipse((0, 0, 2 * noise_radius, 2 * noise_radius), fill=255)
#     img.paste(noise_img, (cx - noise_radius, cy - noise_radius), mask_fp)
#
#     # FFT-watermark (robust) di background, exclude QR & bars
#     wm_mask = np.ones((height_px, width_px), dtype=np.float32)
#     wm_mask[:bar_h, :] = 0.0
#     wm_mask[height_px - bar_h:, :] = 0.0
#     margin = 2
#     x1 = max(0, qr_x - margin); x2 = min(width_px, qr_x + max_qr_side + margin)
#     y1 = max(0, qr_y - margin); y2 = min(height_px, qr_y + max_qr_side + margin)
#     wm_mask[y1:y2, x1:x2] = 0.0
#
#     yy, xx = np.mgrid[0:height_px, 0:width_px]
#     pattern = np.sin(2*np.pi*(fft_fx*xx/width_px + fft_fy*yy/height_px)).astype(np.float32)
#
#     arr = np.array(img)
#     ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb).astype(np.float32)
#     Y = ycrcb[:, :, 0]
#     Y = np.clip(Y + fft_amp * pattern * wm_mask, 0, 255)
#     ycrcb[:, :, 0] = Y
#     img = Image.fromarray(cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB))
#
#     # simpan
#     output_path = os.path.join("data", final_png_name)
#     base_noext = os.path.splitext(output_path)[0]
#     fingerprint_path = base_noext + "_fingerprint.npy"
#     meta_path = base_noext + "_meta.npy"
#
#     np.save(fingerprint_path, noise)
#     meta = {
#         "cx_rel": cx / width_px,
#         "cy_rel": cy / height_px,
#         "r_rel": noise_radius / width_px,
#         "dpi": dpi,
#         "size_px": (width_px, height_px),
#         "size_mm": (6.5, 16.0),
#         "fft": {"amp": float(fft_amp), "fx": int(fft_fx), "fy": int(fft_fy), "masked": True,
#                 "qr_box": [int(x1), int(y1), int(x2), int(y2)], "bar_h": int(bar_h)},
#         "kode_unik": kode_teks,
#         "qr_payload": qr_payload,
#         "output_name": final_png_name,
#         "timestamp": ts,
#     }
#     np.save(meta_path, meta)
#     img.save(output_path, dpi=(dpi, dpi))
#
#     print(f"✅ Label disimpan: {output_path}")
#     print(f"🔑 Kode unik (teks): {kode_teks}")
#     print(f"🔗 QR payload: {qr_payload}")
#     print(f"🧬 Fingerprint: {fingerprint_path}")
#     print(f"📍 Metadata: {meta_path}")
#     print(f"🌊 FFT-WM: amp={fft_amp}, fx={fft_fx}, fy={fft_fy}")
#
#     return {
#         "output_path": output_path,
#         "fingerprint_path": fingerprint_path,
#         "meta_path": meta_path,
#         "output_name": final_png_name,
#         "kode_unik_text": kode_teks,
#         "qr_payload": qr_payload,
#         "width_px": width_px,
#         "height_px": height_px,
#         "dpi": dpi,
#     }


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

    # fingerprint
    noise_radius = max(2, int(round(qr_side * fp_ratio)))
    cx, cy = qr_x + qr_side // 2, qr_y + qr_side // 2
    noise = np.random.randint(0, 256, (2*noise_radius, 2*noise_radius), dtype=np.uint8)
    noise_img = Image.fromarray(np.stack([noise] * 3, axis=2))
    mask_fp = Image.new("L", (2*noise_radius, 2*noise_radius), 0)
    ImageDraw.Draw(mask_fp).ellipse((0, 0, 2*noise_radius, 2*noise_radius), fill=255)
    img.paste(noise_img, (cx-noise_radius, cy-noise_radius), mask_fp)

    # FFT-watermark (exclude QR area)
    wm_mask = np.ones((height_px, width_px), dtype=np.float32)
    margin = 2
    wm_mask[qr_y-margin:qr_y+qr_side+margin, qr_x-margin:qr_x+qr_side+margin] = 0.0
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    pattern = np.sin(2*np.pi*(fft_fx*xx/width_px + fft_fy*yy/height_px)).astype(np.float32)
    arr = np.array(img)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]
    Y = np.clip(Y + fft_amp * pattern * wm_mask, 0, 255)
    ycrcb[:, :, 0] = Y
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
        "fft": {"fx": fft_fx, "fy": fft_fy, "amp": fft_amp}
    })

    img.save(output_path, dpi=(dpi, dpi))

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

def verifikasi_label_fleksibel(
    foto_path: str | None = None,
    output_name: str = "label_ali.png",
    threshold_mse: int = 12000,
    simpan_debug: bool = True,
    fft_snr_threshold: float = 2.0  # ambang deteksi watermark robust
):
    """
    Verifikasi hybrid:
      - Fingerprint (MSE) → autentikasi (fragile)
      - FFT-watermark (SNR) → deteksi watermark robust di background
    Keputusan:
      - ✅ ASLI   : fingerprint cocok (MSE < ambang)
      - ⚠️ RUSAK  : fingerprint TIDAK cocok, tetapi watermark ATAU QR masih OK
      - ❌ SALINAN: fingerprint TIDAK cocok, watermark TIDAK terdeteksi, dan QR TIDAK terbaca
    """
    # ----- Ambil fingerprint & meta -----
    base = os.path.splitext(output_name)[0]
    fingerprint_path = _cari_path(f"{base}_fingerprint.npy")
    meta_path        = _cari_path(f"{base}_meta.npy")
    if not fingerprint_path or not meta_path:
        raise FileNotFoundError("Fingerprint/meta tidak ditemukan di folder data/. Pastikan sudah generate label.")

    fingerprint = np.load(fingerprint_path)  # grayscale (h,w)
    meta = np.load(meta_path, allow_pickle=True).item()
    cx_rel, cy_rel, r_rel = float(meta["cx_rel"]), float(meta["cy_rel"]), float(meta["r_rel"])
    meta_fft = meta.get("fft", None)

    # ----- Tentukan sumber gambar yang diverifikasi -----
    if foto_path is None:
        # mode test: pakai file generate
        sumber = _cari_path(output_name)
        if not sumber:
            raise FileNotFoundError(f"File generate tidak ditemukan: data/{output_name}")
    else:
        sumber = _cari_path(foto_path)
        if not sumber:
            raise FileNotFoundError(f"Foto tidak ditemukan: {foto_path} atau data/{foto_path}")

    img_bgr = cv2.imread(sumber)
    if img_bgr is None:
        raise ValueError("Gagal membaca gambar. Format tidak didukung atau file korup.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ----- Hitung ROI absolut dari rasio relatif -----
    cx = int(round(cx_rel * W))
    cy = int(round(cy_rel * H))
    r  = int(round(r_rel  * W))  # skala horizontal

    r = max(4, min(r, cx, cy, W - cx - 1, H - cy - 1))
    y1, y2 = cy - r, cy + r
    x1, x2 = cx - r, cx + r
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Crop kosong—meta/posisi tidak cocok dengan gambar.")

    crop_resized = cv2.resize(crop, (fingerprint.shape[1], fingerprint.shape[0]), interpolation=cv2.INTER_AREA)

    # ----- MSE autentikasi (fragile fingerprint) -----
    mse = float(np.mean((crop_resized.astype("float32") - fingerprint.astype("float32"))**2))
    fp_ok = bool(mse < threshold_mse)

    # ----- Baca QR (3 cara) -----
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

    # ----- FFT-watermark detection (robust) -----
    dbg_base = os.path.join("data", f"{base}_dbg") if simpan_debug else None
    fft_snr, fft_detail = _fft_snr(
        img_bgr,
        meta_fft=meta_fft,
        fft_snr_radius=3,
        fft_fx=(meta_fft or {}).get("fx", 10),
        fft_fy=(meta_fft or {}).get("fy", 6),
        save_dbg=(dbg_base + "_spec.png") if simpan_debug else None
    )
    wm_ok = bool(fft_snr >= float(fft_snr_threshold))

    # ----- Simpan debug -----
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
        # tambahkan paths FFT
        debug_paths.update(fft_detail.get("debug", {}))

    # ----- Keputusan tiga level + cetak ringkas -----
    if fp_ok and wm_ok:
        status_code = "ASLI"
        status = "✅ ASLI"
        reason = "Fingerprint cocok (MSE di bawah ambang)."
    else:
        if wm_ok or fp_ok or qr_ok:
            status_code = "RUSAK"
            detail = []
            if not wm_ok: detail.append("watermark tidak terdeteksi")
            if not fp_ok: detail.append("fingerprint tidak cocok")
            if not qr_ok: detail.append("QR sulit/tidak terbaca")
            reason = " / ".join(detail) if detail else "Sinyal tidak ideal."
            status = f"⚠️ RUSAK — {reason}"
        else:
            status_code = "SALINAN"
            status = "❌ SALINAN"
            reason = "Fingerprint tidak cocok, watermark tidak terdeteksi, dan QR tidak terbaca."

    print(f"📄 Sumber: {sumber}")
    print(f"📏 MSE = {mse:.2f}  |  Ambang = {threshold_mse}  ->  {'cocok' if fp_ok else 'tidak'}")
    print(f"🌊 FFT SNR ≈ {fft_snr:.2f}  |  Ambang = {fft_snr_threshold}  ->  {'terdeteksi' if wm_ok else 'tidak'}")
    print(f"🔗 QR: {qr_data if qr_ok else '(tidak terbaca)'}")
    print(f"🧾 Status: {status}")

    return {
        "source": sumber,
        "mse": round(mse, 2),
        "threshold_mse": threshold_mse,
        "authentic": fp_ok,                 # tetap expose sebagai "fingerprint ok"
        "fp_ok": fp_ok,
        "fft_snr": round(fft_snr, 2),
        "fft_threshold": float(fft_snr_threshold),
        "watermark_present": wm_ok,
        "wm_ok": wm_ok,
        "qr_data": qr_data,
        "qr_ok": qr_ok,
        "status": status,
        "status_code": status_code,         # 'ASLI' | 'RUSAK' | 'SALINAN'
        "reason": reason,
        "copy_with_watermark": (not fp_ok) and wm_ok,  # kompatibel dengan versi lama
        "methods": {"raw": bool(data_raw), "threshold": bool(data_thr), "crop": bool(data_crop)},
        "roi": {"cx": cx, "cy": cy, "r": r, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "qr_crop": {"X1": X1, "Y1": Y1, "X2": X2, "Y2": Y2}},
        "fft_detail": fft_detail,
        "paths": {"fingerprint": fingerprint_path, "meta": meta_path, "debug": debug_paths},
    }


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
    # hasil = verifikasi_label_fleksibel("data/1757477780-00001.png", "1757477780-00001.png", 12000)
    # print(hasil)
    # b = base64txt_to_file("photo_2025-08-28_13-56-32_base64.txt", "output.png")
    # print(b)
    a = buat_label_keaslian()
    print(a)
    # buat_label_keaslian_b64()
