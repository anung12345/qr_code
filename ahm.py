from PIL import Image, ImageDraw, ImageFont
import qrcode
import numpy as np
import os
import cv2  # untuk edit luminance (Y channel) saat sisipkan FFT watermark

# --- Util font ---
def _load_font(size_px: int):
    try:
        return ImageFont.truetype("arial.ttf", size_px)
    except:
        return ImageFont.load_default()

def _font_fit(draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int,
              min_size: int = 4, max_size: int = 300):
    lo, hi = min_size, max_size
    best = _load_font(min_size)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = _load_font(mid)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        w, h = r - l, b - t
        if w <= max_w and h <= max_h:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def buat_label_keaslian(
    output_name="label_ali.png",
    dpi=1200,
    *,
    # parameter FFT-watermark (robust)
    fft_amp: float = 1.5,   # 0.8–2.5 disarankan; makin besar makin terlihat di FFT
    fft_fx: int   = 10,     # frekuensi horizontal (grid sinus)
    fft_fy: int   = 6       # frekuensi vertikal
):
    """
    Label 6.5mm x 16mm @ 1200dpi:
      - watermark teks 'ALI' halus (visual)
      - bar hitam atas/bawah + teks
      - QR ke https://www.google.com
      - fingerprint: lingkaran noise acak <=10% sisi QR, di tengah QR (fragile)
      - FFT-watermark: pola sinus di background putih (robust), tidak menimpa area QR & bar
    Simpan ke data/: PNG + *_fingerprint.npy + *_meta.npy
    """
    os.makedirs("data", exist_ok=True)

    # Konversi mm -> px
    mm_to_px = lambda mm: int(round((mm / 25.4) * dpi))
    width_px, height_px = mm_to_px(6.5), mm_to_px(16.0)  # width x height

    # Kanvas
    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)

    # --- Watermark teks 'ALI' halus (visual) ---
    spacing = max(mm_to_px(2.0), 6)
    wm_font = _load_font(max(mm_to_px(0.9), 6))
    for y in range(0, height_px, spacing):
        for x in range(0, width_px, spacing):
            draw.text((x, y), "ALI", fill=(205, 205, 205), font=wm_font)

    # --- Bar hitam atas & bawah ---
    bar_h = max(mm_to_px(1.1), int(height_px * 0.08))  # ~8% tinggi, min 1.1mm
    draw.rectangle([0, 0, width_px, bar_h], fill="black")
    draw.rectangle([0, height_px - bar_h, width_px, height_px], fill="black")

    # Teks bar auto-fit
    top_text = "JAMINAN KEASLIAN PRODUK"
    bottom_text = "ALI"
    pad = max(1, bar_h // 8)
    font_top = _font_fit(draw, top_text, max_w=width_px - 2, max_h=bar_h - 2 * pad)
    font_bot = _font_fit(draw, bottom_text, max_w=width_px - 2, max_h=bar_h - 2 * pad)
    draw.text((width_px // 2, bar_h // 2), top_text, fill="white", anchor="mm", font=font_top)
    draw.text((width_px // 2, height_px - bar_h // 2), bottom_text, fill="white", anchor="mm", font=font_bot)

    # --- QR code ---
    tautan = "https://www.google.com"
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(tautan)
    qr.make(fit=True)
    inner_h = height_px - 2 * bar_h
    max_qr_side = int(min(width_px * 0.88, inner_h * 0.88))  # batasi oleh lebar & area putih
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize(
        (max_qr_side, max_qr_side)
    )
    qr_x = (width_px - max_qr_side) // 2
    qr_y = bar_h + (inner_h - max_qr_side) // 2
    img.paste(img_qr, (qr_x, qr_y))

    # --- Fingerprint: lingkaran noise acak (fragile), <=10% sisi QR ---
    noise_radius = max(2, max_qr_side // 10)  # ≈10% sisi QR
    cx, cy = qr_x + max_qr_side // 2, qr_y + max_qr_side // 2
    noise = np.random.randint(0, 256, (2 * noise_radius, 2 * noise_radius), dtype=np.uint8)
    noise_img = Image.fromarray(np.stack([noise] * 3, axis=2))  # RGB dari grayscale
    mask_fp = Image.new("L", (2 * noise_radius, 2 * noise_radius), 0)
    ImageDraw.Draw(mask_fp).ellipse((0, 0, 2 * noise_radius, 2 * noise_radius), fill=255)
    img.paste(noise_img, (cx - noise_radius, cy - noise_radius), mask_fp)

    # === FFT-watermark (robust) di background putih, TIDAK menimpa QR & bar ===
    # Siapkan mask latar putih (1=apply watermark, 0=tidak)
    wm_mask = np.ones((height_px, width_px), dtype=np.float32)

    # Nolkan area bar atas & bawah
    wm_mask[:bar_h, :] = 0.0
    wm_mask[height_px - bar_h:, :] = 0.0

    # Nolkan area QR (tambahkan margin 1–2 px agar aman)
    margin = 2
    x1 = max(0, qr_x - margin); x2 = min(width_px, qr_x + max_qr_side + margin)
    y1 = max(0, qr_y - margin); y2 = min(height_px, qr_y + max_qr_side + margin)
    wm_mask[y1:y2, x1:x2] = 0.0

    # Buat pola sinus 2D
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    pattern = np.sin(2*np.pi*(fft_fx*xx/width_px + fft_fy*yy/height_px)).astype(np.float32)

    # Terapkan pola pada channel Y (luminance) dengan masker
    arr = np.array(img)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]
    Y = np.clip(Y + fft_amp * pattern * wm_mask, 0, 255)
    ycrcb[:, :, 0] = Y
    arr2 = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    img = Image.fromarray(arr2)  # update img setelah watermark robust

    # --- Path output & simpan ---
    output_path = os.path.join("data", output_name)
    fingerprint_path = os.path.splitext(output_path)[0] + "_fingerprint.npy"
    meta_path = os.path.splitext(output_path)[0] + "_meta.npy"

    # Fingerprint & meta (tambahkan info FFT)
    np.save(fingerprint_path, noise)
    meta = {
        "cx_rel": cx / width_px,
        "cy_rel": cy / height_px,
        "r_rel": noise_radius / width_px,
        "dpi": dpi,
        "size_px": (width_px, height_px),
        "size_mm": (6.5, 16.0),
        "fft": {"amp": float(fft_amp), "fx": int(fft_fx), "fy": int(fft_fy), "masked": True,
                "qr_box": [int(x1), int(y1), int(x2), int(y2)], "bar_h": int(bar_h)}
    }
    np.save(meta_path, meta)

    img.save(output_path, dpi=(dpi, dpi))

    # --- Log ---
    print(f"✅ Label disimpan: {output_path}")
    print(f"🔗 QR link: {tautan}")
    print(f"🧬 Fingerprint: {fingerprint_path}")
    print(f"📍 Metadata: {meta_path}")
    print(f"🌊 FFT-WM: amp={fft_amp}, fx={fft_fx}, fy={fft_fy} (exclude QR & bars)")

# --- contoh pakai ---
# buat_label_keaslian(output_name="label_ali.png", dpi=1200)

if __name__ == "__main__":
    buat_label_keaslian()
