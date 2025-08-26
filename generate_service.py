# app.py
from flask import Flask, request, jsonify
import os, base64, traceback
import cv2, numpy as np
from tempfile import TemporaryDirectory

# ==== IMPORT MODUL LOKAL ====
# Generator + POST/PATCH ke API tujuan (harus menyediakan main(counter, do_patch=True, cleanup=True))
from generate import main as run_sticker_main

# Verifier
from ori import verifikasi_label_fleksibel

# Downloader (pakai modul yang kamu sebutkan "download.py")
from download import download_png, download_fingerprint, download_metadata
# jika modulmu bernama downloader.py, ganti baris di atas menjadi:
# from downloader import download_png, download_fingerprint, download_metadata

# ==== FLASK APP ====
app = Flask(__name__)
app.url_map.strict_slashes = False


# ---------- util ----------
def _bool_like(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


# ---------- health & routes ----------
@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/__routes")
def routes():
    return {"routes": [str(r) for r in app.url_map.iter_rules()]}, 200


# ---------- ENDPOINT 1: Trigger generate + POST + PATCH ----------
@app.post("/sticker/run")
def run_sticker_service():
    """
    Body JSON / form:
      - counter  (wajib, int)
      - do_patch (opsional, bool; default true)
      - cleanup  (opsional, bool; default true)

    Response: { ok, sticker_id?, do_patch, cleanup, error? }
    """
    try:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            counter  = payload.get("counter", None)
            do_patch = _bool_like(payload.get("do_patch"), True)
            cleanup  = _bool_like(payload.get("cleanup"), True)
        else:
            counter  = request.values.get("counter", None)
            do_patch = _bool_like(request.values.get("do_patch"), True)
            cleanup  = _bool_like(request.values.get("cleanup"), True)

        if counter is None:
            return jsonify(ok=False, error="Parameter 'counter' wajib."), 400
        try:
            counter = int(counter)
        except ValueError:
            return jsonify(ok=False, error="'counter' harus integer."), 400

        sticker_id = run_sticker_main(counter=counter, do_patch=do_patch, cleanup=cleanup)
        if not sticker_id:
            return jsonify(ok=False, error="Gagal membuat / mengunggah sticker ke API tujuan."), 502

        return jsonify(ok=True, sticker_id=sticker_id, do_patch=bool(do_patch), cleanup=bool(cleanup)), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=f"internal error: {e}"), 500


# ---------- ENDPOINT 2: Verifikasi dari base64 + sticker_id ----------
@app.post("/verify_files")
def verify_files():
    """
    Body JSON:
    {
      "raw_photo": "<base64 string>",
      "sticker_id": "STRxxxx"
    }
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify(error="JSON body tidak valid"), 400

    b64_str    = str(payload.get("raw_photo", "")).strip()
    sticker_id = str(payload.get("sticker_id", "")).strip()
    if not b64_str or not sticker_id:
        return jsonify(error="Harus ada raw_photo (base64) dan sticker_id"), 400

    # Semua kerja di folder sementara -> DIHAPUS TOTAL otomatis setelah keluar dari blok 'with'
    with TemporaryDirectory(prefix="verify_", dir=".") as workdir:
        # 1) simpan foto dari base64
        try:
            img_bytes = base64.b64decode(b64_str, validate=False)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify(error="raw_photo bukan PNG/JPG valid"), 400
            foto_path = os.path.join(workdir, "upload.png")
            cv2.imwrite(foto_path, img)
        except Exception as e:
            return jsonify(error=f"Gagal decode raw_photo: {e}"), 400

        # 2) download referensi ke folder sementara yang sama
        png_ref = download_png(sticker_id,  output_dir=workdir)
        fp_ref  = download_fingerprint(sticker_id, output_dir=workdir)
        meta_ref= download_metadata(sticker_id, output_dir=workdir)
        if not (png_ref and fp_ref and meta_ref):
            return jsonify(error="Gagal download file referensi dari API eksternal"), 502

        # 3) verifikasi
        try:
            result = verifikasi_label_fleksibel(
                foto_path=foto_path,
                output_name=png_ref,        # path PNG referensi (di workdir)
                threshold_mse=12000,
                fft_snr_threshold=2.0,
                simpan_debug=False
            )
        except Exception as e:
            return jsonify(error=f"Gagal verifikasi: {e}"), 500

        # keluar from TemporaryDirectory -> workdir dibersihkan total
        return jsonify(result), 200


# ---------- runner ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9800"))
    app.run(host="0.0.0.0", port=port, threaded=True)
