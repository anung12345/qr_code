# app.py
from flask import Flask, request, jsonify
import os, base64, traceback, shutil
import cv2, numpy as np
from tempfile import TemporaryDirectory

# ==== IMPORT MODUL LOKAL ====
from generate import main as run_sticker_main
from ori import verifikasi_label_fleksibel
from download import download_png, download_fingerprint, download_metadata

# QR lookup & helper untuk kode_unik + validate PATCH
from qr_lookup import get_qr_and_sticker_id_from_base64, qr_to_kode_unik
from validate_patch import patch_validatecp

app = Flask(__name__)
app.url_map.strict_slashes = False


def _bool_like(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/__routes")
def routes():
    return {"routes": [str(r) for r in app.url_map.iter_rules()]}, 200


# ---------- ENDPOINT 1: Trigger generate + POST + PATCH ----------
@app.post("/sticker/run")
def run_sticker_service():
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


# ---------- ENDPOINT 2: Verifikasi (RAW base64 → QR → sticker_id) + PATCH validatecp ----------
@app.post("/verify_files")
def verify_files():
    """
    Body JSON:
    {
      "raw_photo": "<base64 string>",
      "idvalicp": "VALIxxxx",
      "threshold": 12000,           # optional
      "fft_snr_threshold": 2.0,     # optional
      "debug": 0|1                  # optional
    }
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify(error="JSON body tidak valid"), 400

    b64_str    = str(payload.get("raw_photo", "")).strip()
    idvalicp   = str(payload.get("idvalicp", "")).strip()
    thr        = payload.get("threshold", 12000)
    fft_thr    = payload.get("fft_snr_threshold", 2.0)
    debug_on   = _bool_like(payload.get("debug"), False)

    if not b64_str or not idvalicp:
        return jsonify(error="Harus ada 'raw_photo' (base64) dan 'idvalicp'."), 400

    # 1) Baca QR & dapatkan sticker_id
    try:
        lookup = get_qr_and_sticker_id_from_base64(b64_str)
    except Exception as e:
        return jsonify(error=f"Gagal proses QR/list: {e}"), 500

    if not lookup.get("ok"):
        # QR tidak terbaca pada tahap lookup → PATCH status "4"
        try:
            resp_patch = patch_validatecp(
                idvalicp=idvalicp,
                kode_unik_qr="tidak terbaca",
                qr_data="tidak terbaca",
                status_text="4",
                location="JAKARTA001",
                barcode_item="TESTITEMS",
                part_name="TEST AUTO PART",
            )
            patch_info = {
                "status_code": resp_patch.status_code,
                "body": (resp_patch.json() if resp_patch.headers.get("Content-Type", "").startswith("application/json")
                         else resp_patch.text[:500])
            }
        except Exception as e:
            patch_info = {"error": f"Gagal PATCH validatecp: {e}"}

        return jsonify({
            "ok": False,
            "error": "QR/list error",
            "detail": lookup,
            "validate_patch": patch_info
        }), 200

    qr_payload  = lookup["qr_payload"]      # contoh: STICKERTEST001-1755845004-00001
    sticker_id  = lookup["sticker_id"]      # contoh: STR20...
    try:
        kode_unik_qr = qr_to_kode_unik(qr_payload)  # AHM-PM-<sisa>
    except Exception as e:
        return jsonify(error=f"Gagal bentuk kode_unik dari qr_data: {e}"), 400

    # 2) Semua kerja file di folder sementara, termasuk download referensi
    with TemporaryDirectory(prefix="verify_", dir=".") as workdir:
        # simpan foto base64 -> png
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

        # download referensi berdasarkan sticker_id
        png_ref = download_png(sticker_id,  output_dir=workdir)
        fp_ref  = download_fingerprint(sticker_id, output_dir=workdir)
        meta_ref= download_metadata(sticker_id, output_dir=workdir)
        if not (png_ref and fp_ref and meta_ref):
            return jsonify(error="Gagal download file referensi dari API eksternal"), 502

        # 3) verifikasi
        try:
            result = verifikasi_label_fleksibel(
                foto_path=foto_path,
                output_name=png_ref,
                threshold_mse=int(thr),
                fft_snr_threshold=float(fft_thr),
                simpan_debug=bool(debug_on)
            )
        except Exception as e:
            return jsonify(error=f"Gagal verifikasi: {e}"), 500

        # 4) petakan status → {ASLI|RUSAK|SALINAN}
        status_text = result["status_code"].upper()
        if status_text not in ("ASLI", "RUSAK", "SALINAN"):
            status_text = "SALINAN"

        # 5) PATCH ke validatecp (normal flow)
        try:
            resp_patch = patch_validatecp(
                idvalicp=idvalicp,
                kode_unik_qr=kode_unik_qr,
                qr_data=qr_payload,
                status_text=status_text,
                location="JAKARTA001",
                barcode_item="TESTITEMS",
                part_name="TEST AUTO PART",
            )
            patch_info = {"status_code": resp_patch.status_code}
            try:
                patch_info["body"] = resp_patch.json()
            except Exception:
                patch_info["body"] = resp_patch.text[:500]
        except Exception as e:
            patch_info = {"error": f"Gagal PATCH validatecp: {e}"}

        # 6) bersihkan folder data jika lib verifikasi membuatnya
        try:
            shutil.rmtree("data", ignore_errors=True)
        except Exception:
            pass

        # respon akhir
        return jsonify({
            "ok": True,
            "qr_payload": qr_payload,
            "sticker_id": sticker_id,
            "kode_unik": kode_unik_qr,
            "verify_result": result,
            "status_text": status_text,
            "validate_patch": patch_info,
        }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9800"))
    app.run(host="0.0.0.0", port=port, threaded=True)
