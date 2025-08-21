# service_runner.py
from flask import Flask, request, jsonify
import os
import traceback

# impor fungsi main(counter, do_patch=True, cleanup=True) dari skrip-mu
from generate import main as run_sticker_main

app = Flask(__name__)
app.url_map.strict_slashes = False

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/__routes")
def routes():
    return {"routes": [str(r) for r in app.url_map.iter_rules()]}


@app.post("/sticker/run")
def run_sticker_service():
    """
    Terima permintaan untuk membuat sticker (POST ke API tujuan) + upload file.
    Body bisa JSON atau form-data/x-www-form-urlencoded.

    Parameter:
      - counter  (wajib, int)
      - do_patch (opsional, bool; default: true)
      - cleanup  (opsional, bool; default: true)

    Response: { "sticker_id": "...", "ok": true, "detail": "...optional..." }
    """
    try:
        # dukung JSON & form
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            counter  = payload.get("counter", None)
            do_patch = payload.get("do_patch", True)
            cleanup  = payload.get("cleanup", True)
        else:
            counter  = request.values.get("counter", None)
            do_patch = request.values.get("do_patch", "true").lower() in ("1","true","yes","y","on")
            cleanup  = request.values.get("cleanup", "true").lower() in ("1","true","yes","y","on")

        # validasi counter
        if counter is None:
            return jsonify(error="Parameter 'counter' wajib."), 400
        try:
            counter = int(counter)
        except ValueError:
            return jsonify(error="'counter' harus integer."), 400

        # jalankan pipeline: generate -> POST -> PATCH (png/fp/meta) -> cleanup (opsional)
        sticker_id = run_sticker_main(counter=counter, do_patch=do_patch, cleanup=cleanup)

        if not sticker_id:
            # main() sudah cetak log kesalahan; kita kembalikan 502 agar client tahu gagal proses
            return jsonify(ok=False, error="Gagal membuat / mengunggah sticker ke API tujuan."), 502

        return jsonify(ok=True, sticker_id=sticker_id, do_patch=bool(do_patch), cleanup=bool(cleanup)), 200

    except Exception as e:
        # log jejak agar mudah debug di server
        traceback.print_exc()
        return jsonify(ok=False, error=f"internal error: {e}"), 500


if __name__ == "__main__":
    # port default 9800 biar tidak bentrok dengan service lain
    port = int(os.environ.get("PORT", "9800"))
    app.run(host="0.0.0.0", port=port, threaded=True)
