# app.py
from flask import Flask, request, jsonify, send_from_directory
import os

# Pastikan ori.py mengekspor tiga fungsi ini:
# - buat_label_keaslian_b64(kode_unik: str, output_name="label_ali.png", dpi=1200, simpan_b64=True)
# - verifikasi_label_fleksibel(...)
# - base64txt_to_file(txt_path, output_path)
from ori import (
    buat_label_keaslian_b64,
    verifikasi_label_fleksibel,
    base64txt_to_file,
)

app = Flask(__name__)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


@app.get("/health")
def health():
    return jsonify(status="ok"), 200


# ---------- GENERATE ----------
@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "GET":
        return (
            """
            <h3>Generate Label</h3>
            <form method="POST">
              Output name: <input name="output_name" value="label_ali.png"><br/>
              DPI: <input name="dpi" value="1200"><br/>
              Kode Unik: <input name="kode_unik" value="ABC12345"><br/>
              Return Base64? <select name="return_base64">
                <option value="1">Ya</option>
                <option value="0" selected>Tidak</option>
              </select><br/><br/>
              <button type="submit">Generate</button>
            </form>
            """,
            200,
            {"Content-Type": "text/html"}
        )

    output_name = (request.form.get("output_name") or "label_ali.png").strip()
    dpi_s = (request.form.get("dpi") or "1200").strip()
    kode_unik = (request.form.get("kode_unik") or "").strip()
    return_base64 = (request.form.get("return_base64") or "0").strip() == "1"

    try:
        dpi = int(dpi_s)
    except ValueError:
        return jsonify(error="dpi harus integer"), 400

    try:
        result = buat_label_keaslian_b64(
            kode_unik=kode_unik,
            output_name=output_name,
            dpi=dpi,
            simpan_b64=True
        )
        payload = {
            "output_path": result["output_path"],
            "base64_path": result["base64_path"],
            "qr_link": "https://www.google.com",
            "kode_unik": kode_unik
        }
        if return_base64:
            payload["base64"] = result["base64"]

        return jsonify(payload), 200
    except Exception as e:
        return jsonify(error=f"internal error: {str(e)}"), 500


# ---------- VERIFY ----------
@app.route("/verify", methods=["GET", "POST"])
def verify():
    """
    GET  : form HTML
    POST : verifikasi label hybrid
      form-data:
        - file (opsional): foto label; jika kosong -> pakai data/<output_name>
        - output_name (default 'label_ali.png')
        - threshold (default 12000)                  -> MSE fingerprint
        - fft_snr_threshold (default 2.0)            -> SNR watermark robust
        - debug: "1" untuk simpan debug (crop/thr/fft)
    """
    if request.method == "GET":
        return (
            """
            <h3>Verify Label</h3>
            <form method="POST" enctype="multipart/form-data">
              <div>Foto (opsional): <input type="file" name="file" /></div>
              <div>Output name: <input name="output_name" value="label_ali.png" /></div>
              <div>Threshold MSE: <input name="threshold" value="12000" /></div>
              <div>FFT SNR threshold: <input name="fft_snr_threshold" value="2.0" /></div>
              <div>Save debug?
                <select name="debug">
                  <option value="1">Ya</option>
                  <option value="0" selected>Tidak</option>
                </select>
              </div>
              <br/>
              <button type="submit">Verify</button>
            </form>
            """,
            200,
            {"Content-Type": "text/html"},
        )

    # POST
    output_name = (request.form.get("output_name") or "label_ali.png").strip()
    threshold_s = (request.form.get("threshold") or "").strip()
    fft_thr_s = (request.form.get("fft_snr_threshold") or "").strip()
    debug_on = (request.form.get("debug") or "0").strip() == "1"

    try:
        threshold_mse = int(threshold_s) if threshold_s else 12000
    except ValueError:
        return jsonify(error="threshold harus integer"), 400

    try:
        fft_snr_threshold = float(fft_thr_s) if fft_thr_s else 2.0
    except ValueError:
        return jsonify(error="fft_snr_threshold harus numerik"), 400

    # baca file (opsional)
    foto_path = None
    if "file" in request.files and request.files["file"].filename:
        os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
        up = request.files["file"]
        save_path = os.path.join(DATA_DIR, "uploads", up.filename)
        up.save(save_path)
        foto_path = save_path

    try:
        result = verifikasi_label_fleksibel(
            foto_path=foto_path,             # None => pakai data/<output_name>
            output_name=output_name,
            threshold_mse=threshold_mse,
            simpan_debug=debug_on,
            fft_snr_threshold=fft_snr_threshold,
        )
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"internal error: {str(e)}"), 500


# ---------- DECODE Base64 TXT -> File ----------
@app.route("/decode", methods=["GET", "POST"])
def decode():
    """
    GET  : form HTML
    POST : form-data
       - txt_path: path ke file .txt berisi base64 (mis. data/label_ali_base64.txt)
       - output_path: path output file (mis. data/decoded.png)
    """
    if request.method == "GET":
        return (
            """
            <h3>Decode Base64 TXT</h3>
            <form method="POST">
              <div>TXT path: <input name="txt_path" value="data/label_ali_base64.txt" /></div>
              <div>Output path: <input name="output_path" value="data/decoded.png" /></div>
              <br/>
              <button type="submit">Decode</button>
            </form>
            """,
            200,
            {"Content-Type": "text/html"},
        )

    txt_path = (request.form.get("txt_path") or "").strip()
    output_path = (request.form.get("output_path") or "").strip()
    if not txt_path or not output_path:
        return jsonify(error="txt_path dan output_path wajib diisi"), 400

    try:
        out = base64txt_to_file(txt_path, output_path)
        return jsonify(decoded_path=out, preview_url=f"/file/{os.path.basename(out)}"), 200
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    except Exception as e:
        return jsonify(error=f"internal error: {str(e)}"), 500


# ---------- Serve file dari folder data ----------
@app.get("/file/<path:fname>")
def serve_file(fname: str):
    """Akses cepat file hasil di folder data/"""
    return send_from_directory(DATA_DIR, fname, as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9700"))
    app.run(host="0.0.0.0", port=port, threaded=True)
