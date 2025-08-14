# app.py
from flask import Flask, request, jsonify
import os
from ori import verifikasi_label_fleksibel, buat_label_keaslian_b64

app = Flask(__name__)

@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "GET":
        return (
            """
            <h3>Generate Label</h3>
            <form method="POST">
              Output name: <input name="output_name" value="label_ali.png"><br/>
              DPI: <input name="dpi" value="1200"><br/>
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
    return_base64 = (request.form.get("return_base64") or "0").strip() == "1"

    try:
        dpi = int(dpi_s)
    except ValueError:
        return jsonify(error="dpi harus integer"), 400

    try:
        result = buat_label_keaslian_b64(output_name=output_name, dpi=dpi, simpan_b64=True)
        payload = {
            "output_path": result["output_path"],
            "base64_path": result["base64_path"],
            "qr_link": "https://www.google.com"
        }
        if return_base64:
            payload["base64"] = result["base64"]

        return jsonify(payload), 200
    except Exception as e:
        return jsonify(error=f"internal error: {str(e)}"), 500

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.post("/verify")
def verify():
    """
    Multipart form-data:
      - file: (opsional) foto label (jpg/png); jika tidak disertakan -> mode test pakai file generate
      - output_name: (opsional, default "label_ali.png") nama file label saat generate (untuk cari fingerprint/meta di data/)
      - threshold: (opsional, default 12000) ambang MSE
      - debug: (opsional, "1" untuk simpan debug crop/thr)
    """
    output_name = request.form.get("output_name", "label_ali.png").strip()
    threshold = request.form.get("threshold", "").strip()
    debug_on = request.form.get("debug", "").strip() == "1"

    # threshold default
    try:
        threshold_mse = int(threshold) if threshold else 12000
    except ValueError:
        return jsonify(error="threshold harus integer"), 400

    # baca file (opsional)
    foto_path = None
    if "file" in request.files and request.files["file"].filename:
        os.makedirs("data/uploads", exist_ok=True)
        up = request.files["file"]
        save_path = os.path.join("data", "uploads", up.filename)
        up.save(save_path)
        foto_path = save_path

    try:
        result = verifikasi_label_fleksibel(
            foto_path=foto_path,             # None => pakai data/<output_name>
            output_name=output_name,
            threshold_mse=threshold_mse,
            simpan_debug=debug_on
        )
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"internal error: {str(e)}"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
