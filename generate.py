# run_sticker.py
import json
import requests
from requests.auth import HTTPBasicAuth
import os
import shutil
from ori import buat_label_keaslian  # gunakan fungsi generator milikmu

API_BASE = "http://10.84.136.64:8075/db/copyproof/sticker"
API_USER = "testerzero"
API_PASS = "lessthanone"
TIMEOUT  = 20  # detik

auth = HTTPBasicAuth(API_USER, API_PASS)

def post_create_sticker(counter: int):
    """POST Sticker baru & kembalikan (context_generate, response)."""
    gen = buat_label_keaslian(counter=counter)  # generate & simpan file di ./data
    payload = {
        "@type": "Sticker",
        "sticker_uniq_code": gen["kode_unik_text"],
        "sticker_qr_code":   gen["qr_payload"],
        "sticker_status":    0,
        "sticker_counter":   int(counter),
    }
    resp = requests.post(API_BASE, json=payload, auth=auth, timeout=TIMEOUT)
    return {"gen": gen, "payload": payload}, resp

def extract_id_from_response(resp: requests.Response) -> str | None:
    try:
        data = resp.json()
        return data.get("@name")
    except Exception:
        return None

def guess_content_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".png":
        return "image/png"
    elif ext == ".npy":
        return "application/octet-stream"
    return "application/octet-stream"

def _patch_upload_binary(id_sticker: str, field: str, file_path: str) -> requests.Response:
    """
    Upload biner ke field upload tertentu:
      field: "sticker_copyproof_tag" | "sticker_qr_fingerprint" | "sticker_qr_meta"
    """
    url = f"{API_BASE}/{id_sticker}/@upload/{field}"
    headers = {"Content-Type": guess_content_type(file_path)}
    with open(file_path, "rb") as f:
        resp = requests.patch(url, data=f, headers=headers, auth=auth, timeout=TIMEOUT)
    return resp

def cleanup_data_dir(base_dir: str = "data") -> bool:
    """
    Hapus folder `data/` beserta isinya supaya tidak menumpuk.
    Keamanan:
      - hanya menghapus jika nama folder persis 'data'
      - dan folder itu berada di working directory saat ini
    Return True jika berhasil / tidak ada, False jika gagal.
    """
    try:
        if not os.path.exists(base_dir):
            return True
        # Pastikan yang dihapus memang ./data (bukan path lain)
        abs_base = os.path.abspath(base_dir)
        abs_cwd  = os.path.abspath(os.getcwd())
        # valid jika parent-nya cwd dan basename-nya 'data'
        if os.path.dirname(abs_base) != abs_cwd or os.path.basename(abs_base) != "data":
            print(f"⚠️ Skip cleanup: target bukan ./data -> {abs_base}")
            return False
        shutil.rmtree(abs_base, ignore_errors=False)
        print("🧹 Folder ./data berhasil dibersihkan.")
        return True
    except Exception as e:
        print(f"⚠️ Gagal cleanup ./data: {e}")
        return False

def main(counter: int, do_patch: bool = True, cleanup: bool = True):
    # 1) POST buat sticker
    try:
        ctx, resp = post_create_sticker(counter)
    except requests.RequestException as e:
        print(f"❌ Gagal POST: {e}")
        if cleanup:
            cleanup_data_dir()
        return None

    gen = ctx["gen"]
    payload = ctx["payload"]

    print("=== Payload (POST) ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n=== Response (POST) [{resp.status_code}] ===")
    try:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)

    id_sticker = extract_id_from_response(resp)
    if not id_sticker:
        print("⚠️ Tidak ada '@name' dalam response POST.")
        if cleanup:
            cleanup_data_dir()
        return None

    print(f"\n✅ Extracted @name: {id_sticker}")

    if do_patch:
        # 2) PATCH upload file PNG (sticker_copyproof_tag)
        file_png = gen["output_path"]
        print(f"\n🚀 PATCH label PNG -> sticker_copyproof_tag: {file_png}")
        try:
            resp_png = _patch_upload_binary(id_sticker, "sticker_copyproof_tag", file_png)
            print(f"Response (PNG) [{resp_png.status_code}]")
            try:
                print(json.dumps(resp_png.json(), indent=2, ensure_ascii=False))
            except Exception:
                print(resp_png.text)
        except requests.RequestException as e:
            print(f"❌ Gagal PATCH PNG: {e}")

        # 3) PATCH upload fingerprint (.npy) -> sticker_qr_fingerprint
        file_fp = gen["fingerprint_path"]
        print(f"\n🚀 PATCH fingerprint NPY -> sticker_qr_fingerprint: {file_fp}")
        try:
            resp_fp = _patch_upload_binary(id_sticker, "sticker_qr_fingerprint", file_fp)
            print(f"Response (FP) [{resp_fp.status_code}]")
            try:
                print(json.dumps(resp_fp.json(), indent=2, ensure_ascii=False))
            except Exception:
                print(resp_fp.text)
        except requests.RequestException as e:
            print(f"❌ Gagal PATCH fingerprint: {e}")

        # 4) PATCH upload metadata (.npy) -> sticker_qr_meta
        file_meta = gen["meta_path"]
        print(f"\n🚀 PATCH metadata NPY -> sticker_qr_meta: {file_meta}")
        try:
            resp_meta = _patch_upload_binary(id_sticker, "sticker_qr_meta", file_meta)
            print(f"Response (META) [{resp_meta.status_code}]")
            try:
                print(json.dumps(resp_meta.json(), indent=2, ensure_ascii=False))
            except Exception:
                print(resp_meta.text)
        except requests.RequestException as e:
            print(f"❌ Gagal PATCH metadata: {e}")

    # 5) Cleanup folder data
    if cleanup and id_sticker:
        cleanup_data_dir()

    return id_sticker


if __name__ == "__main__":
    sid = main(counter=1, do_patch=True, cleanup=True)
    if sid:
        print(f"\n🎯 Sticker ID final: {sid}")
    else:
        print("\n❌ Gagal membuat/mengupload sticker.")
