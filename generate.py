# run_sticker.py
import json
import requests
from requests.auth import HTTPBasicAuth
from ori import buat_label_keaslian  # gunakan fungsi generator milikmu

API_BASE = "http://10.74.18.162:8075/db/copyproof/sticker"
API_USER = "testerzero"
API_PASS = "lessthanone"
TIMEOUT  = 20  # detik

auth = HTTPBasicAuth(API_USER, API_PASS)

def post_create_sticker(counter: int):
    """POST Sticker baru & kembalikan (context_generate, response)."""
    gen = buat_label_keaslian(counter=counter)  # generate & simpan file
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

def _patch_upload_binary(id_sticker: str, field: str, file_path: str) -> requests.Response:
    """
    Upload biner ke field upload tertentu:
      field: "sticker_copyproof_tag" | "sticker_qr_fingerprint" | "sticker_qr_meta"
    """
    url = f"{API_BASE}/{id_sticker}/@upload/{field}"
    headers = {"Content-Type": "application/octet-stream"}
    with open(file_path, "rb") as f:
        resp = requests.patch(url, data=f, headers=headers, auth=auth, timeout=TIMEOUT)
    return resp

def main(counter: int, do_patch: bool = True):
    # 1) POST buat sticker
    try:
        ctx, resp = post_create_sticker(counter)
    except requests.RequestException as e:
        print(f"❌ Gagal POST: {e}")
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
        return None

    print(f"\n✅ Extracted @name: {id_sticker}")

    if not do_patch:
        return id_sticker

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

    return id_sticker


if __name__ == "__main__":
    sid = main(counter=1, do_patch=True)
    if sid:
        print(f"\n🎯 Sticker ID final: {sid}")
    else:
        print("\n❌ Gagal membuat/mengupload sticker.")
