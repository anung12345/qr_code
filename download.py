# downloader.py
import os
import requests
from requests.auth import HTTPBasicAuth

API_BASE = "http://10.74.18.162:8075/db/copyproof/sticker"
API_USER = "testerzero"
API_PASS = "lessthanone"
TIMEOUT  = 30

auth = HTTPBasicAuth(API_USER, API_PASS)

def _download_file(id_sticker: str, field: str, output_path: str) -> str | None:
    """
    Download file binary dari API berdasarkan field.
    field:
      - "sticker_copyproof_tag"   (PNG)
      - "sticker_qr_fingerprint" (NPY)
      - "sticker_qr_meta"        (NPY)
    """
    url = f"{API_BASE}/{id_sticker}/@download/{field}"
    try:
        resp = requests.get(url, auth=auth, timeout=TIMEOUT, stream=True)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            print(f"✅ Downloaded {field} -> {output_path}")
            return output_path
        else:
            print(f"⚠️ Gagal download {field}, status {resp.status_code}")
            print(resp.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error download {field}: {e}")
        return None


def download_png(id_sticker: str, output_dir: str = "data") -> str | None:
    os.makedirs(output_dir, exist_ok=True)
    return _download_file(id_sticker, "sticker_copyproof_tag", os.path.join(output_dir, f"{id_sticker}.png"))

def download_fingerprint(id_sticker: str, output_dir: str = "data") -> str | None:
    os.makedirs(output_dir, exist_ok=True)
    return _download_file(id_sticker, "sticker_qr_fingerprint", os.path.join(output_dir, f"{id_sticker}_fingerprint.npy"))

def download_metadata(id_sticker: str, output_dir: str = "data") -> str | None:
    os.makedirs(output_dir, exist_ok=True)
    return _download_file(id_sticker, "sticker_qr_meta", os.path.join(output_dir, f"{id_sticker}_meta.npy"))


if __name__ == "__main__":
    sid = "STR20250822134325502"  # contoh id_sticker
    download_png(sid)
    download_fingerprint(sid)
    download_metadata(sid)
