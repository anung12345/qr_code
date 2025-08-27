# validate_patch.py
import requests, json
from requests.auth import HTTPBasicAuth

VALIDATE_API_BASE = "http://10.74.18.162:8075/db/copyproof/validatecp"
API_USER = "testerzero"
API_PASS = "lessthanone"
TIMEOUT   = 30

_auth = HTTPBasicAuth(API_USER, API_PASS)

STATUS_MAP = {"ASLI": 1, "RUSAK": 2, "SALINAN": 3}

def patch_validatecp(
    idvalicp: str,
    kode_unik_qr: str,
    qr_data: str,
    status_text: str,
    location: str = "JAKARTA001",
    barcode_item: str = "TESTITEMS",
    part_name: str = "TEST AUTO PART",
):
    url = f"{VALIDATE_API_BASE}/{idvalicp}"
    status_map = {"ASLI": 1, "RUSAK": 2, "SALINAN": 3}
    status_code = status_map.get(status_text.upper(), 3)

    payload = {
        "validatecp_uniq_code": kode_unik_qr,
        "validatecp_qr_code": qr_data,
        "validatecp_barcode_item": barcode_item,
        "validatecp_part": part_name,
        "validatecp_location": location,
        "validatecp_original_status": status_code,
    }

    # >>> Tambahan log
    print(f"\n--- PATCH ValidateCP ---")
    print(f"URL     : {url}")
    print(f"Payload : {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.patch(url, json=payload, auth=_auth, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"❌ Error PATCH ValidateCP: {e}")
        return None

    return resp
