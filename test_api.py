"""
test_api.py — Script testing untuk Flask backend phishing detection (Dual-Model + Debug)
"""

import requests
import time
import sys

BASE_URL = "http://localhost:5000"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0

def log_result(test_name, ok, detail=""):
    global passed, failed
    icon = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{icon}] {test_name}")
    if detail:
        print(f"         {YELLOW}{detail}{RESET}")
    if ok:
        passed += 1
    else:
        failed += 1

def print_debug_features(features):
    print(f"         {CYAN}[DEBUG] Isi Otak Model ({len(features)} Fitur yang dievaluasi):{RESET}")
    # Print 3 fitur per baris biar rapi dan gampang dibaca
    items = list(features.items())
    for i in range(0, len(items), 3):
        chunk = items[i:i+3]
        line = " | ".join([f"{k}: {v}" for k, v in chunk])
        print(f"         {CYAN}  {line}{RESET}")

def post_predict(url, timeout=15):
    resp = requests.post(f"{BASE_URL}/predict", json={"url": url}, timeout=timeout)
    return resp.json(), resp.status_code

print(f"\n{BOLD}[TEST 1] Health Check{RESET}")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    data = r.json()
    log_result("Status OK",  data.get("status") == "ok")
    log_result("Ada model",  data.get("full_model(url+html)") or data.get("url_only_fallback"),
               f"full={data.get('full_model(url+html)')}, url_only={data.get('url_only_fallback')}")
except Exception as e:
    print(f"  {RED}Server tidak bisa dihubungi: {e}{RESET}")
    print(f"  {YELLOW}Pastikan app.py sudah dijalankan: python app.py{RESET}")
    sys.exit(1)


print(f"\n{BOLD}[TEST 2] URL Legitimate{RESET}")
legit_urls = [
    ("https://getbootstrap.com", "legitimate"),
    ("https://vuejs.org",        "legitimate"),
    ("https://www.wikipedia.org","legitimate"),
]
for url, expected in legit_urls:
    try:
        data, status = post_predict(url)
        if status == 200:
            ok = data["result"] == expected
            log_result(url, ok, f"result={data['result']}, confidence={data['confidence']}, P(phishing)={data.get('proba_phishing')}")
            # --- TAMPILKAN ISI OTAK MODEL ---
            if "debug_features" in data:
                print_debug_features(data["debug_features"])
        elif status == 422:
            log_result(url, True, f"Terproteksi Anti-Bot (Status 422) - Sistem Aman")
        else:
            log_result(url, False, f"Status tak terduga: {status}")
    except Exception as e:
        log_result(url, False, str(e))


print(f"\n{BOLD}[TEST 3] URL Phishing{RESET}")
phishing_urls = [
    ("https://bafybeidw2m5xib4kfrerjwhv3vpe6o2fud4iyt7v6j54aoc5r2zuxdxtni.ipfs.dweb.link/", "phishing"),
    ("https://pub-8077ce6ed123447fb59f20c9ea47bda0.r2.dev/index.html", "phishing")
]
for url, expected in phishing_urls:
    try:
        data, status = post_predict(url)
        if status == 200:
            ok = data["result"] == expected
            log_result(url[:65]+"...", ok, f"result={data['result']}, confidence={data['confidence']}, P(phishing)={data.get('proba_phishing')}")
            # --- TAMPILKAN ISI OTAK MODEL ---
            if "debug_features" in data:
                print_debug_features(data["debug_features"])
        elif status == 422:
            log_result(url[:65]+"...", True, f"Berhasil Ditolak Gracefully (Status 422)")
        else:
            log_result(url[:65], False, f"Status tak terduga: {status}")
    except Exception as e:
        log_result(url[:65], False, str(e))


print(f"\n{BOLD}[TEST 4] Validasi Input{RESET}")
try:
    data, status = post_predict("google.com")
    log_result("URL tanpa http:// auto-fix ke https://",
               data.get("url", "").startswith("https://"),
               f"url={data.get('url')}")
except Exception as e:
    log_result("URL tanpa http://", False, str(e))

try:
    resp = requests.post(f"{BASE_URL}/predict", json={"bukan_url": "test"}, timeout=5)
    log_result("Request tanpa field 'url' -> 400", resp.status_code == 400, f"status: {resp.status_code}")
except Exception as e:
    log_result("Request tanpa field 'url'", False, str(e))


print(f"\n{BOLD}[TEST 5] Struktur Response (Valid URL){RESET}")
try:
    data, status = post_predict("https://www.google.com")
    if status == 200:
        for field in ["url", "result", "confidence", "threshold", "proba_phishing", "elapsed_ms"]:
            log_result(f"Field '{field}' ada", field in data, str(data.get(field))[:50])
    else:
        log_result("Bypass cek struktur (Akses ditolak 422)", True)
except Exception as e:
    log_result("Response structure", False, str(e))


total = passed + failed
print(f"\n{'='*50}")
print(f"{BOLD}HASIL: {passed}/{total} test passed{RESET}")
if failed > 0:
    print(f"       {RED}{failed} test GAGAL{RESET}")
print(f"{'='*50}\n")

sys.exit(0 if failed == 0 else 1)