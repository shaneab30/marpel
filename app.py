"""
app.py — Flask Backend: Phishing URL Detection (Arsitektur ADAPTIF)
====================================================================
Sesuai desain skripsi:
  - SCRAPE HTML dulu. Kalau halaman bisa diakses -> pakai model FULL (URL + HTML).
  - Kalau scrape GAGAL (situs mati / blokir bot) -> fallback ke model URL-only.

Anti train-serving skew:
  - Jalur FULL  : fitur dihitung oleh phiusiil_live.py (URL + HTML) -> sama dgn
                  fitur saat training skripsi_html.ipynb. Model HTML dilatih hanya
                  dengan fitur konten STABIL (fitur rapuh seperti NoOfCSS dibuang).
  - Jalur URL   : fitur dihitung oleh url_features.py -> sama dgn skripsi.ipynb.
Label: 0 = Phishing, 1 = Legitimate.
"""

import os
import time
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from url_features import (
    extract_url_features, normalize_url, extract_registered_domain,
    FEATURE_ORDER as URL_FEATURES,
)
from phiusiil_live import scrape_html_features, PHIUSIIL_FEATURES

DEFAULT_THRESHOLD = 0.50
# Pakai headless browser (Playwright) untuk scrape jalur FULL -> tahan situs JS/anti-bot.
# Set False kalau Playwright belum diinstall (otomatis fallback ke requests juga).
USE_BROWSER = True

# ── ALLOWLIST domain tepercaya ───────────────────────────────────────────────
# Domain populer langsung dianggap legitimate TANPA scrape/model. Alasan: situs
# besar (JS-berat / anti-bot) tak bisa di-scrape konsisten -> model bisa salah.
# Ini lapisan reputasi sederhana (mirip praktik Safe Browsing). Tambah domain di
# file 'allowlist.txt' (satu domain per baris) bila perlu.
ALLOWLIST = {
    "google.com", "youtube.com", "github.com", "wikipedia.org", "amazon.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "microsoft.com", "apple.com", "netflix.com", "whatsapp.com", "tiktok.com",
    "paypal.com", "yahoo.com", "reddit.com", "bing.com", "live.com",
    "cloudflare.com", "wordpress.com", "gmail.com", "google.co.id", "tokopedia.com",
    "shopee.co.id", "bca.co.id", "bankmandiri.co.id", "bri.co.id", "gojek.com",
}
if os.path.exists("allowlist.txt"):
    with open("allowlist.txt", encoding="utf-8") as _fh:
        ALLOWLIST |= {ln.strip().lower() for ln in _fh if ln.strip() and not ln.startswith("#")}

# Model URL-only (fallback) — dari skripsi.ipynb
URL_MODEL_PATH  = "model_xgboost_url.pkl"
URL_SCHEMA_PATH = "url_feature_schema.pkl"
URL_THR_PATH    = "decision_threshold.pkl"

# Model FULL (URL+HTML) — dari skripsi_html.ipynb
FULL_MODEL_PATH  = "model_xgboost_dataset.pkl"
FULL_SCHEMA_PATH = "dataset_feature_schema.pkl"
FULL_THR_PATH    = "dataset_threshold.pkl"

app = Flask(__name__)
CORS(app)


def _load(model_path, schema_path, thr_path, label):
    if not os.path.exists(model_path):
        print(f"[{label}] {model_path} tidak ada -> dilewati.")
        return None, None, DEFAULT_THRESHOLD
    model = joblib.load(model_path)
    schema = list(joblib.load(schema_path))
    thr = float(joblib.load(thr_path)) if os.path.exists(thr_path) else DEFAULT_THRESHOLD
    print(f"[{label}] siap: {len(schema)} fitur, threshold={thr:.3f}")
    return model, schema, thr


print("Loading models...")
url_model, url_schema, url_thr = _load(URL_MODEL_PATH, URL_SCHEMA_PATH, URL_THR_PATH, "URL-only")
full_model, full_schema, full_thr = _load(FULL_MODEL_PATH, FULL_SCHEMA_PATH, FULL_THR_PATH, "FULL(URL+HTML)")

if url_model is None and full_model is None:
    print("GAGAL: tidak ada model termuat. Latih dulu di notebook lalu export.")


def _proba_phishing(model, df):
    proba = model.predict_proba(df)[0]
    cls = list(model.classes_)
    return float(proba[cls.index(0)])     # kelas 0 = phishing


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.time()
    body = request.get_json(silent=True)
    if not body or "url" not in body:
        return jsonify({"error": "Request body harus berisi field 'url'"}), 400

    raw = body["url"].strip()
    if not raw:
        return jsonify({"error": "URL kosong"}), 400
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    # --- 0) ALLOWLIST: domain tepercaya -> langsung legitimate (skip scrape & model) ---
    reg_domain = extract_registered_domain(raw)
    if reg_domain in ALLOWLIST:
        return jsonify({
            "url": raw, "result": "legitimate", "confidence": 1.0,
            "proba_phishing": 0.0, "mode": "allowlist", "scrape_ok": False,
            "domain": reg_domain, "elapsed_ms": round((time.time() - t_start) * 1000, 1),
        })

    # --- 1) Coba jalur FULL: fitur URL (url_features) + fitur HTML (scrape) ---
    url_feats = extract_url_features(raw)
    html_feats = scrape_html_features(raw, render=USE_BROWSER) if full_model is not None else None

    if full_model is not None and html_feats is not None:
        df = pd.DataFrame([{**url_feats, **html_feats}])[full_schema]
        model, thr, mode, scrape_ok = full_model, full_thr, "full", True
    elif url_model is not None:
        # --- 2) Fallback URL-only ---
        df = pd.DataFrame([url_feats])[url_schema]
        model, thr, mode, scrape_ok = url_model, url_thr, "url_only", False
    else:
        return jsonify({"error": "Model belum termuat di server"}), 503

    proba_phishing = _proba_phishing(model, df)
    result = "phishing" if proba_phishing >= thr else "legitimate"
    elapsed = round((time.time() - t_start) * 1000, 1)

    return jsonify({
        "url"           : raw,
        "result"        : result,
        "confidence"    : round(proba_phishing if result == "phishing" else 1 - proba_phishing, 4),
        "proba_phishing": round(proba_phishing, 4),
        "threshold"     : round(thr, 4),
        "mode"          : mode,            # 'full' (URL+HTML) atau 'url_only' (fallback)
        "scrape_ok"     : scrape_ok,
        "elapsed_ms"    : elapsed,
        "debug_features": df.to_dict(orient="records")[0],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if (url_model is not None or full_model is not None) else "model_not_loaded",
        "full_model(url+html)": full_model is not None,
        "url_only_fallback": url_model is not None,
        "allowlist_domains": len(ALLOWLIST),
        "url_features": len(URL_FEATURES),
        "live_features": len(PHIUSIIL_FEATURES),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)